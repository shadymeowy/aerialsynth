//! Event sensor on the GPU: the per-pixel model of `EventSensor::step` (photoreceptor low-pass,
//! high-pass, leak, threshold crossings with refractory period, shot noise) runs in a compute
//! pass (`events.wgsl`) with the pixel state resident on the GPU. Keyframes (radiance and the
//! lamp-flicker split) are uploaded once; the sensor steps between two keyframes are interpolated
//! on the GPU and run in one submission, and only the events come back. Hot pixels, timestamp
//! jitter, sorting and the rate controller run on the CPU (`EventSensor::finish_step`).

use super::device::{self, Gpu};
use crate::events::{Event, EventSensor};
use crate::raster::FrameOut;
use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct StepP {
    n: u32,
    mode: u32,
    step: u32,
    seed: u32,
    seed2: u32,
    split0: u32,
    split1: u32,
    _p: u32,
    a: f32,
    c: f32,
    s: f32,
    dt: f32,
    gain: f32,
    eps: f32,
    cut_hz: f32,
    cut_half: f32,
    cut_min: f32,
    a_hpf: f32,
    refr: f32,
    shot_hz: f32,
    noise_ref: f32,
    dark_lo: f32,
    dark_hi: f32,
    _q: f32,
}

/// uniform offset alignment
const PSTRIDE: u64 = 256;
/// bytes per event record: pixel | polarity << 31, step fraction (f32 bits), step index
const REC: u64 = 12;

struct Queued {
    p: StepP,
    k0: usize,
    /// (t0, t) of a sensor step; None for the initialisation
    span: Option<(f64, f64)>,
}

pub struct GpuEventSensor {
    gpu: Arc<Gpu>,
    /// configuration, fixed pattern, hot pixels, RNG and clock (shared code with the CPU sensor)
    cpu: EventSensor,
    n: u32,
    pipe: wgpu::ComputePipeline,
    state: wgpu::Buffer,
    backup: wgpu::Buffer,
    consts: wgpu::Buffer,
    keys: [wgpu::Buffer; 2],
    split: [bool; 2],
    ev: wgpu::Buffer,
    ev_read: wgpu::Buffer,
    count_read: wgpu::Buffer,
    /// event capacity (records)
    cap: u64,
    params: wgpu::Buffer,
    params_cap: usize,
    queue: Vec<Queued>,
    t_last: Option<f64>,
    slot: usize,
}

fn buffer(d: &wgpu::Device, label: &str, size: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
    d.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: size.max(16), usage, mapped_at_creation: false })
}

impl GpuEventSensor {
    pub fn new(cpu: EventSensor) -> Result<Self> {
        let gpu = device::shared()?;
        let d = &gpu.device;
        let n = cpu.px.len() as u32;
        let module = d
            .create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("events"), source: wgpu::ShaderSource::Wgsl(include_str!("events.wgsl").into()) });
        let pipe = d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("events"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        use wgpu::BufferUsages as U;
        let st = U::STORAGE | U::COPY_SRC | U::COPY_DST;
        let state = buffer(d, "ev state", n as u64 * 16, st);
        let backup = buffer(d, "ev state backup", n as u64 * 16, st);
        let consts = buffer(d, "ev consts", n as u64 * 16, U::STORAGE | U::COPY_DST);
        let cv: Vec<[f32; 4]> = cpu.px.iter().map(|p| [p.c_pos, p.c_neg, p.leak, 0.0]).collect();
        gpu.queue.write_buffer(&consts, 0, bytemuck::cast_slice(&cv));
        let key = |l| buffer(d, l, n as u64 * 36, U::STORAGE | U::COPY_DST);
        let keys = [key("ev key 0"), key("ev key 1")];
        let cap = (n as u64 * 2).max(1 << 16);
        let ev = buffer(d, "events", 4 + cap * REC, U::STORAGE | U::COPY_SRC | U::COPY_DST);
        let ev_read = buffer(d, "events r", 4 + cap * REC, U::MAP_READ | U::COPY_DST);
        let count_read = buffer(d, "events n", 4, U::MAP_READ | U::COPY_DST);
        let params_cap = 16;
        let params = buffer(d, "ev params", params_cap as u64 * PSTRIDE, U::UNIFORM | U::COPY_DST);
        Ok(GpuEventSensor {
            gpu,
            cpu,
            n,
            pipe,
            state,
            backup,
            consts,
            keys,
            split: [false; 2],
            ev,
            ev_read,
            count_read,
            cap,
            params,
            params_cap,
            queue: vec![],
            t_last: None,
            slot: 0,
        })
    }

    /// Slot of the current keyframe (bookkeeping for the caller).
    pub fn key_slot(&self) -> usize {
        self.slot
    }

    pub fn set_key_slot(&mut self, slot: usize) {
        self.slot = slot;
    }

    /// Upload a keyframe into slot 0 or 1 (radiance and, when split, flicker cos / sin).
    pub fn set_key(&mut self, slot: usize, f: &FrameOut) {
        self.set_key_raw(slot, &f.radiance, &f.flicker_cos, &f.flicker_sin);
    }

    fn set_key_raw(&mut self, slot: usize, rad: &[f32], fc: &[f32], fs: &[f32]) {
        let q = &self.gpu.queue;
        let n3 = self.n as u64 * 3 * 4;
        q.write_buffer(&self.keys[slot], 0, bytemuck::cast_slice(rad));
        self.split[slot] = !fc.is_empty();
        if self.split[slot] {
            q.write_buffer(&self.keys[slot], n3, bytemuck::cast_slice(fc));
            q.write_buffer(&self.keys[slot], 2 * n3, bytemuck::cast_slice(fs));
        }
    }

    /// Queue a sensor step at time `t` (s) with the radiance `(1 - a) · key[k0] + a · key[1 - k0]`
    /// at flicker phase omega · t. The first step initialises the pixels.
    pub fn push(&mut self, t: f64, k0: usize, a: f32, omega: f64) {
        let c = &self.cpu.cfg;
        let mut p = StepP {
            n: self.n,
            split0: self.split[k0] as u32,
            split1: self.split[1 - k0] as u32,
            a,
            c: (omega * t).cos() as f32,
            s: (omega * t).sin() as f32,
            gain: c.gain as f32,
            eps: c.log_eps as f32,
            ..Default::default()
        };
        let Some(t0) = self.t_last else {
            self.t_last = Some(t);
            self.queue.push(Queued { p, k0, span: None });
            return;
        };
        let dt = t - t0;
        if dt <= 0.0 {
            return;
        }
        self.cpu.n_steps += 1;
        let step_seed = c.seed ^ self.cpu.n_steps.wrapping_mul(0xA24B_AED4_963E_E407);
        p.mode = 1;
        p.step = self.queue.len() as u32;
        p.seed = step_seed as u32;
        p.seed2 = (step_seed >> 32) as u32;
        p.dt = dt as f32;
        p.cut_hz = c.cutoff_hz as f32;
        p.cut_half = c.cutoff_half_lum as f32;
        p.cut_min = c.cutoff_min_hz as f32;
        p.a_hpf = if c.hpf_hz > 0.0 { (1.0 - (-std::f64::consts::TAU * c.hpf_hz * dt).exp()) as f32 } else { 0.0 };
        p.refr = (c.refractory_us * 1e-6) as f32;
        p.shot_hz = c.shot_noise_hz as f32;
        p.noise_ref = c.noise_ref_lum as f32;
        p.dark_lo = (1.0 / c.shot_noise_dark_gain.max(1.0)) as f32;
        p.dark_hi = c.shot_noise_dark_gain as f32;
        self.t_last = Some(t);
        self.queue.push(Queued { p, k0, span: Some((t0, t)) });
    }

    /// Run the queued steps; returns their events, sorted.
    pub fn flush(&mut self) -> Result<Vec<Event>> {
        if self.queue.is_empty() {
            return Ok(vec![]);
        }
        let prof = std::env::var_os("RENDER_PROFILE").is_some();
        let c0 = std::time::Instant::now();
        let gpu = self.gpu.clone();
        let (d, q) = (&gpu.device, &gpu.queue);
        use wgpu::BufferUsages as U;
        if self.queue.len() > self.params_cap {
            self.params_cap = self.queue.len().next_power_of_two();
            self.params = buffer(d, "ev params", self.params_cap as u64 * PSTRIDE, U::UNIFORM | U::COPY_DST);
        }
        for (i, s) in self.queue.iter().enumerate() {
            q.write_buffer(&self.params, i as u64 * PSTRIDE, bytemuck::bytes_of(&s.p));
        }
        let wg = self.n.div_ceil(256);
        let (gx, gy) = (wg.min(65535), wg.div_ceil(65535));
        let mut first = true;
        let count = loop {
            let mut enc = d.create_command_encoder(&Default::default());
            if first {
                enc.copy_buffer_to_buffer(&self.state, 0, &self.backup, 0, self.n as u64 * 16);
            } else {
                // retry after an overflow: restart from the state before the batch
                enc.copy_buffer_to_buffer(&self.backup, 0, &self.state, 0, self.n as u64 * 16);
            }
            enc.clear_buffer(&self.ev, 0, Some(4));
            let layout = self.pipe.get_bind_group_layout(0);
            for (i, s) in self.queue.iter().enumerate() {
                let bg = d.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: &self.params,
                                offset: i as u64 * PSTRIDE,
                                size: wgpu::BufferSize::new(std::mem::size_of::<StepP>() as u64),
                            }),
                        },
                        wgpu::BindGroupEntry { binding: 1, resource: self.state.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: self.consts.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 3, resource: self.keys[s.k0].as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 4, resource: self.keys[1 - s.k0].as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 5, resource: self.ev.as_entire_binding() },
                    ],
                });
                {
                    let mut cp = enc.begin_compute_pass(&Default::default());
                    cp.set_pipeline(&self.pipe);
                    cp.set_bind_group(0, &bg, &[]);
                    cp.dispatch_workgroups(gx, gy, 1);
                }
                if std::env::var_os("AS_DEBUG_EV_SPLIT").is_some() {
                    q.submit([std::mem::replace(&mut enc, d.create_command_encoder(&Default::default())).finish()]);
                }
            }
            enc.copy_buffer_to_buffer(&self.ev, 0, &self.count_read, 0, 4);
            q.submit([enc.finish()]);
            self.gpu.map_read(&[self.count_read.slice(..)])?;
            let count = bytemuck::cast_slice::<u8, u32>(&self.count_read.slice(..).get_mapped_range()?)[0] as u64;
            self.count_read.unmap();
            first = false;
            if count <= self.cap {
                break count;
            }
            self.cap = count + count / 4;
            self.ev = buffer(d, "events", 4 + self.cap * REC, U::STORAGE | U::COPY_SRC | U::COPY_DST);
            self.ev_read = buffer(d, "events r", 4 + self.cap * REC, U::MAP_READ | U::COPY_DST);
        };
        let t_gpu = c0.elapsed().as_secs_f64();
        // per-step event lists
        let mut per: Vec<Vec<Event>> = vec![vec![]; self.queue.len()];
        if count > 0 {
            let bytes = 4 + count * REC;
            let mut enc = d.create_command_encoder(&Default::default());
            enc.copy_buffer_to_buffer(&self.ev, 0, &self.ev_read, 0, bytes);
            q.submit([enc.finish()]);
            self.gpu.map_read(&[self.ev_read.slice(..bytes)])?;
            {
                let map = self.ev_read.slice(..bytes).get_mapped_range()?;
                let rec: &[u32] = bytemuck::cast_slice(&map[4..]);
                let w = self.cpu.w as u32;
                for r in rec.as_chunks::<3>().0 {
                    let s = r[2] as usize;
                    let Some((t0, t)) = self.queue[s].span else { continue };
                    let idx = r[0] & 0x7FFF_FFFF;
                    let te = t0 + f32::from_bits(r[1]) as f64 * (t - t0);
                    per[s].push(Event { t_us: (te * 1e6).round() as i64, x: (idx % w) as u16, y: (idx / w) as u16, p: (r[0] >> 31) as i8 });
                }
            }
            self.ev_read.unmap();
        }
        let t_read = c0.elapsed().as_secs_f64();
        let nq = self.queue.len();
        let mut out = vec![];
        for (s, ev) in self.queue.drain(..).zip(per) {
            match s.span {
                Some((t0, t)) => {
                    // atomics append in arbitrary order: restore the CPU's (row-major) order, so that
                    // the stable time sort and the rate controller are deterministic
                    let mut ev = ev;
                    ev.sort_unstable_by_key(|e| (e.y, e.x, e.t_us));
                    out.extend(self.cpu.finish_step(t0, t, ev))
                }
                None => self.cpu.t_prev = self.t_last,
            }
        }
        if prof {
            let t = c0.elapsed().as_secs_f64();
            eprintln!(
                "[events gpu] {nq} steps, {count} events: gpu {:.1} ms, read-back {:.1} ms, cpu finish {:.1} ms",
                t_gpu * 1e3,
                (t_read - t_gpu) * 1e3,
                (t - t_read) * 1e3
            );
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::EventConfig;

    fn quiet() -> EventConfig {
        EventConfig {
            contrast_sigma: 0.0,
            shot_noise_hz: 0.0,
            leak_hz: 0.0,
            refractory_us: 0.0,
            hot_pixel_fraction: 0.0,
            timestamp_jitter_us: 0.0,
            ..Default::default()
        }
    }

    fn sensor(cfg: EventConfig, w: usize, h: usize) -> Option<GpuEventSensor> {
        match GpuEventSensor::new(EventSensor::new(cfg, w, h)) {
            Ok(s) => Some(s),
            Err(e) => {
                eprintln!("no GPU, skipped: {e}");
                None
            }
        }
    }

    /// GPU and CPU sensors fed the same images give the same events (deterministic config).
    #[test]
    fn moving_edge_matches_cpu() {
        let (w, h) = (32, 4);
        let cfg = EventConfig { cutoff_hz: 1e9, ..quiet() };
        let Some(mut g) = sensor(cfg.clone(), w, h) else { return };
        let mut c = EventSensor::new(cfg, w, h);
        let (mut eg, mut ec) = (vec![], vec![]);
        for k in 0..=100 {
            let edge = 5.0 + 10.0 * k as f64 / 100.0;
            let rad: Vec<f32> = (0..w * h).flat_map(|i| [if ((i % w) as f64 + 0.5) < edge { 1.0f32 } else { 0.25 }; 3]).collect();
            let t = k as f64 * 0.01;
            ec.extend(c.step(t, &c.log_image(&rad)));
            g.set_key_raw(0, &rad, &[], &[]);
            g.push(t, 0, 0.0, 0.0);
            eg.extend(g.flush().unwrap());
        }
        assert_eq!(eg.len(), 200);
        assert_eq!(eg.len(), ec.len());
        let key = |e: &Event| (e.t_us, e.y, e.x);
        eg.sort_by_key(key);
        ec.sort_by_key(key);
        for (a, b) in eg.iter().zip(&ec) {
            assert!((a.t_us - b.t_us).abs() <= 1 && a.x == b.x && a.y == b.y && a.p == b.p, "{a:?} {b:?}");
        }
    }

    /// Batched, interpolated steps with lamp flicker: same events as the CPU path.
    #[test]
    fn flicker_interpolation_matches_cpu() {
        let (w, h) = (64, 16);
        let n = w * h;
        let cfg = quiet();
        let Some(mut g) = sensor(cfg.clone(), w, h) else { return };
        let mut c = EventSensor::new(cfg, w, h);
        let omega = std::f64::consts::TAU * 100.0;
        let key = |k: usize| -> (Vec<f32>, Vec<f32>, Vec<f32>) {
            let r = (0..n * 3).map(|i| 0.05 + 0.3 * (((i / 3) * 7 + k * 13) % 17) as f32 / 17.0).collect();
            let fc = (0..n * 3).map(|i| if (i / 3) % 5 == 0 { 0.02 * (1 + k) as f32 } else { 0.0 }).collect();
            let fs = (0..n * 3).map(|i| if (i / 3) % 7 == 0 { 0.03 } else { 0.0 }).collect();
            (r, fc, fs)
        };
        let at = |k: &(Vec<f32>, Vec<f32>, Vec<f32>), t: f64| -> Vec<f32> {
            let (co, si) = ((omega * t).cos() as f32, (omega * t).sin() as f32);
            k.0.iter().zip(&k.1).zip(&k.2).map(|((r, a), b)| r + a * co + b * si).collect()
        };
        let (mut ng, mut nc) = (0, 0);
        let mut k0 = key(0);
        g.set_key_raw(0, &k0.0, &k0.1, &k0.2);
        g.push(0.0, 0, 0.0, omega);
        g.flush().unwrap();
        c.step(0.0, &c.log_image(&at(&k0, 0.0)));
        let mut slot = 0;
        for j in 1..=6 {
            let (t0, t1) = ((j - 1) as f64 * 0.01, j as f64 * 0.01);
            let k1 = key(j);
            g.set_key_raw(1 - slot, &k1.0, &k1.1, &k1.2);
            for i in 1..=12 {
                let t = t0 + (t1 - t0) * i as f64 / 12.0;
                let a = ((t - t0) / (t1 - t0)) as f32;
                let rad = if i == 12 {
                    g.push(t, 1 - slot, 0.0, omega);
                    at(&k1, t)
                } else {
                    g.push(t, slot, a, omega);
                    at(&k0, t).iter().zip(&at(&k1, t)).map(|(u, v)| u + a * (v - u)).collect()
                };
                nc += c.step(t, &c.log_image(&rad)).len();
            }
            let ev = g.flush().unwrap();
            assert!(ev.windows(2).all(|p| p[0].t_us <= p[1].t_us));
            ng += ev.len();
            slot = 1 - slot;
            k0 = k1;
        }
        eprintln!("FLICKER gpu {ng} cpu {nc} adapter {:?}", g.gpu.info);
        assert!(nc > 1000, "{nc}");
        assert!((ng as f64 - nc as f64).abs() <= 0.002 * nc as f64, "gpu {ng} cpu {nc}");
    }

    /// Noise statistics (shot noise, leak) agree with the CPU sensor.
    #[test]
    fn background_activity_matches_cpu() {
        let (w, h) = (200, 100);
        let base = EventConfig { hot_pixel_fraction: 0.0, max_rate_mev_s: 0.0, ..Default::default() };
        for lum in [0.5f32, 0.002] {
            let Some(mut g) = sensor(base.clone(), w, h) else { return };
            let mut c = EventSensor::new(base.clone(), w, h);
            let rad = vec![lum; w * h * 3];
            g.set_key_raw(0, &rad, &[], &[]);
            let l = c.log_image(&rad);
            let (mut ng, mut nc) = (0, 0);
            for k in 0..=200 {
                let t = k as f64 * 0.01;
                nc += c.step(t, &l).len();
                g.push(t, 0, 0.0, 0.0);
                if k % 20 == 0 {
                    ng += g.flush().unwrap().len();
                }
            }
            ng += g.flush().unwrap().len();
            let tol = 4.0 * (nc as f64).sqrt() + 0.02 * nc as f64;
            assert!((ng as f64 - nc as f64).abs() < tol, "lum {lum}: gpu {ng} cpu {nc}");
        }
    }

    /// Photoreceptor bandwidth: the first event after a brightening comes at the same time.
    #[test]
    fn low_light_bandwidth_matches_cpu() {
        let cfg = quiet();
        for lum0 in [0.5f32, 2e-7] {
            let Some(mut g) = sensor(cfg.clone(), 1, 1) else { return };
            g.set_key_raw(0, &[lum0; 3], &[], &[]);
            g.push(0.0, 0, 0.0, 0.0);
            g.flush().unwrap();
            g.set_key_raw(0, &[lum0 * 4.0; 3], &[], &[]);
            let mut first = f64::MAX;
            'outer: for b in 0..40 {
                for k in 1..=100 {
                    g.push((b * 100 + k) as f64 * 1e-4, 0, 0.0, 0.0);
                }
                if let Some(e) = g.flush().unwrap().first() {
                    first = e.t_us as f64 * 1e-6;
                    break 'outer;
                }
            }
            let mut c = EventSensor::new(cfg.clone(), 1, 1);
            c.step(0.0, &c.log_image(&[lum0; 3]));
            let mut fc = f64::MAX;
            for k in 1..=4000 {
                if let Some(e) = c.step(k as f64 * 1e-4, &c.log_image(&[lum0 * 4.0; 3])).first() {
                    fc = e.t_us as f64 * 1e-6;
                    break;
                }
            }
            assert!((first - fc).abs() <= 1e-4 + 1e-3 * fc, "lum {lum0}: gpu {first} cpu {fc}");
        }
    }
}
