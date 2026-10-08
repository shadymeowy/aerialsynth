//! The generator's noise instances as GPU tables: every fBm and octave frame set of `World` and
//! `SurfaceModel`, with each octave's rotation, offset and seed (from the CPU's own
//! `OctaveFrames`, so the GPU evaluates the same noise), its wavelength and amplitude. The WGSL
//! side names them with the constants of [`wgsl_consts`].

use crate::noise::{Fbm, OctaveFrames};
use crate::surface::SurfaceModel;
use crate::world::World;
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
pub(crate) struct Oct {
    ax: [f64; 4],
    ay: [f64; 4],
    az: [f64; 4],
    off: [f64; 4],
    lam: f64,
    inv_lam: f64,
    seed: u64,
    amp: f32,
    _p: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
pub(crate) struct FbmD {
    first: u32,
    n: u32,
    norm: f32,
    wavelength: f32,
}

/// One table entry: an fBm (wavelengths and amplitudes per octave) or a bare frame set (the
/// shader runs its own wavelength / amplitude schedule).
enum Src<'a> {
    Fbm(&'a Fbm),
    Frames(&'a OctaveFrames),
}

/// The noise instances in table order, with their WGSL names (`FBM_<name>`).
fn sources<'a>(w: &'a World, s: &'a SurfaceModel) -> Vec<(&'static str, Src<'a>)> {
    use Src::*;
    vec![
        ("CONT", Fbm(&w.cont)),
        ("CONT_WARP0", Fbm(&w.cont_warp[0])),
        ("CONT_WARP1", Fbm(&w.cont_warp[1])),
        ("CONT_WARP2", Fbm(&w.cont_warp[2])),
        ("BELT", Fbm(&w.belt)),
        ("BELT2", Fbm(&w.belt2)),
        ("BELT_VAR", Fbm(&w.belt_var)),
        ("MTN", Frames(&w.mtn_frames)),
        ("MTN_WARP0", Fbm(&w.mtn_warp[0])),
        ("MTN_WARP1", Fbm(&w.mtn_warp[1])),
        ("PLATEAU", Fbm(&w.plateau)),
        ("HILLS", Frames(&w.hills)),
        ("HILL_AMP", Fbm(&w.hill_amp)),
        ("ROUGH", Fbm(&w.rough)),
        ("MICRO", Fbm(&w.micro)),
        ("TEMP", Fbm(&w.temp_n)),
        ("MOIST", Fbm(&w.moist_n)),
        ("RIVER_WARP0", Fbm(&w.river_warp[0])),
        ("RIVER_WARP1", Fbm(&w.river_warp[1])),
        ("RIVER_WIDTH", Fbm(&w.river_width_n)),
        ("MESA", Fbm(&w.mesa_n)),
        ("DUNES", Frames(&w.dune_frames)),
        ("SAND", Fbm(&w.sand_n)),
        ("AGRI", Fbm(&w.agri_n)),
        ("STYLE0", Fbm(&w.style_n[0])),
        ("STYLE1", Fbm(&w.style_n[1])),
        ("STYLE2", Fbm(&w.style_n[2])),
        ("STYLE3", Fbm(&w.style_n[3])),
        ("ROAD_MAJOR", Fbm(&w.road_major)),
        ("ROAD_MINOR", Fbm(&w.road_minor)),
        ("DETAIL", Fbm(&s.detail)),
        ("PATCH", Fbm(&s.patch)),
        ("FOREST", Fbm(&s.forest)),
        ("FIELD_VAR", Fbm(&s.field_var)),
        ("WARP2", Fbm(&s.warp2)),
        ("STRATA", Fbm(&s.strata)),
        ("SNOW", Fbm(&s.snow_n)),
        ("CULT", Fbm(&s.cult_n)),
        ("LAND", Fbm(&s.land_n)),
    ]
}

fn oct(rot: &glam::DMat3, off: glam::DVec3, seed: u64, lam: f64, amp: f64) -> Oct {
    let c = |v: glam::DVec3| [v.x, v.y, v.z, 0.0];
    Oct { ax: c(rot.x_axis), ay: c(rot.y_axis), az: c(rot.z_axis), off: c(off), lam, inv_lam: 1.0 / lam, seed, amp: amp as f32, _p: 0.0 }
}

/// The octave and fBm tables.
pub(crate) fn build(w: &World, s: &SurfaceModel) -> (Vec<Oct>, Vec<FbmD>) {
    let mut octs = Vec::new();
    let mut fbms = Vec::new();
    for (_, src) in sources(w, s) {
        let first = octs.len() as u32;
        match src {
            Src::Fbm(f) => {
                // the same wavelength / amplitude sequence as `Fbm::eval`
                let (mut lam, mut amp) = (f.wavelength, 1.0);
                for i in 0..f.octaves {
                    octs.push(oct(&f.frames.rot[i], f.frames.off[i], f.frames.seeds[i], lam, amp));
                    lam /= f.lacunarity;
                    amp *= f.gain;
                }
                fbms.push(FbmD { first, n: f.octaves as u32, norm: f.norm() as f32, wavelength: f.wavelength as f32 });
            }
            Src::Frames(fr) => {
                for i in 0..fr.rot.len() {
                    octs.push(oct(&fr.rot[i], fr.off[i], fr.seeds[i], 1.0, 1.0));
                }
                fbms.push(FbmD { first, n: fr.rot.len() as u32, norm: 1.0, wavelength: 0.0 });
            }
        }
    }
    (octs, fbms)
}

/// `const FBM_<name>: u32 = <index>u;` for every noise instance.
pub(crate) fn wgsl_consts() -> String {
    // the names only (the instances of any world will do)
    let w = World::new(crate::Config::default());
    let s = SurfaceModel::new(&w);
    sources(&w, &s).iter().enumerate().map(|(i, (name, _))| format!("const FBM_{name}: u32 = {i}u;\n")).collect()
}

/// The gradient table of `perlin3`.
pub(crate) fn grads() -> Vec<[f32; 4]> {
    crate::noise::gradient_table().iter().map(|g| [g[0] as f32, g[1] as f32, g[2] as f32, 0.0]).collect()
}
