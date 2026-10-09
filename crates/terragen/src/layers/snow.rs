//! Slot 9, snow cover: a snow line in altitude (temperature) with crisp, ragged edges,
//! lingering in gullies and hollows. Its mask is known first (natural land use ends at it); it is
//! drawn on top of everything but water and the roads.

use crate::landcover as lc;
use crate::noise::*;
use crate::stack::{Layer, Stack};

/// The snow mask (before the other layers).
pub fn prepare(s: &mut Stack) {
    let t = s.t;
    let p = s.ctx.p;
    let gsd = s.ctx.gsd;
    // (a transition over several degrees looked like cloud or fog lying on the land; a strong
    // noise term drew blobs unrelated to the land)
    let snow_t = t.temp + 1.0 * s.pf.snow - 1.6 * smoothstep(0.1, 0.8, -t.gully)
        + 0.6 * smoothstep(0.2, 0.8, t.gully)
        + 0.5 * perlin3(0x5E0, p / 60.0) * band(60.0, gsd)
        + 0.25 * perlin3(0x5E1, p / 18.0) * band(18.0, gsd);
    s.m.snow = smoothstep(-2.6, -2.8, snow_t) * (1.0 - 0.75 * smoothstep(0.9, 1.6, s.l.slope));
}

pub fn layer(s: &mut Stack) {
    let snow = s.m.snow * (1.0 - s.m.river_cov) * (1.0 - s.m.road_cov);
    if snow > 0.0 {
        let c = s.sm.pal.snow * (1.0 + 0.03 * s.pf.detail);
        s.composite(Layer::paint_cls(snow, c, lc::SNOW));
    }
}
