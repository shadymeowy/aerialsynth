//! Slot 6, linear infrastructure (core): the iso-line road networks where people live and farm
//! tracks along the borders of land-use regions.

use crate::landcover as lc;
use crate::noise::*;
use crate::stack::{hmode, Layer, Stack};
use crate::surface::{band_cov, mixc};

pub fn layer(s: &mut Stack) {
    let world = s.world;
    let roads = world.cfg.landuse.roads;
    if roads <= 0.0 {
        return;
    }
    let pal = &s.sm.pal;
    let t = s.t;
    let l = s.l;
    let gsd = s.ctx.gsd;
    let fw = l.fw;
    let slope = l.slope;
    // a road is there or not (crisp cutoffs): roads fading with habitation or slope were
    // half-transparent ghosts with the trees showing through
    let steep = smoothstep(-0.02, 0.02, 0.5 - slope);
    // (fewer roads where development is low: the threshold rises)
    let habit = smoothstep(-0.005, 0.005, t.habit - 0.03 - 0.05 * (1.0 - t.development)) * steep * (1.0 - s.m.snow) * (1.0 - t.sand * 0.7);
    let mut road_cov: f64 = 0.0;
    let mut road_col = pal.asphalt;
    if habit > 0.0 {
        let w_major = 12.0;
        let c1 = band_cov(l.road_major, w_major * 0.5, fw.max(gsd * 0.5));
        if c1 > 0.0 {
            road_cov = c1 * habit;
            s.m.road_major_cov = road_cov;
            // lighter shoulders
            let sh = band_cov(l.road_major, w_major * 0.5 + 1.5, fw) - band_cov(l.road_major, w_major * 0.5, fw);
            road_col = mixc(pal.asphalt, pal.concrete, sh.max(0.0) * 0.6);
        }
        let w_minor = 6.0;
        let c2 = band_cov(l.road_minor, w_minor * 0.5, fw.max(gsd * 0.5)) * habit;
        if c2 > road_cov {
            road_cov = c2;
            road_col = mixc(pal.asphalt, pal.gravel, smoothstep(0.4, 0.7, t.style[0]));
        }
    }
    // farm tracks along land-use region borders
    if let Some(r) = &s.region {
        if r.agri > 0.1 && s.agri() > 0.05 {
            let pair = t.region.id ^ t.region.id2;
            if u01k(pair, 3) < 0.7 {
                let c3 = band_cov(t.region.edge, 3.0, fw.max(gsd * 0.5)) * steep;
                if c3 > road_cov {
                    road_cov = c3;
                    road_col = if u01k(pair, 4) < 0.5 { pal.gravel } else { pal.asphalt };
                }
            }
        }
    }
    if road_cov > 0.0 {
        s.m.road_cov = road_cov;
        let rc = road_col * (1.0 + 0.05 * s.pf.detail);
        s.composite(Layer { cov: road_cov, albedo: rc, dh: 0.0, hmode: hmode::BLEND, cls: lc::ROAD, relit: 0.5, ..Default::default() });
    }
}
