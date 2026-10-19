// The mountains kit's layer (`kits/mountains.rs`): the same function in the same order of
// operations. Fields, zones and fans: `mountains_relief.wgsl`.

fn mt_strata_col(i: u32) -> vec3<f32> {
    switch i {
        case 0u: { return srgb(196.0, 176.0, 154.0); }
        case 1u: { return srgb(178.0, 135.0, 108.0); }
        case 2u: { return srgb(202.0, 178.0, 146.0); }
        case 3u: { return srgb(178.0, 149.0, 125.0); }
        case 4u: { return srgb(186.0, 145.0, 120.0); }
        case 5u: { return srgb(188.0, 165.0, 140.0); }
        case 6u: { return srgb(198.0, 140.0, 96.0); }
        case 7u: { return srgb(188.0, 116.0, 76.0); }
        case 8u: { return srgb(176.0, 98.0, 64.0); }
        case 9u: { return srgb(150.0, 78.0, 56.0); }
        case 10u: { return srgb(164.0, 92.0, 64.0); }
        default: { return srgb(132.0, 72.0, 54.0); }
    }
}

/// `mountains::linear`.
fn mountains_slot_linear(s: ptr<function, Stack>) {
    let t = (*s).t;
    let ground = (*s).l.ground;
    let slope = (*s).l.slope;
    if (ground < 0.5) {
        return;
    }
    let p = (*s).c.p;
    let fl = mt_fields((*s).c);
    var zi: MtZIn;
    zi.moist = t.moist;
    zi.temp = t.temp;
    zi.mountain = t.mountain;
    zi.uplift = t.uplift;
    zi.cont = t.cont;
    zi.coast_km = t.coast_km;
    zi.sand = t.sand;
    zi.style = 0.0;
    let z = mt_zones(zi, fl.w, fl.n1, fl.n2);
    let rn = (*s).pf.f[PF_PATCH];
    let arid = 1.0 - smoothstep1(0.16, 0.34, t.moist);
    let soft = fl.w.x + fl.w.y;
    let clay = z.badland * 0.85;
    if (clay > 0.0) {
        let col = mixc(srgb(196.0, 184.0, 164.0), srgb(176.0, 160.0, 138.0), 0.5 + 0.5 * rn) * (1.0 + 0.05 * (*s).pf.f[PF_DETAIL]);
        var ly = layer_paint(clay, col, LC_BADLANDS);
        ly.clear = 1.0;
        composite(s, ly);
        (*s).m.natural_ok *= 1.0 - clay;
    }
    let wall = smoothstep1(0.42, 0.7, slope) * arid * soft;
    let bad = z.badland * smoothstep1(0.2, 0.45, slope);
    let cb = max(bad, wall);
    if (cb > 0.0) {
        var bandh = 34.0;
        var cl = LC_CLIFF;
        if (z.badland > 0.2) {
            bandh = 9.0;
            cl = LC_BADLANDS;
        }
        var kb = floor(ground / bandh + 0.6 * rn);
        var off = 0u;
        var nc = 5.999;
        var floor_col = srgb(190.0, 176.0, 150.0);
        if (z.badland <= 0.2) {
            kb = strata(ground - 3.0, MT_STRATUM).x;
            off = 9u;
            if (u01k(hash1(0xBE4Clu, i64(kb)), 2lu) < 0.55) {
                off = 6u;
            }
            nc = 2.999;
            floor_col = srgb(172.0, 128.0, 96.0);
        }
        let h = hash1(MT_SEED + 40lu, i64(kb));
        let col0 = mt_strata_col(off + u32(u01k(h, 1lu) * nc)) * (0.94 + 0.12 * u01k(h, 2lu));
        var cliff = smoothstep1(0.7, 1.3, slope);
        if (z.badland > 0.2) {
            cliff = 0.0;
        }
        var col = mixc(floor_col, col0, 0.8 * smoothstep1(0.25, 0.6, slope));
        if (z.badland <= 0.2) {
            let gsd = (*s).c.gsd;
            let face = col0 * (1.0 - 0.2 * cliff) * (1.0 + 0.1 * perlin3(MT_SEED + 41lu, p / 7.0lf) * band(7.0, gsd));
            let tal = smoothstep1(0.3, 0.5, slope) * (1.0 - smoothstep1(0.75, 1.05, slope));
            let talus = srgb(176.0, 146.0, 118.0) * (1.0 + 0.16 * perlin3(MT_SEED + 42lu, p / 3.5lf) * band(3.5, gsd) + 0.1 * perlin3(MT_SEED + 43lu, p / 11.0lf) * band(11.0, gsd));
            col = mixc(mixc(floor_col, face, 0.85 * smoothstep1(0.25, 0.6, slope)), talus, 0.7 * tal);
        }
        var ly = layer_paint(cb, col, cl);
        ly.clear = 1.0;
        composite(s, ly);
        (*s).m.natural_ok *= 1.0 - cb;
    }
    let cf = z.tower * smoothstep1(0.3, 0.7, (*s).m.rock) * (1.0 - smoothstep1(3.5, 6.0, slope + 0.8 * rn));
    if (cf > 0.0) {
        let col = mixc(srgb(40.0, 60.0, 36.0), srgb(64.0, 86.0, 48.0), 0.5 + 0.5 * rn) * (1.0 + 0.15 * (*s).pf.f[PF_DETAIL]);
        composite(s, layer_paint(cf, col, LC_BROADLEAF_FOREST));
    }
    let ck = min(z.tower * smoothstep1(3.5, 6.0, slope + 0.8 * rn) + z.cockpit * smoothstep1(0.8, 1.4, slope + 0.2 * rn), 1.0);
    if (ck > 0.0) {
        let col = mixc(srgb(168.0, 164.0, 150.0), srgb(120.0, 118.0, 108.0), 0.5 + 0.5 * rn) * (1.0 + 0.06 * (*s).pf.f[PF_DETAIL]);
        var ly = layer_paint(ck, col, LC_CLIFF);
        ly.clear = 1.0;
        composite(s, ly);
        (*s).m.natural_ok *= 1.0 - ck;
    }
    let glac = smoothstep1(0.3, 0.6, t.glaciation) * smoothstep1(0.25, 0.55, t.mountain);
    let cg = glac * smoothstep1(0.75, 1.1, slope);
    if (cg > 0.0) {
        let col = mixc(srgb(176.0, 174.0, 168.0), srgb(132.0, 130.0, 128.0), 0.5 + 0.5 * rn);
        var ly = layer_paint(cg, col, LC_ROCK);
        ly.clear = 1.0;
        composite(s, ly);
        (*s).m.natural_ok *= 1.0 - cg;
    }
    let blk = (*s).l.blk;
    let cnt = inst_count(FAM_MOUNTAINS_FANS, blk);
    for (var i = 0u; i < cnt; i++) {
        let it = inst_get(FAM_MOUNTAINS_FANS, blk, i);
        let fa = mt_fan_at(it, p, (*s).c.east, (*s).c.north);
        if (fa.ok) {
            let streak = 0.5 + 0.5 * sin(18.0 * fa.th + 6.0 * rn);
            let v = fa.sec * sqrt(1.0 - fa.rn) - 0.1 + 0.05 * (*s).pf.f[PF_DETAIL];
            let c = smoothstep1(0.0, 0.05, v);
            if (c > 0.0) {
                let col = mixc(srgb(206.0, 186.0, 152.0), srgb(176.0, 152.0, 122.0), 0.7 * streak + 0.2 * fa.rn);
                var ly = layer_paint(c, col, LC_GRAVEL);
                ly.clear = 0.85;
                composite(s, ly);
                (*s).m.natural_ok *= 1.0 - c;
            }
            break;
        }
    }
}
