// The mountains kit's layer (`kits/mountains.rs`): the same function in the same order of
// operations. Fields, zones and fans: `mountains_relief.wgsl`.

fn mt_strata_col(i: u32) -> vec3<f32> {
    switch i {
        case 0u: { return srgb(196.0, 188.0, 172.0); }
        case 1u: { return srgb(150.0, 92.0, 66.0); }
        case 2u: { return srgb(214.0, 198.0, 160.0); }
        case 3u: { return srgb(108.0, 86.0, 72.0); }
        case 4u: { return srgb(170.0, 112.0, 94.0); }
        default: { return srgb(176.0, 160.0, 140.0); }
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
    let gsd = (*s).c.gsd;
    let fl = mt_fields((*s).c);
    var zi: MtZIn;
    zi.moist = t.moist;
    zi.temp = t.temp;
    zi.mountain = t.mountain;
    zi.uplift = t.uplift;
    zi.cont = t.cont;
    zi.coast_km = t.coast_km;
    zi.sand = t.sand;
    let z = mt_zones(zi, fl.lit, fl.n1, fl.n2);
    let rn = (*s).pf.f[PF_PATCH];
    let arid = 1.0 - smoothstep1(0.16, 0.34, t.moist);
    let soft = f32(fl.lit == MT_SEDIMENTARY || fl.lit == MT_CARBONATE || fl.lit == MT_UNCONSOLIDATED);
    let wall = smoothstep1(0.42, 0.7, slope) * arid * soft;
    let bad = z.badland * smoothstep1(0.08, 0.25, slope);
    let cb = max(bad, wall);
    if (cb > 0.0) {
        var bandh = 14.0;
        var cl = LC_CLIFF;
        if (z.badland > 0.2) {
            bandh = 3.4;
            cl = LC_BADLANDS;
        }
        let kb = floor(ground / bandh + 0.6 * rn);
        let h = hash1(MT_SEED + 40lu, i64(kb));
        let col0 = mt_strata_col(u32(u01k(h, 1lu) * 5.999)) * (0.9 + 0.2 * u01k(h, 2lu));
        let col = mixc(srgb(190.0, 176.0, 150.0), col0, smoothstep1(0.1, 0.3, slope));
        var ly = layer_paint(cb, col, cl);
        ly.clear = 1.0;
        composite(s, ly);
        (*s).m.natural_ok *= 1.0 - cb;
    }
    let kz = min(z.tower + z.cockpit, 1.0);
    let ck = kz * smoothstep1(0.7, 1.1, slope);
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
