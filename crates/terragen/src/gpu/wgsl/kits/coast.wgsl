// The coast kit's layers (`kits/coast.rs`): the same functions in the same order of operations.
// Types, creeks and relief: `coast_relief.wgsl`.

/// `coast::azonal`.
fn coast_slot_azonal(s: ptr<function, Stack>) {
    let t = (*s).t;
    let ground = (*s).l.ground;
    if (ground > -0.3 && ground < 4.0 && t.coast_km < -50.0) {
        co_motus(s);
    }
    if (t.coast_km > 60.0 || t.coast_km < -40.0 || ground > 500.0 || ground < -1.0) {
        return;
    }
    let ty = co_types((*s).c.p, t.uplift, t.mountain, t.glaciation, t.temp);
    if (ty.cliff > 0.0) {
        co_cliff_face(s, ty);
    }
    if (ty.skerry > 0.0) {
        co_rocky_shore(s, ty);
    }
    if (ty.flat > 0.0 && ground < 3.6) {
        co_flats(s, ty);
    }
}

fn co_motus(s: ptr<function, Stack>) {
    let p = (*s).c.p;
    let gsd = (*s).c.gsd;
    let ground = (*s).l.ground;
    let blk = (*s).l.blk;
    let cnt = inst_count(FAM_COAST_ATOLLS, blk);
    for (var i = 0u; i < cnt; i++) {
        let a = co_atoll_at(inst_get(FAM_COAST_ATOLLS, blk, i), p, gsd);
        if (a.ok && abs(a.d) < a.wre + 20.0) {
            let n = co_nz(CO_SEED + 45lu, p, 14.0, gsd);
            var sy = layer_paint(1.0, srgb(236.0, 226.0, 194.0) * (1.0 + 0.04 * n), LC_BEACH);
            sy.clear = 1.0;
            composite(s, sy);
            let c = smoothstep1(0.7, 1.4, ground + 0.5 * n) * 0.85;
            if (c > 0.0) {
                let col = mixc(srgb(60.0, 104.0, 52.0), srgb(86.0, 120.0, 62.0), 0.5 + 0.5 * n);
                var ly = layer_paint(c, col, LC_SHRUB);
                ly.clear = 1.0;
                composite(s, ly);
            }
            return;
        }
    }
}

fn co_cliff_face(s: ptr<function, Stack>, ty: CoTypes) {
    let p = (*s).c.p;
    let gsd = (*s).c.gsd;
    let ground = (*s).l.ground;
    let slope = (*s).l.slope;
    let c = ty.cliff * smoothstep1(0.4, 0.75, slope) * (1.0 - smoothstep1(300.0, 500.0, ground));
    if (c <= 0.0) {
        return;
    }
    var base = srgb(150.0, 98.0, 72.0);
    let litho = (*s).bio.st.litho;
    if (litho == 1u) {
        base = srgb(206.0, 200.0, 182.0);
    } else if (litho == 2u) {
        base = srgb(112.0, 106.0, 100.0);
    } else if (litho == 3u) {
        base = srgb(56.0, 50.0, 48.0);
    } else if (litho == 4u) {
        base = srgb(176.0, 148.0, 112.0);
    }
    let st = 0.5 + 0.5 * sin(ground * TAU / 3.4 + 2.0 * co_nz(CO_SEED + 40lu, p, 60.0, gsd)) * band(3.4 / (0.1 + slope), gsd);
    let streak = co_nz(CO_SEED + 41lu, p, 18.0, gsd);
    let col = base * (0.86 + 0.2 * st + 0.1 * streak);
    var ly = layer_paint(c, col, LC_CLIFF);
    ly.clear = 1.0;
    composite(s, ly);
    (*s).m.natural_ok *= 1.0 - c;
}

fn co_rocky_shore(s: ptr<function, Stack>, ty: CoTypes) {
    let p = (*s).c.p;
    let gsd = (*s).c.gsd;
    let ground = (*s).l.ground;
    let m = ty.skerry * (1.0 - smoothstep1(4.0, 16.0, ground + 4.0 * co_nz(CO_SEED + 42lu, p, 120.0, gsd)));
    if (m <= 0.0) {
        return;
    }
    let rock = mixc(spal(s, BP_ROCK), spal(s, BP_ROCK + 1u), 0.5 + 0.5 * co_nz(CO_SEED + 43lu, p, 25.0, gsd));
    let lichen = 0.5 + 0.5 * co_nz(CO_SEED + 44lu, p, 6.0, gsd);
    let col = mixc(rock, srgb(104.0, 112.0, 80.0), 0.25 * lichen * smoothstep1(1.0, 6.0, ground)) * (1.0 + 0.16 * co_nz(CO_SEED + 46lu, p, 5.0, gsd) + 0.1 * co_nz(CO_SEED + 47lu, p, 1.6, gsd));
    var ly = layer_paint(m, col, LC_ROCK);
    ly.clear = 1.0;
    composite(s, ly);
    (*s).m.natural_ok *= 1.0 - m;
}

fn co_flats(s: ptr<function, Stack>, ty: CoTypes) {
    let p = (*s).c.p;
    let gsd = (*s).c.gsd;
    let fw = (*s).l.fw;
    let ground = (*s).l.ground;
    let wet = 1.0 - smoothstep1(0.15, 0.7, ground);
    let high = smoothstep1(1.0, 1.9, ground + 0.25 * co_nz(CO_SEED + 50lu, p, 30.0, gsd));
    let zone = ty.flat * (1.0 - smoothstep1(2.8, 3.6, ground));
    if (zone <= 0.0) {
        return;
    }
    (*s).no_town = true;
    let ripple = co_nz(CO_SEED + 51lu, p, 14.0, gsd);
    let mud = mixc(srgb(122.0, 108.0, 90.0), srgb(86.0, 78.0, 68.0), wet) * (1.0 + 0.07 * ripple);
    var marsh = mixc(srgb(92.0, 104.0, 64.0), srgb(128.0, 96.0, 66.0), smoothstep1(0.1, 0.6, co_nz(CO_SEED + 52lu, p, 70.0, gsd)));
    marsh = marsh * (1.0 + 0.1 * co_nz(CO_SEED + 53lu, p, 7.0, gsd));
    let col = mixc(mud, marsh, high);
    var k = LC_TIDAL_FLAT;
    if (high > 0.5) {
        k = LC_MARSH;
    }
    var ly = layer_paint(zone, col, k);
    ly.clear = 1.0;
    composite(s, ly);
    (*s).m.natural_ok *= 1.0 - zone;
    let win = 1.0 - smoothstep1(1.6, 3.2, ground);
    if (win > 0.0) {
        let c = co_creek(p, gsd, max(fw, 0.3 * gsd)) * zone * win;
        if (c > 0.0) {
            let cc = mixc(srgb(62.0, 58.0, 48.0), srgb(52.0, 56.0, 40.0), high);
            var ly2 = layer_paint(c, cc, LC_TIDAL_FLAT);
            ly2.clear = 1.0;
            composite(s, ly2);
        }
    }
    co_ponds(s, zone);
}

fn co_ponds(s: ptr<function, Stack>, zone: f32) {
    let t = (*s).t;
    let ground = (*s).l.ground;
    if (t.temp < 14.0 || ground < 0.3 || ground > 3.2 || zone < 0.5) {
        return;
    }
    let p = (*s).c.p;
    let gsd = (*s).c.gsd;
    let fw = (*s).l.fw;
    let area = smoothstep1(-0.2, 0.1, perlin3(CO_SEED + 60lu, p * (1.0lf / 2400.0lf)));
    if (area <= 0.0) {
        return;
    }
    var q = (*s).q_loc;
    if ((*s).has_region) {
        q = (*s).q_rot;
    }
    let ca = 150.0;
    let cb = 85.0;
    let ix = floor(q.x / ca);
    let iy = floor(q.y / cb);
    let fx = q.x - (ix + 0.5) * ca;
    let fy = q.y - (iy + 0.5) * cb;
    let h = hash2(CO_SEED + 61lu, i64(ix), i64(iy));
    let salt = t.moist < 0.45;
    var mean_col = srgb(88.0, 130.0, 116.0);
    if (salt) {
        mean_col = srgb(184.0, 150.0, 140.0);
    }
    let e = smoothstep1(1.2 * gsd, 3.0 * gsd, cb);
    var exact = 0.0;
    var col = mean_col;
    if (e > 0.0 && u01k(h, 1lu) < 0.7) {
        let f = max(fw, 0.3 * gsd);
        let hx = 0.5 * ca - 4.0;
        let hy = 0.5 * cb - 4.0;
        exact = min(clamp((hx - abs(fx)) / f + 0.5, 0.0, 1.0), clamp((hy - abs(fy)) / f + 0.5, 0.0, 1.0));
        let pick = u01k(h, 2lu);
        if (salt) {
            if (pick < 0.3) {
                col = srgb(206.0, 120.0, 132.0);
            } else if (pick < 0.5) {
                col = srgb(232.0, 214.0, 208.0);
            } else if (pick < 0.75) {
                col = srgb(78.0, 166.0, 168.0);
            } else {
                col = srgb(96.0, 150.0, 98.0);
            }
        } else if (pick < 0.4) {
            col = srgb(70.0, 140.0, 130.0);
        } else if (pick < 0.75) {
            col = srgb(52.0, 110.0, 100.0);
        } else {
            col = srgb(104.0, 112.0, 84.0);
        }
    }
    let c = area * zone * (e * exact + (1.0 - e) * 0.7 * 0.62);
    if (c > 0.0) {
        var ly = layer_paint(c, mixc(mean_col, col, e), LC_AQUACULTURE);
        ly.clear = 1.0;
        ly.relit = 0.6;
        composite(s, ly);
        (*s).m.natural_ok *= 1.0 - c;
    }
}

/// `coast::agriculture`: the cover of the deltas' plains.
fn coast_slot_agriculture(s: ptr<function, Stack>) {
    let ground = (*s).l.ground;
    if (ground < 0.0 || ground >= 3.0 || (*s).l.water > ground) {
        return;
    }
    let t = (*s).t;
    if (t.temp >= 20.0 && t.moist >= 0.4) {
        return;
    }
    let p = (*s).c.p;
    let gsd = (*s).c.gsd;
    let blk = (*s).l.blk;
    var plain = 0.0;
    var lev = 0.0;
    for (var k = 0u; k < feat_stamp_n(blk); k++) {
        let st = feat_stamp(blk, k);
        if (st.tmpl == 128u) {
            let id = u64(u32(st.v0.z)) | (u64(u32(st.v0.w)) << 24u) | (u64(u32(st.v1.x)) << 48u);
            let sh = coast_delta_shape(st.center.xyz, f64(st.v0.x), st.v0.y, id, st.h, p, gsd);
            plain = max(plain, sh.plain);
            lev = max(lev, sh.lev);
        }
    }
    let ragged = 0.6 * co_nz(CO_SEED + 63lu, p, 170.0, gsd) + 0.4 * co_nz(CO_SEED + 64lu, p, 52.0, gsd);
    let zone = plain * smoothstep1(-0.2, 0.2, plain - 0.5 + 0.5 * ragged) * (1.0 - smoothstep1(2.4, 3.0, ground)) * (1.0 - (*s).m.field_cov);
    if (zone <= 0.0) {
        return;
    }
    let arid = 1.0 - smoothstep1(0.2, 0.34, t.moist);
    let low = 1.0 - smoothstep1(0.6, 1.1, ground + 0.3 * co_nz(CO_SEED + 60lu, p, 90.0, gsd));
    let silt = srgb(144.0, 134.0, 114.0) * (1.0 + 0.1 * co_nz(CO_SEED + 65lu, p, 38.0, gsd));
    let salt = smoothstep1(0.2, 0.45, co_nz(CO_SEED + 66lu, p, 120.0, gsd) + 0.5 * co_nz(CO_SEED + 67lu, p, 33.0, gsd)) * (1.0 - low);
    let dry = mixc(silt, srgb(192.0, 186.0, 170.0), 0.7 * salt);
    let mud = mixc(mixc(srgb(118.0, 106.0, 88.0), srgb(84.0, 76.0, 66.0), low), mixc(dry, srgb(96.0, 90.0, 78.0), low), arid);
    let reed = mixc(srgb(96.0, 106.0, 62.0), srgb(134.0, 112.0, 70.0), smoothstep1(0.0, 0.7, co_nz(CO_SEED + 61lu, p, 260.0, gsd)));
    let m = (1.0 - arid) * (1.0 - low);
    var col = mixc(mud, reed, m) * (1.0 + 0.08 * co_nz(CO_SEED + 62lu, p, 9.0, gsd));
    let scrub = lev * (0.45 + 0.4 * smoothstep1(-0.2, 0.3, co_nz(CO_SEED + 68lu, p, 24.0, gsd)));
    col = mixc(col, mixc(srgb(70.0, 84.0, 48.0), srgb(84.0, 88.0, 58.0), arid), scrub);
    col = mixc(col, (*s).col, 0.3 + 0.45 * (1.0 - plain));
    var k = LC_TIDAL_FLAT;
    if (m > 0.5) {
        k = LC_MARSH;
    }
    var ly = layer_paint(zone, col, k);
    ly.clear = 1.0;
    composite(s, ly);
    (*s).m.natural_ok *= 1.0 - zone;
    let c = co_creek(p, gsd, max((*s).l.fw, 0.3 * gsd)) * zone * (0.3 + 0.7 * low) * (1.0 - scrub);
    if (c > 0.0) {
        var lc = layer_paint(c, srgb(70.0, 66.0, 56.0), LC_TIDAL_FLAT);
        lc.clear = 1.0;
        composite(s, lc);
    }
}

/// `coast::water`.
fn coast_slot_water(s: ptr<function, Stack>) {
    let l = (*s).l;
    if (!(l.water > l.ground) || l.water_kind != W_OCEAN) {
        return;
    }
    let depth = l.water - l.ground;
    let t = (*s).t;
    if (depth > 45.0) {
        return;
    }
    let p = (*s).c.p;
    let gsd = (*s).c.gsd;
    var atoll = 0.0;
    let blk = l.blk;
    let cnt = inst_count(FAM_COAST_ATOLLS, blk);
    for (var i = 0u; i < cnt; i++) {
        let a = co_atoll_at(inst_get(FAM_COAST_ATOLLS, blk, i), p, gsd);
        if (a.ok && a.d < a.wre + 300.0) {
            atoll = 1.0;
        }
    }
    let coastal = t.coast_km <= 40.0 && t.coast_km >= -120.0;
    if (!coastal && atoll == 0.0) {
        return;
    }
    var ty: CoTypes;
    if (coastal) {
        ty = co_types(p, t.uplift, t.mountain, t.glaciation, t.temp);
    }
    if (ty.reef > 0.0 || atoll > 0.0) {
        let a = max(ty.reef, atoll) * (1.0 - smoothstep1(18.0, 40.0, depth));
        if (a > 0.0) {
            let sand = mixc(srgb(150.0, 214.0, 200.0), srgb(36.0, 168.0, 180.0), smoothstep1(1.5, 14.0, depth));
            let n = 0.6 * co_nz(CO_SEED + 70lu, p, 26.0, gsd) + 0.4 * co_nz(CO_SEED + 71lu, p, 7.0, gsd);
            let wd = 0.06 + 0.3 * (1.0 - band(26.0, gsd));
            let coral = smoothstep1(0.12 - wd, 0.12 + wd, n) * smoothstep1(0.6, 1.8, depth) * (1.0 - smoothstep1(10.0, 20.0, depth));
            let cc = mixc(srgb(88.0, 98.0, 62.0), srgb(122.0, 84.0, 60.0), 0.5 + 0.5 * co_nz(CO_SEED + 72lu, p, 12.0, gsd));
            let col = mixc(sand, cc, 0.7 * coral);
            var k = LC_OCEAN;
            if (depth < 10.0) {
                k = LC_CORAL_REEF;
            }
            var ly = layer_paint(a, col, k);
            ly.relit = 1.0;
            composite(s, ly);
        }
    }
    if (ty.barrier > 0.0 && depth < 7.0) {
        let a = ty.barrier * (1.0 - smoothstep1(5.0, 7.0, depth));
        let col = mixc(srgb(150.0, 186.0, 160.0), srgb(70.0, 130.0, 128.0), smoothstep1(0.5, 6.0, depth));
        var ly = layer_paint(0.85 * a, col, LC_OCEAN);
        ly.relit = 1.0;
        composite(s, ly);
    }
    if (ty.flat > 0.0 && depth < 3.0) {
        let a = ty.flat * (1.0 - smoothstep1(1.2, 3.0, depth));
        let col = mixc(srgb(122.0, 118.0, 92.0), srgb(86.0, 98.0, 84.0), smoothstep1(0.2, 2.5, depth)) * (1.0 + 0.06 * co_nz(CO_SEED + 73lu, p, 40.0, gsd));
        var ly = layer_paint(0.9 * a, col, LC_OCEAN);
        ly.relit = 1.0;
        composite(s, ly);
    }
    if (ty.skerry > 0.0 && depth > 0.8 && depth < 16.0) {
        let n = 0.6 * co_nz(CO_SEED + 74lu, p, 45.0, gsd) + 0.4 * co_nz(CO_SEED + 75lu, p, 12.0, gsd);
        let wd = 0.06 + 0.3 * (1.0 - band(45.0, gsd));
        let k = ty.skerry * smoothstep1(0.05 - wd, 0.05 + wd, n) * smoothstep1(0.8, 2.0, depth) * (1.0 - smoothstep1(10.0, 16.0, depth));
        if (k > 0.0) {
            var ly = layer_paint(0.75 * k, srgb(22.0, 52.0, 42.0), LC_OCEAN);
            ly.relit = 1.0;
            composite(s, ly);
        }
    }
    if (ty.cliff > 0.0 && depth < 2.6) {
        let a = ty.cliff * (1.0 - smoothstep1(0.8, 2.6, depth));
        var ly = layer_paint(0.6 * a, srgb(112.0, 150.0, 140.0), LC_OCEAN);
        ly.relit = 1.0;
        composite(s, ly);
    }
}
