// The cold kit's relief (`kits/cold.rs`; every module with pass A): drumlin fields with eskers
// in glaciated lowlands.

const COLD_DRUMLIN_MEAN: f32 = 0.109;

fn cold_drumlins_exists(s: InstSite) -> InstOut {
    var o: InstOut;
    o.ok = false;
    if (u01k(s.id, 1lu) >= 0.7) {
        return o;
    }
    let a = atlas_sample(vec3<f32>(normalize(s.center)));
    if (a.glaciation < 0.5 || a.coast_km < 5.0 || a.elevation > 700.0) {
        return o;
    }
    o.ok = true;
    o.v = array<f32, 8>(9000.0 + 5000.0 * u01k(s.id, 2lu), TAU * u01k(s.id, 3lu), 8.0 + 14.0 * u01k(s.id, 4lu), 0.7 + 0.6 * u01k(s.id, 5lu), u01k(s.id, 6lu), 0.0, 0.0, 0.0);
    return o;
}

/// Height (m), drumlin crest (0..1) and esker (0..1) of the drumlin fields at `p`
/// (`drumlin_field`).
fn cold_drumlin_field(l: InstList, p: vec3<f64>, gsd: f32) -> vec3<f32> {
    var z = 0.0;
    var crest = 0.0;
    var esk = 0.0;
    for (var k = 0u; k < l.n; k++) {
        let it = l.items[k];
        let r = it.v[0];
        let rel = vec3<f32>(p - it.center);
        let d2 = dot(rel, rel);
        if (d2 >= r * r) {
            continue;
        }
        let up = normalize(vec3<f32>(it.center));
        var e = vec3<f32>(1.0, 0.0, 0.0);
        if (abs(up.z) <= 0.999) {
            e = normalize(cross(vec3<f32>(0.0, 0.0, 1.0), up));
        }
        let n = cross(up, e);
        let sn = sin(it.v[1]);
        let cs = cos(it.v[1]);
        let u = dot(rel, e * cs + n * sn);
        let wq = dot(rel, n * cs - e * sn);
        let fade = smoothstep1(r, 0.6 * r, sqrt(d2));
        let sc = it.v[3];
        let cu = 900.0 * sc;
        let cw = 320.0 * sc;
        let bnd = band(0.35 * cw, gsd);
        var zd = 0.0;
        if (bnd > 0.0) {
            let iu = i64(floor(u / cu));
            let iw = i64(floor(wq / cw));
            for (var du = -1; du <= 1; du++) {
                for (var dw = -1; dw <= 1; dw++) {
                    let gu = iu + i64(du);
                    let gw = iw + i64(dw);
                    let h = hash2(it.id, gu, gw);
                    if (u01k(h, 1lu) >= 0.85) {
                        continue;
                    }
                    let c0 = (f32(gu) + 0.5 + 0.6 * (u01k(h, 2lu) - 0.5)) * cu;
                    let c1 = (f32(gw) + 0.5 + 0.6 * (u01k(h, 3lu) - 0.5)) * cw;
                    let a = cu * (0.33 + 0.12 * u01k(h, 4lu));
                    let b = cw * (0.3 + 0.1 * u01k(h, 5lu));
                    var x = (u - c0) / a;
                    if (x < 0.0) {
                        x *= 1.25;
                    }
                    let y = (wq - c1) / b;
                    let r2 = x * x + y * y;
                    if (r2 < 1.0) {
                        zd = max(zd, (1.0 - r2) * (1.0 - r2) * (0.7 + 0.6 * u01k(h, 6lu)));
                    }
                }
            }
        }
        z += it.v[2] * (zd * bnd + COLD_DRUMLIN_MEAN * (1.0 - bnd)) * fade;
        crest = max(crest, zd * bnd * fade);
        let we = 40.0;
        let be = band(2.0 * we, gsd);
        if (be > 0.0) {
            let ph = it.v[4];
            let cl = 0.5 * r * (ph - 0.5) + 600.0 * sin(u / 2300.0 + ph * 6.3) + 250.0 * sin(u / 700.0 + ph * 17.0);
            let dd = abs(wq - cl);
            if (dd < we) {
                let on = smoothstep1(-0.2, 0.0, sin(u / 3100.0 + ph * 11.0));
                let x = dd / we;
                let ze = (1.0 - x * x) * on * be * fade;
                z += 12.0 * ze;
                esk = max(esk, ze);
            }
        }
    }
    return vec3<f32>(z, crest, esk);
}

fn cold_relief(c: Ctx, m: Macro, r: ReliefIn, h: ptr<function, f32>) {
    let l = inst_list(FAM_COLD_DRUMLINS, r.blk, c.p);
    if (l.n == 0u) {
        return;
    }
    *h += cold_drumlin_field(l, c.p, c.gsd).x;
}
