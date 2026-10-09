// The test-only example kit's relief (every module with pass A).

fn example_cones_exists(s: InstSite) -> InstOut {
    var o: InstOut;
    o.ok = false;
    if (u01k(s.id, 1lu) >= 0.3) {
        return o;
    }
    if (continent(s.center, 50000.0) < 0.05) {
        return o;
    }
    o.ok = true;
    o.v = array<f32, 8>(2000.0 + 3000.0 * u01k(s.id, 2lu), 300.0 + 600.0 * u01k(s.id, 3lu), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    return o;
}

fn example_relief(c: Ctx, m: Macro, r: ReliefIn, h: ptr<function, f32>) {
    let l = inst_list(FAM_EXAMPLE_CONES, r.blk, c.p);
    var z = 0.0;
    for (var k = 0u; k < l.n; k++) {
        let rr = l.items[k].v[0];
        let hh = l.items[k].v[1];
        let d = dist64(c.p, l.items[k].center);
        if (d < rr) {
            z = max(z, hh * pow(1.0 - d / rr, 1.5) * band(rr, c.gsd));
        }
    }
    *h += z;
}
