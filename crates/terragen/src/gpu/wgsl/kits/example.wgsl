// The test-only example kit (`kits/example.rs`).

fn example_slot_azonal(s: ptr<function, Stack>) {
    let l = inst_list(FAM_EXAMPLE_CONES, (*s).l.blk, (*s).c.p);
    var cov = 0.0;
    for (var k = 0u; k < l.n; k++) {
        let d = dist64((*s).c.p, l.items[k].center);
        cov = max(cov, kband_cov(d, 0.6 * l.items[k].v[0], max((*s).l.fw, 0.5 * (*s).c.gsd)));
    }
    if (cov > 0.0) {
        var ly = layer_paint(cov, srgb(70.0, 64.0, 62.0), LC_VOLCANIC_ASH);
        ly.clear = 1.0;
        composite(s, ly);
    }
}

fn example_rings(li: u32, i: KIn) -> KOut {
    let k = kp_load(li);
    let r = k.v[0];
    let c = floor(i.q / (4.0 * r));
    let rel = i.q - (c + 0.5) * 4.0 * r;
    var o = kout_none();
    o.cov = kband_cov(length(rel) - r, 0.15 * r, i.fw) * min(i.amount, 1.0);
    o.albedo = k.col[0];
    return o;
}
