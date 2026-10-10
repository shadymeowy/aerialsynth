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

fn example_slot_linear(s: ptr<function, Stack>) {
    let p = (*s).c.p;
    let blk = (*s).l.blk;
    let fw = max((*s).l.fw, 0.5 * (*s).c.gsd);
    var cov = 0.0;
    for (var k = 0u; k < feat_seg_n(blk); k++) {
        let seg = feat_seg(blk, k);
        if (seg.kind == 240u) {
            cov = max(cov, kband_cov(seg_frame(seg, p).x, seg.hw, fw));
        }
    }
    for (var k = 0u; k < feat_stamp_n(blk); k++) {
        let st = feat_stamp(blk, k);
        if (st.tmpl == 240u) {
            let uv = stamp_uv(st, p);
            cov = max(cov, clamp(min(st.half.x - abs(uv.x), st.half.y - abs(uv.y)) / fw + 0.5, 0.0, 1.0));
        }
    }
    if (cov > 0.0) {
        var ly = layer_paint(cov, srgb(162.0, 146.0, 120.0), LC_TRACK);
        ly.hmode = HM_BLEND;
        ly.relit = 0.5;
        composite(s, ly);
    }
}
