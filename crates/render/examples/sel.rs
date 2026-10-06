use render::lod::*;
fn main() {
    let s = render::scenario::Scenario::load(std::path::Path::new("configs/quick.yaml")).unwrap();
    let ell = geodesy::Ellipsoid::WGS84;
    let poses = render::trajectory::load(&s.trajectory.file, &ell).unwrap();
    let gen = terragen::Generator::new(s.world.clone());
    let est = render::pipeline::generator_range_estimator(&gen);
    let oracle = PlanOracle { estimate: Some(&est), fixed: (0.0, 0.0) };
    let model = s.camera.build().unwrap();
    let cam = poses[0].camera(&s.extrinsics, &ell);
    let params = LodParams { min_zoom: 2, max_zoom: 17, texel_px: 0.8, cone_margin: 0.08, ..Default::default() };
    let sel = Selector::new(&cam, model.as_ref(), ell, &params, &oracle);
    let mut v = sel.select();
    v.sort_by_key(|u| u.id.z);
    for u in &v { let r = est(u.id); let (lat, lon) = u.id.center(); println!("{:?} range {:?} center {:.4},{:.4}", u.id, r, lat.to_degrees(), lon.to_degrees()); }
    println!("cam lla {:?}", geodesy::ecef2geodetic(cam.pos, &ell));
    let id = geodesy::tiles::TileId::new(7, 71, 48);
    let (c, r) = tile_sphere(id, (-40.0, 4863.0), &ell);
    let d = c - cam.pos;
    let axis = cam.r_ecef_cam * glam::DVec3::Z;
    println!("dist {} r {} angle {} half {} vis {:?}", d.length(), r, (d.dot(axis)/d.length()).acos().to_degrees(), model.max_half_angle().to_degrees(), sel.visible(id, (-40.0, 4863.0)));
    println!("axis {:?} up {:?}", axis, cam.pos.normalize());
}
#[allow(dead_code)]
fn _x() {}
