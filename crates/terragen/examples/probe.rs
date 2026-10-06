fn main() {
    let g = terragen::Generator::new(terragen::Config::default());
    let args: Vec<f64> = std::env::args().skip(1).map(|s| s.parse().unwrap()).collect();
    let (lat, lon) = (args[0].to_radians(), args[1].to_radians());
    for gsd in [600.0, 150.0, 38.0, 9.5, 2.4, 0.6] {
        let (t, h, c) = g.probe(lat, lon, gsd);
        println!("gsd {gsd:6.1}: class {c:2} dsm {h:7.1} ground {:7.1} temp {:5.1} moist {:.2} agri {:.2} habit {:.2} mtn {:.2} sand {:.2} rock {:.2}",
            t.ground, t.temp, t.moist, t.agri, t.habit, t.mountain, t.sand, t.rock_expect);
    }
}
