//! Close-up of a planet with an ideal pinhole camera pointed at it (star field only, no
//! terrain): planet_frame UNIX LAT_DEG LON_DEG NAIF FOCAL_PX GAIN OUT.png
use geodesy::frames::{geodetic2ecef, Geodetic};
use geodesy::Ellipsoid;
use glam::{DMat3, DVec3};
use render::camera::CameraConfig;
use render::stars::{StarField, StarsConfig};
use render::trajectory::CamPose;

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let (unix, lat, lon): (f64, f64, f64) = (a[0].parse()?, a[1].parse::<f64>()?.to_radians(), a[2].parse::<f64>()?.to_radians());
    let (naif, f, gain): (u32, f64, f64) = (a[3].parse()?, a[4].parse()?, a[5].parse()?);
    let (w, h) = (320usize, 180usize);
    let ell = Ellipsoid::WGS84;
    let pos = geodetic2ecef(Geodetic::new(lat, lon, 1500.0), &ell);
    let field = StarField::new(&StarsConfig::default())?;
    let body = field.bodies(unix, pos, &ell).into_iter().find(|b| b.id == (1 << 30) | naif).expect("body");
    // camera: z at the body, image up towards the zenith
    let up = DVec3::new(lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin());
    let z = body.dir;
    let y = -(up - z * up.dot(z)).normalize();
    let x = y.cross(z);
    let cam = CamPose { t: 0.0, pos, r_ecef_cam: DMat3::from_cols(x, y, z) };
    let model = CameraConfig::pinhole_hfov(w as u32, h as u32, 2.0 * (0.5 * w as f64 / f).atan().to_degrees()).build()?;
    let mut rad = vec![0f32; w * h * 3];
    let points = vec![None; w * h];
    let obs = field.render_track(&mut rad, &points, (w, h), model.as_ref(), &[cam], &cam, unix, &ell, &Default::default());
    let o = obs.iter().find(|o| o.id == (1 << 30) | naif).unwrap();
    eprintln!("V {:.2}, radius {:.2}\", centre GT ({:.2}, {:.2})", body.v, body.radius.to_degrees() * 3600.0, o.x, o.y);
    let px: Vec<u8> = rad.iter().map(|v| ((v * gain as f32).clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0) as u8).collect();
    image::save_buffer(&a[6], &px, w as u32, h as u32, image::ExtendedColorType::Rgb8)?;
    Ok(())
}
