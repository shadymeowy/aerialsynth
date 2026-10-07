# Stars

Night skies show the real stars at their apparent positions for the scenario's date, time and
the camera's position and attitude. The accuracy is star-tracker grade, the brightness is
radiometric, and every frame can carry star ground truth.

```yaml
render:
  lighting: { mode: clock, date: "2026-01-20", time_utc: "19:30:00", stars: true }
  stars:                     # all optional
    catalog: null            # default: built-in Hipparcos + Tycho-2, V ≤ 9 (130k stars)
    mag_limit: 99            # faintest V rendered
    psf_sigma_px: 0.5        # star image: Gaussian PSF integrated over the pixels
    brightness: 1.0          # 1 = physical
    refraction: true
    extinction: true
    aberration: true         # also enables light deflection by the Sun
    dut1_s: 0.0              # UT1 − UTC (IERS Bulletin A)
    polar_motion_arcsec: [0, 0]
cameras:
  - path: /cam0
    rgb: {}
    stars: { mag_limit: 9 }  # ground truth: catalogue stars in each frame
```

## Catalogue

The built-in catalogue, `crates/render/data/stars_v9.bin`, holds 130,442 stars and takes
3.7 MB. It is complete to V = 9, about 1.5–2 magnitudes fainter than typical star trackers
detect (V ≈ 6.5–7.5).

* **Hipparcos** (ESA 1997; new reduction, van Leeuwen 2007): 83,337 stars. Positions come
  from the new reduction; V and B−V from the 1997 catalogue.
* **Tycho-2** (Høg et al. 2000): 47,105 stars with no Hipparcos entry, where Hipparcos is
  incomplete (fainter than V ≈ 7.3–8). V and B−V are converted from the Tycho BT/VT
  magnitudes.

Positions are ICRS at epoch J2000.0, with proper motion and parallax, sorted by V. Each
record is 28 bytes. All stars are real, so the ground truth covers every star in the image.

| V ≤ | stars |
|----:|------:|
| 6 | 5,049 |
| 7 | 15,641 |
| 8 | 45,925 |
| 9 | 130,442 |
| 10 | 355,045 (Tycho-2 file) |
| 12 | 2,078,257 (Tycho-2 file) |

A deeper catalogue (all of Tycho-2, about 58 MB) can be built from the CDS files and set as
`render.stars.catalog`:

```sh
python scripts/build_stars.py CATDIR --vmax 12.5 -o tycho2.stars   # see the script for the download
```

What the catalogue does not contain: planets, and the unresolved Milky Way glow.

## Astrometry (`stars/astro.rs`)

Each render takes the catalogue through the following chain:

1. Catalogue (ICRS, J2000).
2. Proper motion.
3. Annual parallax.
4. Light deflection by the Sun.
5. Aberration, from the Earth's orbital velocity plus the observer's diurnal velocity.
6. GCRS → ITRS, using:
   * IAU 2006 precession and IAU 2000B nutation;
   * Greenwich apparent sidereal time from UT1 = UTC + `dut1_s`;
   * TT = UTC + leap seconds + 32.184 s;
   * polar motion.
7. Refraction at the observer.
8. Camera frame.
9. Camera model (any model, including fisheye).

The algorithms follow ERFA / IAU SOFA. The Earth's barycentric velocity comes from Keplerian
elements with the Sun's reflex motion (Jupiter, Saturn), which is good to about 0.01″ in
aberration.

Refraction uses the ERFA `refco` model (Green's A tan z + B tan³ z: dry air, 0.574 µm) above
20° elevation, and Sæmundsson's formula below 10°, blended in between. Pressure and temperature
come from the US standard atmosphere at the observer's altitude, and refraction vanishes above
80 km.

**Validation** (`cargo test -p render stars`):
* Positions were compared against Skyfield 1.53 (JPL DE421, IAU 2000A) for the same catalogue
  records: 1,600 star/place/time cases, sea level to 10 km, years 2026–2031. The mean
  separation is 0.006–0.008″ and the worst 0.011″.
* Refraction matches pyERFA's `refco` to within 0.02″.

**Error budget of the inputs:**
* `dut1_s`: leaving it at 0 rotates the sky by up to 15″/s × |UT1 − UTC| (≤ 13″). Set it from
  IERS Bulletin A for arcsecond work.
* Polar motion: ≤ 0.5″ if left at zero.
* Vehicle velocity is not included in the aberration: about 0.17″ at 250 m/s, 5″ at orbital
  speed.

## Brightness

* A star's irradiance relative to the Sun's is 10^(−0.4 (V + 26.74)).
* This uses the renderer's units: radiance 1 is a white Lambertian surface under the direct
  Sun at one air mass. A star is a point source, so its pixel radiance is its irradiance
  divided by the pixel's solid angle, taken from the camera model.
* The star is dimmed by the renderer's atmosphere (Rayleigh and haze layers) along its air
  mass from the camera's altitude.
* Its colour is a blackbody at the temperature implied by B−V (Ballesteros 2012), relative to
  the Sun's white.
* The image is a Gaussian PSF (`psf_sigma_px`) integrated over the pixels, added to the pixels
  that show sky. Terrain occludes at pixel granularity.
* The camera's exposure, noise, optics (defocus, bloom), motion blur and the event sensor all
  apply as to the rest of the scene.

## Ground truth (`<camera>/stars/`)

Per frame (the render at mid-exposure), all catalogue stars inside the image down to the
modality's `mag_limit`. Each star records:
* the catalogue id: the HIP number, or for Tycho-2 `1<<31 | TYC1<<17 | TYC2<<3 | TYC3`;
* the sub-pixel position x, y, with pixel centres at integer coordinates;
* V;
* the irradiance at the camera after extinction;
* whether the pixel shows sky.

`index[k]..index[k+1]` selects frame k. The camera pose at the frame time is in `pose/`.

## Data credits

* Hipparcos and Tycho-2: ESA, via CDS / VizieR (catalogues I/239, I/311, I/259).
* ERFA: BSD-3 licence, derived from IAU SOFA. The nutation table is taken from ERFA `nut00b.c`.
