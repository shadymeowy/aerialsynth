# Stars

Night skies show the real stars, the planets and the Moon at their apparent positions for the scenario's date, time and
the camera's position and attitude. The accuracy is star-tracker grade, the brightness is
radiometric, and every frame can carry star ground truth.

```yaml
render:
  lighting: { mode: clock, date: "2026-01-20", time_utc: "19:30:00", stars: true }
  stars:                     # all optional
    catalog: null            # default: built-in Hipparcos + Tycho-2, V ≤ 9 (130k stars)
    mag_limit: null          # faintest V rendered (null: the whole catalogue)
    psf_sigma_px: 0.5        # star image: Gaussian PSF integrated over the pixels
    brightness: 1.0          # 1 = physical
    refraction: true
    extinction: true
    aberration: true         # also enables light deflection by the Sun
    planets: true            # Mercury … Neptune (DE440, 1990–2060); Moon in the ground truth
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

The catalogue contains stars only; the planets come from the ephemeris (below). It has no
unresolved Milky Way glow.

## Planets and the Moon (`stars/ephem.rs`, `stars/planets.rs`)

**Ephemeris.** JPL DE440, as its Chebyshev segments, is built into the binary for 1990–2060
(`crates/render/data/planets.bin`, 2.8 MB, from `scripts/build_planets.py`). It covers the Sun,
the planet system barycentres, the Earth–Moon barycentre and the Moon; the Earth is −Moon / EMRAT.
Constant terms are stored as f64 and the others as f32, so rounding is ≤ 2 km for the planets
(≤ 0.008″ even at Venus's closest) and 8 m for the Moon. The same ephemeris gives the Earth's
barycentric position and velocity for the star aberration and parallax. Outside 1990–2060 the
stars fall back to Keplerian elements (≈ 0.01″) and there are no planets.

**Apparent places.** Positions are topocentric, with light time, light deflection by the Sun,
and aberration from the observer's orbital and diurnal velocity, followed by the same Earth
rotation, refraction and camera chain as the stars. They are the positions of the planet
system barycentres: Jupiter's and Saturn's centres lie within ~200–300 km of theirs, ≤ 0.07″.

**Brightness and size.**
* V magnitudes follow Mallama & Hilton 2018, the formulas used by the Astronomical Almanac and
  Skyfield. They depend on phase angle, Saturn's ring tilt and Uranus's sub-observer and
  sub-solar latitudes.
* Colour comes from each planet's mean B−V.
* Planets larger than 0.3 px are drawn as their globe of apparent equatorial radius, Lambert
  shaded by the Sun. That gives the real phases: crescent and half Venus, gibbous Mars and
  Mercury. Jupiter at opposition is 46.6″.
* Saturn has its rings: the C ring, B ring, Cassini division and A ring, each with its own
  brightness and opacity, in the ring plane (IAU pole).
  * The globe hides the rings behind it, and the rings in front veil the globe.
  * The unlit face of the rings is dim.
  * The flux splits between globe and rings by the ring term of the magnitude formula, so the
    rings close up correctly at the ring-plane crossings (2025, 2038–39).
* Shapes are sampled at ≤ 0.25 px and smeared along the exposure track like the stars.
* The ground-truth position is the planet's centre, not the light centroid of a phase.
* Not drawn: the Galilean moons, Saturn's ring shadows, the planets' flattening.

**The Moon.** The Moon is drawn by the sky, as before. Its position and phase, which also drive
moonlight, now come from DE440 (topocentric, with sea-level refraction) instead of mean elements
(~0.5°). It appears in the star ground truth at exactly the drawn position: observer on the
ellipsoid, UT1 = UTC, within ~2″ of the camera's true view.

**Ground-truth ids.** Planets and the Moon have id `1<<30 | NAIF id`: 199 Mercury, 299 Venus,
499 Mars, 599 Jupiter, 699 Saturn, 799 Uranus, 899 Neptune and 301 the Moon.

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

From altitude, stars below the horizontal but above the Earth's limb are visible, through a
ray that dips to its tangent height and back up. Those get twice the horizontal refraction at
the tangent height, less the part above the observer: about 1° at the limb seen from 10 km.
Below the limb (hidden) the refraction fades out.

**Validation** (`cargo test -p render stars`, plus `configs/examples/star_tracker.yaml`):
* Star positions were compared against Skyfield 1.53 (JPL DE421, IAU 2000A) for the same
  catalogue records: 1,600 star/place/time cases, sea level to 10 km, years 2026–2031. The
  mean separation is 0.0002–0.0005″ and the worst 0.0009″.
* Planets and the Moon were compared against Skyfield with DE440s over five dates from 1995 to
  2045. The worst planet separation is 0.007″, the Moon's 0.002″, and magnitudes agree with
  `skyfield.magnitudelib` to within 0.03.
* Jupiter's rendered disc centroids to 0.012 px of its ground truth.
* Refraction matches pyERFA's `refco` to within 0.02″.
* Centroids measured in the rendered images (`configs/examples/star_tracker.yaml`: 25° field of view,
  100 ms, 8-bit output) against the ground truth:
  * without motion: 0.02 px for bright stars and 0.06 px at V 4–5.5, limited by noise at
    fainter magnitudes;
  * with 2–5 px vibration trails: trail centroids match xm, ym to a median of 0.06–0.13 px,
    limited by 8-bit quantisation of the dim trail pixels.

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
* **Trails:** frame cameras draw each star along its image track over the open shutter (17
  camera sub-poses, then steps of at most 0.25 px, equal energy per unit time). Camera rotation
  (manoeuvres, vibration) smears stars exactly. This happens after the image-space motion blur
  of the terrain. Event cameras see the stars at each instant.
* The camera's exposure, noise, optics (defocus, bloom), motion blur and the event sensor all
  apply as to the rest of the scene.

## Ground truth (`<camera>/stars/`)

Per frame (the render at mid-exposure), all catalogue stars inside the image down to the
modality's `mag_limit`. Each star records:
* the catalogue id: the HIP number, or for Tycho-2 `1<<31 | TYC1<<17 | TYC2<<3 | TYC3`;
* the sub-pixel position x, y at the frame time (mid-exposure), with pixel centres at integer
  coordinates;
* the position averaged over the exposure, xm, ym. This is the centroid of the star's trail,
  which is what a centroiding star tracker measures under motion;
* V;
* the irradiance at the camera after extinction;
* whether the pixel shows sky.

`index[k]..index[k+1]` selects frame k. The camera pose at the frame time is in `pose/`.

## Sky-pointing cameras on a fresh world

The renderer must know that no terrain can rise into the view. While tile elevations are
unknown it assumes up to 6 km, so the pre-render dry run learns the real heights one zoom level
per pass instead of refining that conservative volume to the finest zoom. Tiles lying entirely
below the view cone are also culled. The example's first run generates about 140 tiles instead
of over 5,000.

## Limitations (known, not modelled)

**Catalogue**
* Built in to V = 9; a Tycho-2 file goes deeper. There is no unresolved background (Milky Way
  glow, zodiacal light, airglow structure) beyond the renderer's uniform night-sky floor.
* Stars are points at mean catalogue values: no variability, no orbital motion of binaries, no
  radial velocity (perspective acceleration, sub-mas for decades).
* Tycho-2 stars without a mean position (flag X, a few thousand) carry no proper motion.
* V and B−V of Tycho-only stars come from the BT/VT conversion (≈ 0.05 mag).

**Astrometry**
* UT1 − UTC (`dut1_s`) and polar motion are constants given per scenario (default 0: up to 13″
  and 0.5″), not interpolated from IERS tables.
* Aberration uses the Earth's orbital velocity and the observer's diurnal velocity, but not the
  vehicle's: ≈ 0.17″ at 250 m/s, 5″ at orbital speed.
* Light deflection is by the Sun only. Jupiter's limb would add up to 16 mas.
* Refraction:
  * It comes from the standard atmosphere at the camera's altitude at one wavelength
    (0.574 µm): no weather, humidity or chromatic refraction (stars are not dispersed into
    spectra near the horizon).
  * Below the horizontal, seen from altitude, it is a simplified tangent-height model.
* No scintillation (twinkling) and no seeing: the PSF is a fixed Gaussian, the same across the
  field.

**Ephemeris and planets**
* JPL DE440 covers 1990–2060 only. Outside that range there are no planets, the Earth's motion
  comes from Keplerian elements (≈ 0.01″ for stars) and the Moon from mean elements (≈ 0.5°).
* Positions are planet system barycentres: Jupiter's and Saturn's centres are within
  ≈ 0.07″. TDB is taken equal to TT (≤ 1.7 ms).
* Globes are Lambert-shaded spheres: no limb darkening beyond Lambert, no albedo features
  (Jupiter's bands, Mars's markings), no flattening.
* Saturn's rings are 4 uniform zones: no ring shadow on the globe and no globe shadow on the
  rings.
* No moons other than ours: the Galilean moons (V 4.6–5.7) are visible to star trackers and are
  missing.
* Magnitudes follow Mallama & Hilton 2018 within their phase-angle ranges. Colours use mean B−V.
* The ground-truth position of a planet is its centre, not the photocentre of a phase.

**The Moon**
* The sky draws it as a uniform disc scaled by the illuminated fraction, not as a shaded
  crescent, and without earthshine.
* Its ground truth matches the drawn disc: observer on the ellipsoid, UT1 = UTC, refraction at
  sea level. That is within ~2″, plus the refraction difference for high cameras near the
  horizon.

**Photometry and rendering**
* The V band stands in for luminance, and colours are blackbodies from B−V; neither is a
  spectral integration over the sensor's channels.
* Extinction uses the renderer's two-layer atmosphere with the Kasten–Young air mass, which is
  approximate for elevated observers and near or below the horizon.
* Stars are added only to pixels that show sky: terrain occludes at pixel granularity, with no
  partial occlusion at silhouettes.
* Trails sample the exposure with 17 camera poses (linear in between). Event cameras see the
  stars at the render instants, interpolated between keyframes like the rest of the image.
* The sensor writes 8-bit images, which limits faint-star and trail centroiding (≈ 0.1 px).
* The leap-second table ends at 2017-01-01 (none announced since); before 1972 TAI − UTC is
  approximated.

## Data credits

* Hipparcos and Tycho-2: ESA, via CDS / VizieR (catalogues I/239, I/311, I/259).
* ERFA: BSD-3 licence, derived from IAU SOFA. The nutation table is taken from ERFA `nut00b.c`.
* JPL DE440 (Park et al. 2021): public domain.
