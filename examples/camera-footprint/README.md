# Camera footprint: camera ↔ geodesy ↔ XYZ tiles

A teaching example in one file ([`src/main.rs`](src/main.rs)), read top to bottom. It takes one
camera position and

1. renders an **oblique** pinhole image (40° below the horizon, turned and slightly banked),
2. renders a **nadir** image from the same position (straight down, north up, east right),
3. casts the rays of the oblique image's 4 corners onto the terrain,
4. converts the 4 ground points to fractional Web-Mercator **XYZ tile** coordinates,
5. stitches the tiles under them (albedo and elevation) and draws the camera's **footprint**,
6. solves the **homography** between the 4 map pixels and the 4 image corners and warps the map
   into the oblique camera's view.

Every step prints what it computes.

```sh
cargo run --release -p camera-footprint            # images in out/camera-footprint/
cargo run --release -p camera-footprint -- DIR     # ... or in DIR
cargo test --release -p camera-footprint           # conversions and homography
```

The tiles are generated on first use into `out/camera-footprint/world.h5` (the default world,
seed 1), on the GPU when there is one. The scene is set by the constants at the top of
`main.rs`: `TIME`, `LAT`, `LON`, `ALT`, `YAW`, `PITCH`, `ROLL`, the image size and focal length,
the map zoom `LOD`, the zoom of the elevation used for the corner rays, and the seed.

## Frames

```text
geodetic (lat, lon, h) ──▶ ECEF (x, y, z) ──▶ ENU (e, n, u) at the camera ──▶ camera (x, y, z) ──K──▶ pixel (u, v)
(lat, lon) ──Web Mercator──▶ fractional tile (x, y) at zoom LOD ──▶ stitched-map pixel
```

| frame | convention | in the code |
|---|---|---|
| geodetic | WGS84 latitude, longitude, height `h` above the ellipsoid (not sea level) | `geodesy::Geodetic` |
| ECEF | Earth-centred, Earth-fixed metres; x to (0°, 0°), z to the north pole | `geodetic2ecef`, `ecef2geodetic` |
| ENU | east, north, up at the camera; up = the ellipsoid normal | `CameraPose::enu` (`geodesy::LocalFrame`) |
| camera | OpenCV: x right, y down, z = optical axis | `CameraPose::r_enu_cam`, `r_ecef_cam` |
| image pixel | (0, 0) = centre of the top-left pixel; the image spans [-0.5, w-0.5] × [-0.5, h-0.5] | `Intrinsics` (`K`) |
| fractional tile | `(x, y) = (u, v)·2^z`, `u = (lon+π)/2π` east, `v = ½ − asinh(tan lat)/2π` south; integer part = the XYZ tile `z/x/y` | `TileXY` |
| stitched-map pixel | `((x − x0)·256, (y − y0)·256)` for the block of tiles starting at `(x0, y0)`; pixel (i, j) covers [i, i+1) × [j, j+1) | `MapGrid` |

**Orientation.** `(YAW, PITCH, ROLL) = (0, 0, 0)` is the nadir camera: optical axis down, image
top north, image right east. Other angles rotate away from it like an aircraft carrying the
camera in its belly with the image top towards the nose: aerospace Z-Y-X angles of the body in
north-east-down, so YAW is the heading of the image top (clockwise from north), PITCH (nose up)
tilts the view from nadir towards the image top (50° = 40° below the horizon), ROLL (right wing
down) tilts it to the left:

```text
R_enu_cam = R_enu_ned · Rz(yaw) · Ry(pitch) · Rx(roll) · R_body_cam
```

Both cameras share the position: `CameraPose::new(position, r_enu_cam(YAW, PITCH, ROLL))` and
`CameraPose::new(position, r_enu_cam(0, 0, 0))`. They are rendered with the bindings' camera
(`aerialsynth_core::Camera`, the renderer of `terrain run`) with the `nadir` mount, whose pose
angles mean the same; the example checks that the renderer's camera pose equals the one it
computed.

## Steps

**Corner rays ∩ terrain.** A corner pixel's ray is `R_ecef_cam · normalize(K⁻¹ (u, v, 1))` from
the camera centre. A fixed number of iterations finds where it meets the ground: start at the
ground height below the camera; then *point on the ray at height h → elevation of the terrain
below that point (elevation tiles at a coarser zoom, bilinear) → new h*. On fairly level ground
the error shrinks by about slope / tan(ray depression) per step; the printed heights show it
settle.

**Tiles.** The ground points become fractional tile coordinates at `LOD`; the integer parts give
the block of whole tiles to fetch (`World::tiles_into`, generated if missing), stitched with tile
`(x0, y0)` at the top left. Albedo is the unlit surface colour; elevation (a DSM: ground, trees,
buildings, metres above the ellipsoid) is drawn as grey from its minimum (black) to its maximum
(white), printed. The footprint is drawn through the 4 corners' map pixels.

**Homography.** `H` maps the 4 map pixels onto the 4 image corners: the direct linear transform
with `h33 = 1` (two equations per point pair, an 8 × 8 linear system, Hartley-normalised points).
The warped map takes, for each image pixel `p`, the map at `H⁻¹ p`. A homography maps a plane
exactly; the ground is only nearly one, so the example measures how far: the rendered depth
gives every pixel's true ground point, which goes through tile → map pixel → `H` back into the
image; the printed offsets come from relief, trees and buildings standing off the plane through
the 4 corners. `8_check_oblique_vs_warped.png` alternates squares of the render and the warped map.

## Output

All in `out/camera-footprint/` (PNG, full size; the images here are small JPEG copies).

The scene: boreal forest with ponds and clearings on the west shore of a lake in the default
world (59.746° N, 15.248° E), 900 m above the ellipsoid (834 m above the ground), a June
morning; the oblique camera turned to 115° (east-south-east), tilted 50° from nadir and banked
4°; 1024 × 768 pixels, f = 900 px (59° wide); map tiles at zoom 15 (2.4 m pixels).

| | |
|---|---|
| ![](images/1_oblique_camera.jpg) **1** oblique camera | ![](images/2_nadir_camera.jpg) **2** nadir camera, same position: north up, east right |
| ![](images/3_map_albedo.jpg) **3** stitched albedo, tiles 15/17772..17776/9561..9565 | ![](images/4_map_albedo_footprint.jpg) **4** with the oblique camera's footprint |
| ![](images/5_map_warped_into_oblique.jpg) **5** the map warped into the oblique camera by the homography | ![](images/8_check_oblique_vs_warped.jpg) **8** check: squares of 1 and 5 alternating |
| ![](images/6_map_elevation.jpg) **6** stitched elevation, 63 m (black) .. 269 m (white) | ![](images/7_map_elevation_footprint.jpg) **7** with the footprint |

**Known defects in these images.** They show bugs in the current generator and renderer,
not intended looks: the lakes and ponds come out as flat grey or silvery sheets with hard edges,
there is a white streak and a light-blue band along the big lake's shore, and the nadir view has
a striped green ellipse (a clearing). Renders at the same poses looked different with v0.2.0.
The coordinates, tiles, footprint and homography do not depend on these defects.

What it prints (excerpt):

```text
== 2. Extrinsics: one position, two rotations
position  geodetic lat 59.745800°, lon 15.248000°, h 900.00 m  (833.7 m above the ground at 66.3 m)
          ECEF     (3108625.26, 847391.94, 5487039.97) m
nadir     camera axes in ENU: x(right) (+1.000, +0.000, +0.000)  y(down) (-0.000, -1.000, +0.000)  z(optical) (+0.000, +0.000, -1.000)
oblique   camera axes in ENU: x(right) (-0.373, -0.927, -0.045)  y(down) (-0.583, +0.272, -0.766)  z(optical) (+0.722, -0.260, -0.641)
...
oblique   renderer's pose vs ours: position 0.00e0 m, rotation 5.00e-16
...
== 4. Corner rays of the oblique camera ∩ terrain (elevation tiles at zoom 13, 6 iterations)
top-left     pixel (   -0.5,   -0.5)
             heights 66.31 -> 239.88 -> 236.34 -> 236.02 -> 235.99 -> 235.99 -> 235.99
             GPS  lat 59.748920°, lon 15.296405°, h 235.99 m
             ECEF (3107295.89, 849850.39, 5486641.55) m
             ENU  (2721.52, 348.65, -664.60) m  (range 2823 m)
top-right    pixel ( 1023.5,   -0.5)
             heights 66.31 -> 63.43 -> 63.43 -> 63.43 -> 63.43 -> 63.43 -> 63.43
             GPS  lat 59.725834°, lon 15.281202°, h 63.43 m
...
== 5. Ground corners -> fractional XYZ tiles at zoom 15
top-left     tile (x, y) = (17776.3127, 9561.3506)  -> in tile 15/17776/9561 at (80.1, 89.8) px
top-right    tile (x, y) = (17774.9290, 9565.5203)  -> in tile 15/17774/9565 at (237.8, 133.2) px
bottom-right tile (x, y) = (17772.2586, 9562.8195)  -> in tile 15/17772/9562 at (66.2, 209.8) px
bottom-left  tile (x, y) = (17772.8016, 9561.4134)  -> in tile 15/17772/9561 at (205.2, 105.8) px
coverage at zoom 15: x 17772..=17776 , y 9561..=9565  (5 x 5 tiles, 1280 x 1280 px)
...
== 9. Check: does the warped map line up with the rendered image?
221 grid pixels: homography vs rendered ground point: median 17.9 px, 90% 29.8 px, max 43.0 px (at (0, 256))
(the plane through the 4 corners stands in for terrain of 63 .. 253 m: relief, trees and buildings shift points)
```

The render and the warped map show the same scene: the lake shore, the ponds and clearings
continue across the squares of image 8, offset by up to a few tens of pixels. The ground here is
not a plane: the far left corner lies on the upland north of the lake at 236 m, the far right one
on the lake at 63 m, and the plane through the 4 corners cuts through the terrain between them,
so points off that plane land elsewhere in the warp (median 18 px, worst at the left edge).
Over level ground the offsets shrink to a few pixels (a camera looking over the lowland and the
lake alone gave a median of 1.4 px). In the render the trees also stand up in 3D.
