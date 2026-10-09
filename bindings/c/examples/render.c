/*
 * aerialsynth C API example: render a camera image of a world and write it as a PPM.
 *
 *   cc -std=c99 render.c -I../include -L../../../target/release -laerialsynth \
 *      -Wl,-rpath,$PWD/../../../target/release -o render
 *   ./render TILES.h5 [CONFIG.yaml|- [LAT LON HEIGHT_ABOVE_GROUND [OUT.ppm]]]
 *
 * CONFIG.yaml is a scenario (its `world:` section) or a world config; "-" or nothing = the default
 * world. A 160 x 120 pinhole camera (70 degree field of view) looks north-east, 30 degrees down,
 * from HEIGHT_ABOVE_GROUND metres (default 1500) above the surface at LAT, LON (default 45, 10),
 * on 2026-06-21 at 07:30 UTC. Tiles the view needs are generated into the store on first use.
 */
#include "aerialsynth.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WIDTH 160
#define HEIGHT 120

int main(int argc, char **argv) {
    if (argc < 2 || argc == 4 || argc == 5 || argc > 7) {
        fprintf(stderr, "usage: %s TILES.h5 [CONFIG.yaml|- [LAT LON HEIGHT_ABOVE_GROUND [OUT.ppm]]]\n", argv[0]);
        return 2;
    }
    const char *config = (argc > 2 && strcmp(argv[2], "-") != 0) ? argv[2] : NULL;
    double lat = 45.0, lon = 10.0, above = 1500.0;
    if (argc >= 6) {
        lat = strtod(argv[3], NULL);
        lon = strtod(argv[4], NULL);
        above = strtod(argv[5], NULL);
    }
    const char *out_path = argc == 7 ? argv[6] : "render.ppm";

    as_world *w = as_open(argv[1], config, -1);
    if (!w) {
        fprintf(stderr, "as_open: %s\n", as_last_error());
        return 1;
    }
    int status = 1;
    as_camera *cam = NULL;
    uint8_t *rgb = NULL;
    float *depth = NULL;
    FILE *f = NULL;

    /* the height of the surface (ground, trees, buildings, water) below the camera */
    double ground = 0.0;
    if (as_surface_height(w, lat, lon, &ground) != AS_OK) {
        fprintf(stderr, "as_surface_height: %s\n", as_last_error());
        goto done;
    }

    /* forward mount: the pose's yaw / pitch are the heading / elevation of the optical axis */
    cam = as_camera_pinhole(w, WIDTH, HEIGHT, 70.0, AS_MOUNT_FORWARD, AS_BACKEND_DEFAULT);
    if (!cam) {
        fprintf(stderr, "as_camera_pinhole: %s\n", as_last_error());
        goto done;
    }
    uint32_t width = 0, height = 0;
    as_camera_size(cam, &width, &height);
    printf("camera %ux%u on the %s\n", width, height, as_camera_backend(cam) == AS_BACKEND_GPU ? "GPU" : "CPU");

    size_t n = (size_t)width * height;
    rgb = malloc(n * 3);
    depth = malloc(n * sizeof(float));
    if (!rgb || !depth) goto done;
    as_pose pose = {lat, lon, ground + above, 0.0 /* roll */, -30.0 /* pitch */, 45.0 /* yaw */};
    double t = 1782027000.0; /* 2026-06-21T07:30:00Z */
    int rc = as_render(cam, &pose, t, rgb, n * 3, depth, n, NULL, 0);
    if (rc != AS_OK) {
        fprintf(stderr, "as_render: %d %s\n", rc, as_last_error());
        goto done;
    }

    /* depth: z-depth in metres, +inf = sky */
    size_t sky = 0;
    float near = INFINITY, far = 0.0f;
    for (size_t i = 0; i < n; i++) {
        if (isinf(depth[i])) {
            sky++;
            continue;
        }
        if (depth[i] < near) near = depth[i];
        if (depth[i] > far) far = depth[i];
    }
    printf("ground %.1f m; depth %.0f .. %.0f m, sky %.1f%%\n", ground, near, far, 100.0 * (double)sky / (double)n);

    /* rgb: u8 x 3, row 0 = image top */
    f = fopen(out_path, "wb");
    if (!f) {
        perror(out_path);
        goto done;
    }
    fprintf(f, "P6\n%u %u\n255\n", width, height);
    if (fwrite(rgb, 1, n * 3, f) != n * 3) {
        perror(out_path);
        goto done;
    }
    printf("wrote %s\n", out_path);

    /* errors: a latitude out of range is refused */
    as_pose bad = pose;
    bad.lat_deg = 91.0;
    rc = as_render(cam, &bad, t, rgb, n * 3, NULL, 0, NULL, 0);
    printf("latitude 91: %s (%d: %s)\n", rc == AS_ERR_INVALID_ARGUMENT ? "refused" : "unexpected", rc, as_last_error());
    status = rc == AS_ERR_INVALID_ARGUMENT ? 0 : 1;

done:
    if (f) fclose(f);
    free(depth);
    free(rgb);
    as_camera_close(cam);
    as_close(w);
    return status;
}
