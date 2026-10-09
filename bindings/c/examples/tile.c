/*
 * aerialsynth C API example: read (or generate) a tile and print statistics of its layers.
 *
 *   cc -std=c99 tile.c -I../include -L../../../target/release -laerialsynth \
 *      -Wl,-rpath,$PWD/../../../target/release -o tile
 *   ./tile out/world.h5 [CONFIG.yaml|- [Z X Y]]
 *
 * CONFIG.yaml is a scenario (its `world:` section) or a world config; "-" or nothing = the default
 * world. The tile store is created if missing; the tile is generated and stored on first use.
 */
#include "aerialsynth.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc < 2 || argc == 4 || argc == 5 || argc > 6) {
        fprintf(stderr, "usage: %s TILES.h5 [CONFIG.yaml|- [Z X Y]]\n", argv[0]);
        return 2;
    }
    const char *config = (argc > 2 && strcmp(argv[2], "-") != 0) ? argv[2] : NULL;
    uint32_t z = 3, x = 5, y = 3;
    if (argc == 6) {
        z = (uint32_t)strtoul(argv[3], NULL, 10);
        x = (uint32_t)strtoul(argv[4], NULL, 10);
        y = (uint32_t)strtoul(argv[5], NULL, 10);
    }
    printf("aerialsynth %s\n", as_version());

    as_world *w = as_open(argv[1], config, -1 /* keep the config's seed */);
    if (!w) {
        fprintf(stderr, "as_open: %s\n", as_last_error());
        return 1;
    }
    int status = 1;
    float *elev = NULL;
    uint8_t *rgb = NULL;

    /* elevation: f32 metres above the WGS84 ellipsoid (little-endian, as on this host) */
    size_t n = as_layer_size(AS_LAYER_ELEVATION);
    elev = malloc(n);
    if (!elev) goto done;
    int rc = as_tile(w, z, x, y, AS_LAYER_ELEVATION, elev, n);
    if (rc != AS_OK) {
        fprintf(stderr, "as_tile(elevation): %d %s\n", rc, as_last_error());
        goto done;
    }
    size_t npx = (size_t)AS_TILE_SIZE * AS_TILE_SIZE;
    float lo = elev[0], hi = elev[0];
    double sum = 0.0;
    for (size_t i = 0; i < npx; i++) {
        if (elev[i] < lo) lo = elev[i];
        if (elev[i] > hi) hi = elev[i];
        sum += elev[i];
    }
    printf("tile %u/%u/%u elevation: min %.1f m, max %.1f m, mean %.1f m\n", z, x, y, lo, hi, sum / (double)npx);

    /* rgb: u8 x 3, row-major (row 0 = north) */
    as_layer_info info;
    if (as_layer_describe(AS_LAYER_RGB, &info) != AS_OK) goto done;
    rgb = malloc(info.size);
    if (!rgb) goto done;
    rc = as_tile(w, z, x, y, AS_LAYER_RGB, rgb, info.size);
    if (rc != AS_OK) {
        fprintf(stderr, "as_tile(rgb): %d %s\n", rc, as_last_error());
        goto done;
    }
    double mean[3] = {0.0, 0.0, 0.0};
    for (size_t i = 0; i < npx; i++)
        for (uint32_t c = 0; c < info.channels; c++) mean[c] += rgb[i * info.channels + c];
    printf("tile %u/%u/%u %s: %u channels, %zu bytes, mean colour (%.0f, %.0f, %.0f)\n", z, x, y, info.name, info.channels, info.size,
           mean[0] / (double)npx, mean[1] / (double)npx, mean[2] / (double)npx);

    /* errors: a zoom above the world's max zoom is refused */
    rc = as_tile(w, (uint32_t)as_max_zoom(w) + 1, 0, 0, AS_LAYER_RGB, rgb, info.size);
    printf("zoom %d: %s (%d: %s)\n", as_max_zoom(w) + 1, rc == AS_ERR_INVALID_ARGUMENT ? "refused" : "unexpected", rc, as_last_error());
    status = rc == AS_ERR_INVALID_ARGUMENT ? 0 : 1;

done:
    free(rgb);
    free(elev);
    as_close(w);
    return status;
}
