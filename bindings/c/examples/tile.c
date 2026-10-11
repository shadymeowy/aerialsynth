/*
 * aerialsynth C API example: read (or generate) a tile and print statistics of its layers.
 *
 * In the source tree (after `cargo build --release -p aerialsynth-capi`), from the repository root:
 *   cc -std=c99 bindings/c/examples/tile.c -I bindings/c/include -L target/release -laerialsynth \
 *      -Wl,-rpath,$PWD/target/release -o tile
 * In a release archive (terrain-<version>-<target>), from its top directory:
 *   cc -std=c99 examples/tile.c -I include -L lib -laerialsynth -Wl,-rpath,$PWD/lib -o tile
 *
 *   ./tile out/world.h5 [CONFIG.yaml|- [Z X Y]]
 *
 * CONFIG.yaml is a scenario (its `world:` section) or a world config; "-" or nothing = the default
 * world. The tile store is created if missing; the tile is generated and stored on first use.
 * Then the 2 x 2 tiles from Z/X/Y (to the east and south) are read in one call (as_tiles).
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
    float *elev = NULL, *block = NULL;
    uint8_t *rgb = NULL;

    /* the in-memory cache of decoded tiles (default AS_DEFAULT_CACHE_MB; 0 = off) */
    if (as_set_cache_mb(w, 64) != AS_OK) goto done;

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

    /* many tiles in one call: 2 x 2 tiles (x, x + 1) x (y, y + 1) of elevation (wrapping at the
       edge), generated together where missing; tile i at offset i * n */
    uint32_t side = 1u << z, zxy[4 * 3];
    for (uint32_t i = 0; i < 4; i++) {
        zxy[3 * i] = z;
        zxy[3 * i + 1] = (x + i % 2) % side;
        zxy[3 * i + 2] = (y + i / 2) % side;
    }
    block = malloc(4 * n);
    if (!block) goto done;
    rc = as_tiles(w, zxy, 4, AS_LAYER_ELEVATION, block, 4 * n);
    if (rc != AS_OK) {
        fprintf(stderr, "as_tiles: %d %s\n", rc, as_last_error());
        goto done;
    }
    printf("as_tiles: 4 tiles, the first %s as_tile\n", memcmp(block, elev, n) == 0 ? "equal to" : "DIFFERENT from");
    if (memcmp(block, elev, n) != 0) goto done;

    /* errors: a zoom above the world's max zoom is refused */
    rc = as_tile(w, (uint32_t)as_max_zoom(w) + 1, 0, 0, AS_LAYER_RGB, rgb, info.size);
    printf("zoom %d: %s (%d: %s)\n", as_max_zoom(w) + 1, rc == AS_ERR_INVALID_ARGUMENT ? "refused" : "unexpected", rc, as_last_error());
    status = rc == AS_ERR_INVALID_ARGUMENT ? 0 : 1;

done:
    free(block);
    free(rgb);
    free(elev);
    as_close(w);
    return status;
}
