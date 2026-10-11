# Coast and ocean kit: cliffs, skerries, tidal flats, barrier islands, reefs, atolls, ponds

`crates/terragen/src/kits/coast.rs` (types, creeks, relief, atoll family, layers, tests), its WGSL
twins `src/gpu/wgsl/kits/coast_relief.wgsl` (types, creeks, relief, atoll family) and
`coast.wgsl` (the layers). No biomes (design: [terrain-next.md §5.5](../design/terrain-next.md),
contract: [kit-guide.md](../design/kit-guide.md)).

## What it draws

**Coast type per ~50 km stretch.** Three noises (55, 38 and 47 km) × the relief potential
(atlas uplift, the mountain mask) × glaciation × the sea-level temperature give smooth weights of
five types (the strongest wins, near-ties blend): cliff (mountain / tectonic coasts), skerry
(glaciated), tidal flat (low coasts), barrier (low coasts, the other half of the noise), and
reef (warm water, on top of any type). Every effect is a function of the pre-remap height `h0`
near sea level, so none needs a distance to the coast; all run in pass A (`relief`) and gate on
the atlas coast distance (−100..30 km) and on a window of `h0` before any noise.

| feature | pass A (height remap) | pass B |
|---|---|---|
| cliffs | `h + C·smoothstep(0, ε, h)`, C 12-112 m, fading above 1.5-4 C, ε = max(2 m, 0.04 gsd); below sea level a wave-cut platform (`-1.3 + 0.1 h` in the first 10 m of depth) | rock face on slopes > 0.4 (strata by lithology: chalk, red sandstone, granite, basalt, alluvium; streaks), pale shallow water over the platform (core surf at the foot) |
| rocky shores, skerries | noise 700 / 230 / 75 m in a window around -8 m: islets and coves | bare rock and lichen on the islets, kelp patches in the water |
| archipelagos | the same at 2.6 km / 900 m / 260 m, wider window (island-arc coasts) | |
| tidal flats, salt marsh | the intertidal band compressed to a fifth of its slope; dendritic creeks (3 levels of ridged noise, 14 / 6 / 2.4 m wide) cut 1.9 m | grey-brown mud (wet darker), marsh above 1 m (green with rust patches), creeks as dark channels, no towns |
| barrier islands, lagoons | bump at -6 m (8.7 m, width 2.8 m), inlets where a 3.2 km noise closes it | the core's beach on the island, green-sand lagoon water |
| reefs | bump at -7 m (fringing) or -28 m (barrier reef) up to -0.9 m, passes where a noise closes it | turquoise sand by depth, coral heads, the core's surf on the crest |
| salt / aquaculture ponds | | on the high flats and marsh, in the land-use region's field frame: 150 × 85 m ponds with dykes; salt (pink, white, turquoise, green) where moist < 0.45, aquaculture (blue-greens) elsewhere; patches by a 2.4 km noise |
| atolls | family `coast_atolls` (cell 80 km, reach 24 km): ring of reef R 2-14 km, rim 180-400 m wide at -0.3 m, motus (+1.9 m, noise > threshold), lagoon 12-30 m, outer slope 0.5 to the sea floor; ocean floor 1.5-5.5 km deep, > 150 km from any coast, SST > 25 °C, mostly hotspot tracks | turquoise lagoon with coral, scrub on the motus (the core's beach is the sand) |

Creek, ring and rim widths below a pixel are widened or show their mean (creeks fade to a mean
cover of 5.5 % by pixel size, no step between zooms).

## Placement

Weights, not decisions: a coast has a type by the atlas fields at the pixel, so tiles agree
across zooms and backends. Atolls are an instance family (existence at the site from the atlas).

