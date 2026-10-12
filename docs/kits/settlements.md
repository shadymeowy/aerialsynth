# Settlements kit: hamlets to megacities, airports, ports, dams, mines, energy

`crates/terragen/src/kits/settlements.rs` (the kit, host function, slots 7 and 8, tests),
`settlements/places.rs` (the central-place hierarchy), `settlements/town.rs` (the settlement
pixel model), `settlements/sites.rs` (infrastructure placement and looks), and the WGSL twin
`src/gpu/wgsl/kits/settlements.wgsl` (`st_*`, same names and order of operations). No biomes.
Design: [terrain-next.md §5.9, §6.1, §6.2, §6.4](../design/terrain-next.md); contract:
[kit-guide.md](../design/kit-guide.md).

## What it draws

**Places (host, analytic).** Five levels, each a 3D jittered lattice: metropolis (300 km cells,
13–32 km radius, 2–25 M people), city (90 km, 3.5–13 km), town (25 km, 1–3.6 km), village
(6 km, 180–800 m), hamlet (1.6 km, 45–150 m). Every cell has K hashed candidates (1–8 by level)
scored by the atlas' population potential × climate × coast access, then flatness on a ring of
smooth-elevation samples (Mediterranean and highland villages prefer hilltops) for the best
three; scores are quantized to 1/64 with the candidate hash as tie-breaker. Existence by level
density × population potential × culture (per archetype: hamlet, village and town multipliers,
e.g. many hamlets in dispersed cultures) × `landuse.towns`. Suppression is top-down and
bounded: same-level sites by priority (level, score, id) within the spacing or overlapping;
lower levels inside a larger place's footprint are removed, except towns and cities within 0.8
of a larger radius, which become **sub-centres** (their own core and districts: polycentric
conurbations). Population (Zipf-like per level) sets the radius; culture sets sprawl, form
and street orientation (survey-grid cultures keep north-south streets). Everything uses the
atlas and the smooth relief only, so `settlements_near(world, center, radius)` is the same on
both backends, cheap (memoized per lattice cell) and independent of tiles.

**Towns and cities (per pixel, `town.rs`).** A place stamp (centre, street frame, radius, form,
culture, development, population, downwind direction) drives:

| step | how |
|---|---|
| footprint | relative distance in the elongated frame / (R (1 + two noise octaves)); density 1 → 0 over its outer half |
| zones | CBD (cities: 7–16 % of R), inner, outer, suburb, peri-urban; industrial wedges: 1–2 angular sectors around the downwind direction |
| districts | W2 cells in the place frame (400 m villages … 1.4 km metropolises); their edges are arterials (4–9 m half width); per-district pattern, angle, block size, density multiplier (some districts dense, some half fields: the coarse look is a patchwork) |
| patterns | grid / rotated grid (jittered lines, main streets every fourth), radial-concentric (rings, spokes doubling per band), organic (warped W2 at 40–90 m; medinas narrow), curvilinear suburbs (warped long blocks, most cross streets missing), superblocks (280–390 m, slab and point towers in rows, green and parking), informal (W2 at 7.5 m: tiny corrugated roofs, alleys, thinning with the footprint), industrial (230–340 m blocks) |
| lots | hashed BSP of the block, depth 1 (CBD) … 4 (suburbs); a lot exists by the density at its centre (lots stay whole) |
| buildings | by zone × culture × lot: detached houses with setbacks (pools in rich suburbs, yard trees, porch lights), row houses, perimeter blocks with courtyards (European, Mediterranean, monsoon cultures), mid-rises, towers on podiums (40–440 m, by population), warehouses with tanks; gable or flat roofs, roof palettes by culture (terracotta, slate, metal, shingles, thatch, corrugated, whitewashed) |
| parcels | parks (paths, tree clumps, ponds), plazas, parking (stalls and cars), sports fields with athletics tracks, cemeteries, stadiums (stands, pitch, floodlights; cities), construction sites |
| trees | yard and park trees, street trees every 11 m in tree-lined cultures |
| shadows | a 3-step march toward the sun through the same pixel model (one copy in WGSL) |
| lights | lamps along the nearest street (sodium / warm / white LED by zone, culture and development), thinned with their energy kept where they would merge, their mean where blocks are unresolved; lit window rims (white in CBDs); floods in industry and stadiums |

Below ~22 m pixels the district's mean (zone colour, built fraction, height) replaces the
explicit blocks, below ~6 m the lot means replace sub-pixel buildings, so parents match their
children. The kit publishes the town masks (`town_urban`, `town_cov`) and clears natural
vegetation where it builds (the canopy is not drawn there).

**Villages by culture.** Nucleated (organic lanes, 52 m cells), perched hilltop villages (34 m
cells, dense, stone lanes; Mediterranean, highland), street villages (farms along one street,
house in front, barn behind; the street follows the valley), **forest-strip villages**
(Waldhufen: a street village in the temperate hills whose every farm owns a 40–60 m wide strip
up the valley side, 0.5–1.4 km long, fields in segments with hedges between the strips, the
forest beyond the strip ends), dispersed farms (yard, house, barn, shelter trees), kraals (a
ring of round huts around a cattle enclosure, a thorn fence; savanna pastoral), survey-grid
villages. Villages have a plaza with a church (nave and tower), a mosque (dome, minaret) or a
temple by culture. Lanes are paved or dirt by development.

**Sites (host placement, `sites.rs`).**

| site | placement | look |
|---|---|---|
| airports | cities (developed), metropolises, large towns: 16 candidates on 8 bearings beyond the edge, the flattest by 5 point samples; runways along the atlas wind ± hashed | 1 runway (towns), 2 parallel (cities > 1 M), 2 parallel + crossing (metropolises): markings, piano keys, touchdown bars, taxiways and connectors, apron with aircraft, terminal, hangars, car park, mown grass, perimeter road; the graded area flattened to the plane with a 120 m cut-and-fill margin; edge, approach and taxiway lights, apron floods |
| ports | coastal towns and cities: the nearest ocean along 8 bearings by point-sample rays (rivers crossed), the coast direction from two lateral rays | land: container stacks, warehouses, gantry cranes, quay; water (slot 8, solid): piers every 260 m, ships alongside, a breakwater; floodlights |
| dams, reservoirs | 15 % of towns, 50 % of cities, 20 % of a 55 km mountain lattice: the narrowest valley cross-section among 6 level 0–1 drainage pieces (point samples), the crest below the lower bank (25–150 m) | the reservoir is the upstream drainage (≤ 40 pieces of the same level) within a reach of 35 × the dam height, upstream of the dam line, below the water level: dendritic by the ground; a pale bathtub ring up to the crest; the concrete wall across the valley (slot 8) |
| quarries, mines | a 16 km lattice near towns (32 %) or in the hills: lithology picks limestone quarries, granite quarries, other quarries, open-pit mines (hard rock, 30 %) | an irregular stepped bowl (benches 10–15 m, faces, a spiral haul road), a pit lake, a spoil heap, tailings ponds with beaches (mines); DSM only (pass B) |
| solar farms | up to 2 per town in sunny, dry, developed regions, beyond the edge, flat by 5 point samples | panel rows facing the equator (resolved below the row pitch, their mean above), service roads, fence |
| wind farms | a 45 km lattice: windy (atlas wind 3.5–7 m/s), developed, by culture, away from towns | turbines on a jittered grid (rows across the wind): white tower and blades, a long shadow, gravel pad and track, red aviation lights |
| gas flares | a 30 km lattice: arid sedimentary basins | well pads and tracks, a scorched ring, a bright orange flare at night |

Greenhouse seas are drawn by the agriculture kit (§5.8); roads, rail, bridges, power lines and
cars belong to the transport kit.

## Placement and levels of detail

Places of a level are listed (as stamps) for tile areas whose pixel is below 150 m (hamlets),
800 m (villages), 3 km (towns), 12 km (cities), 50 km (metropolises): a place covers at most
~0.2 pixel when it is dropped. Sites: airports and ports below 600 m, dams below 1.2 km,
quarries and solar farms below 300 m, wind farms below 150 m, flares below 2.5 km. All host
results are memoized per lattice cell or place (pure functions of the world).

## Decisions

* **Host stamps, not instance families.** The hierarchy needs neighbour suppression and the
  sites need terrain samples; both are host Rust shared by the backends. Pass B reads a place
  stamp per bin and evaluates one place (the most central) per sample.
* **The core's grid towns are off with the kit** (`Kit::replaces_towns`, `kits::legacy_towns()`,
  WGSL `KITS_LEGACY_TOWNS`): no town site in pass A, no town requests, no town code in pass B.
  The core's farmsteads stay (dispersed farmsteads between the hamlets).
* **No river access in the scores**: the CPU drainage per candidate costs too much for an
  analytic API; water access comes from the atlas coast distance. Dams use the CPU drainage
  (`World::river_segments`, levels 0–1) on the host, once per dam source.
* **Quarries change the DSM only** (pass B): the drainage does not see the pits.
* **Pass B in four stages** (`[0, 3, 5, 7]`): the kit's built layer is in the linear–built stage,
  its water layer (reservoirs, dams, port piers) in the canopy–seasonal stage.

## Places

Places of the default world (`PLACES` in the tests): metropolis (30.35, 128.849) and its outer
districts (30.45, 128.95), survey-grid city (−6.177, −114.240), desert town with a medina
(−12.198, −21.454), city with informal fringes (−20.865, −142.615), forest-strip village
(20.715, −22.646), street village (−9.162, 96.553), kraal village (−15.971, 62.836), perched
village (−38.778, −166.808), airport (−10.539, 115.357), port (−11.717, −173.520), quarry
(−20.458, −152.138), solar farm (40.497, 46.920), wind farm (42.418, 33.345), gas flares
(24.031, −57.483), dam (−0.699, −17.687). Find more with the ignored tests
`find_settlement_places` (places by level, form and culture), `find_sites` and `debug_ports`.

## Tests

`places::tests` (determinism and tile independence, the hierarchy's counts, no overlaps except
sub-centres), stamp encoding, BSP partition, legacy towns off, LOD consistency, seams E-W /
N-S, determinism (fresh generators and caches), `tiles_match_the_cpu` (22 tiles at z11–z17),
`places_stay_on_the_gpu` (z2–z16 at every place), ignored `bench_places`. No kit kernels (no
kernel parity test).

## Known limits

* Places score flatness and coast access, not rivers; a town may sit on a lake shore or across
  a small valley, and its lots do not follow the terrain beyond the slope gate and the river
  banks.
* Streets are not connected between districts (district edges are arterials) nor to the core's
  rural roads; the transport kit's graph will link places.
* Towers cast shadows only within their own block (3 steps of 3.5 m); tall towers' long shadows
  across streets are missing.
* Reservoirs fill every ground below the water level within their reach upstream of the dam,
  also a neighbouring basin; the drainage and rivers in pass A do not know them (a river is
  drawn below the water, the reservoir over it). Dams use point samples, so a dam can sit on a
  wide valley floor where the banks are far.
* Airports flatten the DSM, not pass A: rivers and lakes inside the graded area stay; a river
  can cross a runway.
* Ports exist where a point-sample ray finds the ocean within R + 12 km (10 of 13 coastal towns
  in `debug_ports`); estuaries and lakes have no ports.
* Quarries are pass-B pits; their floors are not drainage sinks.
* Hamlets below 150 m pixels, villages below 800 m, towns below 3 km are not drawn (their lights
  vanish from night views at those zooms).
* Night lights: no lit facades of towers beyond the window rims, no interchange lighting
  (transport kit).
