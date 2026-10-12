# Transport kit: roads, railways, power lines

`crates/terragen/src/kits/transport.rs` (the kit, slots 7 and 8), `transport/net.rs` (the graph),
`transport/route.rs` (routing), `transport/build.rs` (the features of an area: profiles,
segments, crossings, interchanges, stations, yards, site roads, power lines), `transport/px.rs`
(the looks), `transport/tests.rs`, and the WGSL twin `src/gpu/wgsl/kits/transport.wgsl` (`tr_*`,
same names and order of operations). No biomes, no kernels. Design:
[terrain-next.md §3.5, §6.3, §6.4, P2/P3](../design/terrain-next.md); contract:
[kit-guide.md](../design/kit-guide.md). The kit builds on the settlements kit
([settlements.md](settlements.md)): its places are the graph's nodes, its sites the
destinations.

## What it draws

**The graph (host, analytic).** Relative-neighbourhood graphs (an edge a–b exists iff no node of
the network is closer to both than they are to each other) per network, each with a longest
edge; every pair of places belongs to one network:

| network | nodes | longest edge | class |
|---|---|---|---|
| arterial | cities, metropolises | 420 km | motorway (both developed ≥ 0.35, shorter with less development), else trunk |
| regional | towns and larger | 90 km | trunk (secondary where development < 0.12) |
| local | villages and larger | 20 km | secondary (local where development < 0.08) |
| hamlet | hamlets and larger | 4.5 km | local road (paved where developed) |
| track | a 1.3 km lattice of track ends in rural land outside places, hamlets, villages | 2.2 km | farm / forest track |
| rail | cities, metropolises, developed towns > 40 k | 300 km | railway (double track between developed cities, electrified where development > 0.55) |
| grid | developed cities | 300 km | high-voltage power line |

Plus: **spurs** from towns within 1.5 R + 4 km of a passing motorway (a trunk road ending in a
trumpet); roads to the settlements kit's **airports** (trunk for cities, secondary for towns),
**ports** (secondary road; a rail spur and a yard for cities), **dams** (local road to the wall),
**quarries** (haul road to the rim), **solar** and **wind farms** (local roads); **power lines**
from dams, solar and wind farms to the nearest town or city > 25 k within 55 km. Local roads and
tracks do not cut through towns (their streets do); site roads stop at the site's edge.

**Routing (host, analytic).** A dynamic programme on a corridor grid between the ends (≤ 41
nodes along × 11 across, monotone along the edge, lateral steps ≤ 0.45 of the step along),
refined piece by piece where the coarse grid is coarser than 260 m (≤ 41 × 7 per piece); cost
= length × (1 + k g² + k₂ (g − g_lim)²₊) + sea + towns that are not ends (motorways, trunk roads,
railways and power lines go around them) + turns; major roads and railways change their
lateral step by at most one per node (no hairpins) and are smoothed by a moving average (400,
250, 600 m) and Chaikin; minor roads by Chaikin ×3. Steeper than they may climb (secondary 11 %,
local 13 %, tracks 16 %), minor roads get **switchbacks**: zigzag legs across the slope with
rounded hairpins, long enough to keep the grade. Power lines run straight between angle points.
Costs use the CPU-analytic relief (`World::relief_height`, no drainage): routes are the same on
both backends, memoized per edge.

**Engineered profiles (host, point samples).** Motorways, trunk roads and railways sample the
full terrain every 40 m (one GPU round for all samples of an area; memoized per sample) and
derive deck heights by a local operator: a box mean of the ground (raised over water), then the
mean of its largest grade-limited minorant and smallest majorant within ±2.4 km (grades 4.5 %,
6.5 %, 2 %). Any stretch of a route gets the same decks, so tiles agree. Where the deck is 6 m
above the ground (or over water) the stretch is a **bridge** (deck only, parapets, its shadow
on the ground and water below); 25 m below the ground over at least three samples a **tunnel**
(not drawn; **portals**: a concrete headwall and the dark mouth). Elsewhere the carriageway sits
on the deck and **embankments / cuttings** (1:2, grass on fills, rock and soil with strata in
cuts) join it to the ground in the DSM. Profiles are computed for areas with pixels below 40 m;
coarser tiles draw the roads on the ground (sub-pixel there).

**Crossings.** Where routes cross away from places: a motorway over or under everything; minor
roads go over on a bridge with 5 % approach ramps (7.5 m clearance); at motorway × motorway a
**cloverleaf**, motorway × trunk a **partial cloverleaf**, motorway × secondary a **diamond**,
spurs a **trumpet** (ramps as segments with lit carriageways, loops of 62 m radius); railways
pass under roads ≥ secondary, level crossings with local roads and tracks; trunk and secondary
roads meet at grade, with a **roundabout** in roundabout cultures (Mediterranean and
temperate-village archetypes, a third of other developed ones). Roundabouts also mark where
trunk and secondary roads enter towns of those cultures. No ramps inside towns.

**Railways.** Ballast, two rails per track at standard gauge and sleepers (visible below
~0.3 m pixels, their mean above), catenary masts, now and then a train. **Stations** where a
line ends (once per place; 3–8 tracks with platforms, canopies, the station building and its
forecourt) and where it passes through a town; **yards** at cities (8–18 tracks fanning out in a
ladder, strings of wagons, floodlights) and at ports.

**Power lines.** A corridor cleared of trees (44 m high-voltage, 28 m from the sites), lattice
pylons every 350 / 300 m (footprint, bracing and cross-arm; in the DSM) with their tapered
shadows and the arm's, three conductors sagging between the towers and their shadows.

**Cars and lights.** Cars in their lanes on slots of 24 / 30 m (density by class and development,
×1.5–2.4 in towns by size; trucks on motorways and trunk roads; none on tracks), their shadows,
the mean of unresolved cars at coarse pixels; parked cars in the settlements kit's parking lots
and airport car parks (`px::parked`). Lamps every 40 m on motorway medians near cities and
everywhere in highly developed regions, on interchange ramps, every 35 m along roads inside
towns (sodium or LED by road), thinned with their energy kept where they would merge.

**Cross-sections.** Motorways: two carriageways of 2–3 lanes, a concrete median barrier, edge and
lane lines, hard shoulders, wide verges; trunk and secondary roads: centre dashes, edge lines;
local roads paved or gravel; tracks: two ruts and a grass strip; urban stretches: sidewalks and
curbs instead of verges. Verges and slopes clear the trees; towns are drawn over by the roads
(the kit runs in slot 7 after the settlements kit), so the rural network continues into the
towns' street grids to their centres.

## Levels of detail

Listed for areas whose pixel is below (m): motorway 600, trunk 240, secondary and ramps 160, local
110, rail 200, tracks 60, power lines 800 — each covers ≤ ~5 % of a pixel above. Polylines are
emitted with vertices one per 20 m × 2^k ≥ 1.5 pixels (a zoom's choice), coverages are exact box
filters of the sub-sample footprint, so a parent is the mean of its children. Along-distances are
kept modulo 8.4 km (every pattern period divides it).

## Decisions

* **Host-built graph, analytic routing.** The design's §6.3 pipeline as specified, with RNGs per
  network instead of per road cell (an edge is a pure function of the node set; no cell
  ownership needed) and refinement per coarse piece so motorways adapt locally.
* **Slot 7, after the settlements kit**, not slot 6: drawn after the towns, the roads continue
  into them over the blocks without exposing what lies under the town at their edges
  (compositing a town cut by the road left the ground showing at antialiased edges). Site roads
  stop at the site's box so they do not cross runways.
* **The core's noise-network roads are off** with the kit (`Kit::replaces_roads`,
  `kits::legacy_roads()`, WGSL `KITS_LEGACY_ROADS`: no iso-line distances in pass A, no road
  grid in the pre pass); the core's farm tracks along land-use region borders stay.
* **Bridges over water per pixel** (slot 8): wherever a road or railway crosses a river,
  reservoir, lake or the sea it gets a deck above the water (the deck from the profile where
  known), so small streams under local roads need no host knowledge.
* **Volcanoes:** on `s.no_town` ground (cones, lava) only motorways, trunk roads and railways.
* **Deck sentinel:** "no deck" is −1e7, not NaN (WGSL cannot test for NaN).

## Places

Places of the default world (`PLACES` in the tests): …

## Tests

`box_filters_are_exact`, `graph_is_a_relative_neighbourhood_graph` (symmetry, no blockers,
determinism), `edges_are_tile_independent`, `routes_are_deterministic`,
`profiles_are_local_and_grade_limited`, `features_are_tile_independent` (overlapping areas give
the same segments bit for bit; cleared caches), `ramps_connect`, LOD consistency, seams E-W /
N-S, determinism (fresh generators and caches), `tiles_match_the_cpu`, `places_stay_on_the_gpu`;
ignored: `find_transport_places`, `find_transport_places2`, `render_tiles`, `debug_parity`.

## Compile time

…

## Stills

…

## Known limits

…
