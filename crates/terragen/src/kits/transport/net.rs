//! The network graph (`docs/design/terrain-next.md` §6.3): nodes are the settlements kit's places
//! (`places::places_near`) and a lattice of track ends; edges are relative-neighbourhood graphs
//! per network (cities: motorways and trunk roads, towns: trunk roads, villages: secondary roads,
//! hamlets: local roads, track ends: tracks, cities and large towns: railways, cities: the power
//! grid). An edge (a, b) of a network exists iff no node c of that network is closer to both
//! than they are to each other: a pure function of the node set, so every tile sees the same
//! edges. Neighbour lists are memoized per node.

use super::cls;
use crate::kits::settlements::places::{self, Place};
use crate::noise::*;
use crate::world::World;
use glam::DVec3;
use std::sync::{Arc, RwLock};

/// A node of a network.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Node {
    pub id: u64,
    pub p: DVec3,
    /// 0 hamlet … 4 metropolis; 5: a track end (not a place)
    pub level: u8,
    pub pop: f64,
    pub radius: f64,
    pub dev: f64,
    pub arch: u8,
}

pub const TRACK_END: u8 = 5;

impl Node {
    pub fn of(p: &Place) -> Node {
        Node { id: p.id, p: p.center, level: p.level, pop: p.population, radius: p.radius, dev: p.dev, arch: p.arch }
    }
}

/// The networks.
pub mod network {
    pub const ARTERIAL: usize = 0;
    pub const REGIONAL: usize = 1;
    pub const LOCAL: usize = 2;
    pub const HAMLET: usize = 3;
    pub const TRACK: usize = 4;
    pub const RAIL: usize = 5;
    pub const GRID: usize = 6;
    pub const N: usize = 7;
}

/// Longest edge (m) per network.
pub const LMAX: [f64; network::N] = [420_000.0, 90_000.0, 20_000.0, 4_500.0, 2_200.0, 300_000.0, 300_000.0];

/// The finest class a network can produce (its visibility: `cls::GSD_MAX`).
pub fn net_class_max_gsd(n: usize) -> f64 {
    match n {
        network::ARTERIAL => cls::GSD_MAX[cls::MOTORWAY],
        network::REGIONAL => cls::GSD_MAX[cls::TRUNK],
        network::LOCAL => cls::GSD_MAX[cls::SECONDARY],
        network::HAMLET => cls::GSD_MAX[cls::LOCAL],
        network::TRACK => cls::GSD_MAX[cls::TRACK],
        network::RAIL => cls::GSD_MAX[cls::RAIL],
        _ => cls::GSD_MAX[cls::POWER],
    }
}

/// Track-end lattice cell (m).
pub const TRACK_CELL: f64 = 1_300.0;

/// The track end of a lattice cell (rural land with people; none in the sea or on ice).
fn track_end(w: &World, c: (i64, i64, i64)) -> Option<Node> {
    let h = hash3(w.seed() ^ 0x7AC4, c.0, c.1, c.2);
    let q = DVec3::new(c.0 as f64 + 0.5 + 0.8 * (u01k(h, 1) - 0.5), c.1 as f64 + 0.5 + 0.8 * (u01k(h, 2) - 0.5), c.2 as f64 + 0.5 + 0.8 * (u01k(h, 3) - 0.5))
        * TRACK_CELL;
    let s = super::route::surface(w, q);
    if (q - s).length() >= 0.5 * TRACK_CELL {
        return None;
    }
    let a = w.atlas().sample(s.normalize());
    let pe = 0.7 * smoothstep(0.0, 0.3, a.population) * (1.0 - smoothstep(0.3, 0.6, a.glaciation)) * smoothstep(-8.0, 2.0, a.temp_c);
    if a.coast_km < 0.5 || u01k(h, 4) >= pe {
        return None;
    }
    if w.smooth_relief(s, 1_000.0).0 < 2.0 {
        return None;
    }
    Some(Node { id: h | 1, p: s, level: TRACK_END, pop: 0.0, radius: 0.0, dev: a.development, arch: a.archetype as u8 })
}

fn track_ends(w: &World, center: DVec3, radius: f64) -> Vec<Node> {
    let rr = radius + 0.5 * TRACK_CELL;
    let lo = ((center - DVec3::splat(rr)) / TRACK_CELL).floor();
    let hi = ((center + DVec3::splat(rr)) / TRACK_CELL).floor();
    let mut v = Vec::new();
    for z in lo.z as i64..=hi.z as i64 {
        for y in lo.y as i64..=hi.y as i64 {
            for x in lo.x as i64..=hi.x as i64 {
                let cc = (DVec3::new(x as f64, y as f64, z as f64) + 0.5) * TRACK_CELL;
                if (cc.length() - w.ell.a).abs() > 30_000.0 || (cc - center).length() > rr + 0.87 * TRACK_CELL {
                    continue;
                }
                if let Some(n) = track_end(w, (x, y, z)) {
                    if (n.p - center).length() <= radius {
                        v.push(n);
                    }
                }
            }
        }
    }
    v
}

/// The nodes of network `n` within `radius` of `center`, by id.
pub fn nodes(w: &World, n: usize, center: DVec3, radius: f64) -> Vec<Node> {
    let min_level = match n {
        network::ARTERIAL | network::RAIL | network::GRID => places::CITY,
        network::REGIONAL => places::TOWN,
        network::LOCAL => places::VILLAGE,
        _ => places::HAMLET,
    };
    let max_level = if n == network::TRACK { places::VILLAGE } else { places::METRO };
    let mut v: Vec<Node> = Vec::new();
    for level in min_level as usize..=max_level as usize {
        for p in places::places_near(w, level, center, radius) {
            v.push(Node::of(&p));
        }
    }
    if n == network::RAIL {
        // large developed towns have stations too
        for p in places::places_near(w, places::TOWN as usize, center, radius) {
            if p.population > 40_000.0 && p.dev > 0.35 {
                v.push(Node::of(&p));
            }
        }
        v.retain(|x| x.dev > 0.2);
    }
    if n == network::GRID {
        v.retain(|x| x.dev > 0.3);
    }
    if n == network::TRACK {
        v.extend(track_ends(w, center, radius));
    }
    v.sort_unstable_by_key(|x| x.id);
    v.dedup_by_key(|x| x.id);
    v
}

/// Does network `n` connect `a` and `b` (pair filter: each pair belongs to one network)?
fn pair_ok(n: usize, a: &Node, b: &Node) -> bool {
    match n {
        network::REGIONAL => !(a.level >= places::CITY && b.level >= places::CITY),
        network::LOCAL => !(a.level >= places::TOWN && b.level >= places::TOWN),
        network::HAMLET => !(a.level >= places::VILLAGE && b.level >= places::VILLAGE),
        network::TRACK => a.level == TRACK_END || b.level == TRACK_END,
        _ => true,
    }
}

/// The class of the edge (a, b) of network `n`.
pub fn edge_class(n: usize, a: &Node, b: &Node) -> usize {
    let dev = a.dev.min(b.dev);
    match n {
        network::ARTERIAL => {
            if dev >= 0.35 && (a.p - b.p).length() < 320_000.0 + 200_000.0 * dev {
                cls::MOTORWAY
            } else {
                cls::TRUNK
            }
        }
        network::REGIONAL => {
            if dev < 0.12 {
                cls::SECONDARY
            } else {
                cls::TRUNK
            }
        }
        network::LOCAL => {
            if dev < 0.08 {
                cls::LOCAL
            } else {
                cls::SECONDARY
            }
        }
        network::HAMLET => cls::LOCAL,
        network::TRACK => cls::TRACK,
        network::RAIL => cls::RAIL,
        _ => cls::POWER,
    }
}

/// An edge of the network.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Edge {
    /// a.id < b.id for graph edges (sites: a is the site)
    pub a: Node,
    pub b: Node,
    pub class: usize,
    pub net: usize,
    pub key: u64,
}

impl Edge {
    pub fn new(a: Node, b: Node, class: usize, netw: usize) -> Edge {
        let key = mix64(a.id ^ b.id.rotate_left(23) ^ (netw as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (class as u64) << 56);
        Edge { a, b, class, net: netw, key }
    }
    pub fn len(&self) -> f64 {
        (self.b.p - self.a.p).length()
    }
}

type NbKey = (u64, u8, u64);
static NB: RwLock<Option<FxHashMap<NbKey, Arc<Vec<Node>>>>> = RwLock::new(None);

/// The neighbours of `a` in network `n` (relative-neighbourhood graph within `LMAX`).
pub fn neighbours(w: &World, n: usize, a: &Node) -> Arc<Vec<Node>> {
    let key = (w.cache_key, n as u8, a.id);
    if let Some(v) = NB.read().unwrap().as_ref().and_then(|m| m.get(&key)) {
        return v.clone();
    }
    let cands = nodes(w, n, a.p, LMAX[n]);
    let mut out = Vec::new();
    for b in &cands {
        if b.id == a.id || !pair_ok(n, a, b) {
            continue;
        }
        let d = (b.p - a.p).length();
        if d > LMAX[n] {
            continue;
        }
        // blockers lie within d of a: all are candidates
        let blocked = cands.iter().any(|c| c.id != a.id && c.id != b.id && (c.p - a.p).length() < d && (c.p - b.p).length() < d);
        if !blocked {
            out.push(*b);
        }
    }
    let v = Arc::new(out);
    let mut g = NB.write().unwrap();
    let m = g.get_or_insert_with(FxHashMap::default);
    if m.len() > 400_000 {
        m.clear();
    }
    m.insert(key, v.clone());
    v
}

/// Distance from `p` to the chord a–b (m).
pub fn chord_dist(a: DVec3, b: DVec3, p: DVec3) -> f64 {
    let ab = b - a;
    let t = ((p - a).dot(ab) / ab.length_squared().max(1e-9)).clamp(0.0, 1.0);
    (p - (a + ab * t)).length()
}

/// The edges of network `n` whose route can come within `radius` of `center`.
pub fn edges_near(w: &World, n: usize, center: DVec3, radius: f64) -> Vec<Edge> {
    let lmax = LMAX[n];
    let mut out: Vec<Edge> = Vec::new();
    // one end of every such edge lies within radius + lmax / 2 + deviation of the centre
    let dev_max = super::route::max_deviation(super::cls::TRACK, lmax).max(super::route::max_deviation(super::cls::MOTORWAY, lmax));
    for a in nodes(w, n, center, radius + 0.5 * lmax + dev_max) {
        for b in neighbours(w, n, &a).iter() {
            let d = (b.p - a.p).length();
            // each edge once: from the end nearer to the centre (ties: lower id)
            let (da, db) = ((a.p - center).length(), (b.p - center).length());
            if db < da || (db == da && b.id < a.id) {
                continue;
            }
            let class = edge_class(n, &a, b);
            if chord_dist(a.p, b.p, center) > radius + super::route::max_deviation(class, d) + 1_000.0 {
                continue;
            }
            let (x, y) = if a.id < b.id { (a, *b) } else { (*b, a) };
            out.push(Edge::new(x, y, class, n));
        }
    }
    out.sort_unstable_by_key(|e| e.key);
    out.dedup_by_key(|e| e.key);
    out
}

/// Clear the caches (tests).
#[cfg(test)]
pub fn clear_cache() {
    *NB.write().unwrap() = None;
}
