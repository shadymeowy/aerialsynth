use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;

use h5::{Attrs, Error, File};

fn tmp(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("h5-tests");
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

fn zlib(bytes: &[u8], level: u32) -> Vec<u8> {
    let mut e = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::new(level));
    e.write_all(bytes).unwrap();
    e.finish().unwrap()
}

fn unzlib(bytes: &[u8]) -> Vec<u8> {
    use std::io::Read;
    let mut out = Vec::new();
    flate2::read::ZlibDecoder::new(bytes).read_to_end(&mut out).unwrap();
    out
}

/// Standard HDF5 byte shuffle (H5Z_FILTER_SHUFFLE): byte j of element i goes
/// to position j*n + i; trailing bytes (len % elem_size) are copied as is.
fn shuffle(bytes: &[u8], elem: usize) -> Vec<u8> {
    let n = bytes.len() / elem;
    let mut out = vec![0u8; bytes.len()];
    for i in 0..n {
        for j in 0..elem {
            out[j * n + i] = bytes[i * elem + j];
        }
    }
    out[n * elem..].copy_from_slice(&bytes[n * elem..]);
    out
}

fn unshuffle(bytes: &[u8], elem: usize) -> Vec<u8> {
    let n = bytes.len() / elem;
    let mut out = vec![0u8; bytes.len()];
    for i in 0..n {
        for j in 0..elem {
            out[i * elem + j] = bytes[j * n + i];
        }
    }
    out[n * elem..].copy_from_slice(&bytes[n * elem..]);
    out
}

fn as_bytes<T: Copy>(v: &[T]) -> Vec<u8> {
    // Test helper for plain numeric types only.
    let n = std::mem::size_of_val(v);
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, n) }.to_vec()
}

#[test]
fn create_open_reopen() {
    let p = tmp("basic.h5");
    {
        let f = File::create(&p).unwrap();
        f.set_attr("version", 3u32).unwrap();
        let d = f.new_dataset::<i32>().shape(&[4]).create("v").unwrap();
        d.write_all(&[1, 2, 3, 4]).unwrap();
        f.flush().unwrap();
        assert!(f.filename().ends_with("basic.h5"));
        assert_eq!(f.root().unwrap().name(), "/");
    }
    {
        let f = File::open(&p).unwrap();
        assert_eq!(f.attr::<u32>("version").unwrap(), 3);
        assert_eq!(f.dataset("v").unwrap().read_all::<i32>().unwrap(), vec![1, 2, 3, 4]);
        // Read-only: writing must fail with an HDF5 error, not crash.
        let e = f.set_attr("x", 1u8).unwrap_err();
        assert!(matches!(e, Error::Hdf5 { .. }), "{e}");
    }
    {
        let f = File::open_rw(&p).unwrap();
        f.dataset("v").unwrap().write_slice(&[9i32], &[2], &[1]).unwrap();
        f.set_attr("version", 4u32).unwrap();
    }
    let f = File::open(&p).unwrap();
    assert_eq!(f.dataset("v").unwrap().read_all::<i32>().unwrap(), vec![1, 2, 9, 4]);
    assert_eq!(f.attr::<u32>("version").unwrap(), 4);
    // Truncate on create.
    drop(f);
    let f = File::create(&p).unwrap();
    assert!(f.member_names().unwrap().is_empty());
}

#[test]
fn errors_carry_stack_and_context() {
    let p = tmp("does-not-exist.h5");
    let _ = std::fs::remove_file(&p);
    let e = File::open(&p).unwrap_err();
    let msg = e.to_string();
    assert!(msg.contains("H5Fopen"), "{msg}");
    assert!(msg.contains("does-not-exist.h5"), "{msg}");
    assert!(msg.contains("H5F"), "stack missing: {msg}");
    assert!(File::open_rw(&p).is_err());

    let f = File::create(tmp("errors.h5")).unwrap();
    let e = f.dataset("nope/deeper").unwrap_err().to_string();
    assert!(e.contains("H5Dopen2") && e.contains("nope/deeper") && e.contains("errors.h5"), "{e}");
    // the stack is captured before the target is described (which would clear it)
    assert!(!e.contains("empty HDF5 error stack"), "{e}");
    let d = f.new_dataset::<u8>().shape(&[4, 4]).create("d").unwrap();
    assert!(matches!(d.read_slice::<u8>(&[3, 0], &[2, 1]), Err(Error::InvalidArgument(_))));
    assert!(matches!(d.write_slice(&[1u8; 3], &[0, 0], &[2, 2]), Err(Error::InvalidArgument(_))));
    assert!(matches!(d.read_slice::<u8>(&[0], &[1]), Err(Error::InvalidArgument(_))));
    // Not resizable.
    let e = d.resize(&[8, 4]).unwrap_err().to_string();
    assert!(e.contains("H5Dset_extent"), "{e}");
    assert!(!e.contains("empty HDF5 error stack"), "{e}");
    // Requires chunk.
    assert!(f.new_dataset::<u8>().shape(&[0]).max_shape(&[None]).create("x").is_err());
    assert!(f.new_dataset::<u8>().create("noshape").is_err());
}

#[test]
fn groups() {
    let f = File::create(tmp("groups.h5")).unwrap();
    let a = f.create_group("a").unwrap();
    a.create_group("b").unwrap();
    assert!(f.create_group("a").is_err());
    assert!(f.create_group("x/y").is_err(), "no intermediate creation in create_group");
    assert_eq!(f.group("a/b").unwrap().name(), "/a/b");
    assert_eq!(a.group("/a/b").unwrap().name(), "/a/b");

    let z = f.ensure_group("tiles/z/3").unwrap();
    assert_eq!(z.name(), "/tiles/z/3");
    // Idempotent and works with existing prefixes / absolute paths.
    assert_eq!(a.ensure_group("/tiles/z/3/q").unwrap().name(), "/tiles/z/3/q");
    assert_eq!(f.ensure_group("tiles/z").unwrap().name(), "/tiles/z");
    assert_eq!(a.ensure_group("b/c").unwrap().name(), "/a/b/c");

    assert!(f.exists("a"));
    assert!(f.exists("a/b/c"));
    assert!(f.exists("/tiles/z/3/q"));
    assert!(!f.exists("missing"));
    assert!(!f.exists("missing/deeper/still"));
    assert!(!f.exists("a/missing/c"));
    assert!(a.exists("b"));
    assert!(!a.exists("tiles"));

    let mut names = f.member_names().unwrap();
    names.sort();
    assert_eq!(names, vec!["a", "tiles"]);
    f.new_dataset::<f32>().shape(&[2]).create("a/ds").unwrap();
    assert_eq!(a.member_names().unwrap(), vec!["b", "ds"]);
    // new_dataset creates intermediate groups.
    f.new_dataset::<f32>().shape(&[2]).create("deep/er/ds").unwrap();
    assert!(f.exists("deep/er/ds"));

    a.delete("ds").unwrap();
    assert!(!f.exists("a/ds"));
    assert!(f.delete("a/ds").is_err());
    f.delete("tiles").unwrap();
    assert!(!f.exists("tiles/z/3"));
}

#[test]
fn attributes() {
    let p = tmp("attrs.h5");
    {
        let f = File::create(&p).unwrap();
        let g = f.create_group("g").unwrap();
        let d = g.new_dataset::<u16>().shape(&[3]).create("d").unwrap();

        f.set_attr("u8", 200u8).unwrap();
        f.set_attr("i8", -5i8).unwrap();
        f.set_attr("u16", 60000u16).unwrap();
        f.set_attr("i16", -30000i16).unwrap();
        f.set_attr("u32", 4_000_000_000u32).unwrap();
        f.set_attr("i32", -2_000_000_000i32).unwrap();
        f.set_attr("u64", u64::MAX).unwrap();
        f.set_attr("i64", i64::MIN).unwrap();
        f.set_attr("f32", 1.5f32).unwrap();
        f.set_attr("f64", std::f64::consts::PI).unwrap();
        f.set_attr_str("title", "terrain tiles — ünïcødé").unwrap();

        g.set_attr_array("bbox", &[-180.0f64, -85.0, 180.0, 85.0]).unwrap();
        g.set_attr_array::<i32>("empty", &[]).unwrap();
        g.set_attr("n", 1u32).unwrap();
        g.set_attr("n", 2u32).unwrap(); // overwrite same type
        g.set_attr_str("n", "now a string").unwrap(); // overwrite other type
        d.set_attr_str("units", "m").unwrap();
        d.set_attr("scale", 0.25f32).unwrap();
        d.set_attr_str("empty_str", "").unwrap();

        // A fixed-length, space-padded ASCII string attribute written via raw
        // sys calls (like numpy bytes / Fortran style producers).
        let _l = h5::lock();
        unsafe {
            use ::h5::sys::{h5a, h5p, h5s, h5t};
            let t = h5t::H5Tcopy(*h5t::H5T_C_S1);
            h5t::H5Tset_size(t, 8);
            h5t::H5Tset_strpad(t, h5t::H5T_str_t::H5T_STR_SPACEPAD);
            let s = h5s::H5Screate(h5s::H5S_class_t::H5S_SCALAR);
            let a = h5a::H5Acreate2(d.id(), c"fixed".as_ptr(), t, s, h5p::H5P_DEFAULT, h5p::H5P_DEFAULT);
            assert!(a >= 0);
            assert!(h5a::H5Awrite(a, t, b"abc     ".as_ptr() as *const _) >= 0);
            h5a::H5Aclose(a);
            let t2 = h5t::H5Tcopy(*h5t::H5T_C_S1);
            h5t::H5Tset_size(t2, 6);
            let a = h5a::H5Acreate2(d.id(), c"fixed0".as_ptr(), t2, s, h5p::H5P_DEFAULT, h5p::H5P_DEFAULT);
            assert!(h5a::H5Awrite(a, t2, [b'x', b'y', 0u8, 0, 0, 0].as_ptr() as *const _) >= 0);
            h5a::H5Aclose(a);
            h5s::H5Sclose(s);
            h5t::H5Tclose(t);
            h5t::H5Tclose(t2);
        }
    }
    let f = File::open(&p).unwrap();
    assert_eq!(f.attr::<u8>("u8").unwrap(), 200);
    assert_eq!(f.attr::<i8>("i8").unwrap(), -5);
    assert_eq!(f.attr::<u16>("u16").unwrap(), 60000);
    assert_eq!(f.attr::<i16>("i16").unwrap(), -30000);
    assert_eq!(f.attr::<u32>("u32").unwrap(), 4_000_000_000);
    assert_eq!(f.attr::<i32>("i32").unwrap(), -2_000_000_000);
    assert_eq!(f.attr::<u64>("u64").unwrap(), u64::MAX);
    assert_eq!(f.attr::<i64>("i64").unwrap(), i64::MIN);
    assert_eq!(f.attr::<f32>("f32").unwrap(), 1.5);
    assert_eq!(f.attr::<f64>("f64").unwrap(), std::f64::consts::PI);
    // Numeric conversion on read.
    assert_eq!(f.attr::<f64>("u8").unwrap(), 200.0);
    assert_eq!(f.attr_str("title").unwrap(), "terrain tiles — ünïcødé");
    assert!(f.attr_str("u8").is_err());
    assert!(f.attr::<u8>("title").is_err());
    assert!(f.attr::<u8>("missing").is_err());
    assert!(f.has_attr("u8") && !f.has_attr("missing"));
    let names = f.attr_names().unwrap();
    assert_eq!(names.len(), 11);
    assert!(names.contains(&"title".to_string()));

    let g = f.group("g").unwrap();
    assert_eq!(g.attr_array::<f64>("bbox").unwrap(), vec![-180.0, -85.0, 180.0, 85.0]);
    assert!(g.attr_array::<i32>("empty").unwrap().is_empty());
    assert!(g.attr::<f64>("bbox").is_err(), "4 elements is not a scalar");
    assert_eq!(g.attr_str("n").unwrap(), "now a string");
    assert_eq!(g.attr_names().unwrap(), vec!["bbox", "empty", "n"]);

    let d = g.dataset("d").unwrap();
    assert_eq!(d.attr_str("units").unwrap(), "m");
    assert_eq!(d.attr::<f32>("scale").unwrap(), 0.25);
    assert_eq!(d.attr_str("empty_str").unwrap(), "");
    assert_eq!(d.attr_str("fixed").unwrap(), "abc");
    assert_eq!(d.attr_str("fixed0").unwrap(), "xy");
    assert_eq!(d.attr_array::<f32>("scale").unwrap(), vec![0.25]);
    // Open objects keep the file open (weak close degree): drop them all
    // before re-opening read-write.
    drop((f, g, d));

    let f = File::open_rw(&p).unwrap();
    f.delete_attr("u8").unwrap();
    assert!(!f.has_attr("u8"));
}

const T: usize = 256;

fn tile(seed: u8) -> Vec<u8> {
    (0..T * T * 3).map(|i| ((i * 7 + i / 769) as u8).wrapping_add(seed.wrapping_mul(31))).collect()
}

#[test]
fn extensible_tile_store() {
    let p = tmp("tiles.h5");
    {
        let f = File::create(&p).unwrap();
        let z = f.ensure_group("z3").unwrap();
        let rgb = z
            .new_dataset::<u8>()
            .shape(&[0, T, T, 3])
            .max_shape(&[None, Some(T), Some(T), Some(3)])
            .chunk(&[1, T, T, 3])
            .shuffle(true)
            .deflate(6)
            .create("rgb")
            .unwrap();
        assert_eq!(rgb.shape().unwrap(), vec![0, T, T, 3]);
        assert_eq!(rgb.max_shape().unwrap(), vec![None, Some(T), Some(T), Some(3)]);
        assert_eq!(rgb.chunk_shape().unwrap(), Some(vec![1, T, T, 3]));
        assert_eq!(rgb.dtype_size().unwrap(), 1);
        for i in 0..5usize {
            rgb.resize(&[i + 1, T, T, 3]).unwrap();
            rgb.write_slice(&tile(i as u8), &[i, 0, 0, 0], &[1, T, T, 3]).unwrap();
        }
        // Append several at once.
        rgb.resize(&[7, T, T, 3]).unwrap();
        let two: Vec<u8> = [tile(5), tile(6)].concat();
        rgb.write_slice(&two, &[5, 0, 0, 0], &[2, T, T, 3]).unwrap();
        // Grow without writing: unwritten tile reads as the fill value.
        rgb.resize(&[8, T, T, 3]).unwrap();

        let elev = z.new_dataset::<f32>().shape(&[10, 10]).max_shape(&[None, None]).chunk(&[4, 4]).deflate(4).fill_value(-9999.0).create("elev").unwrap();
        // Hyperslab crossing chunk boundaries.
        let blk: Vec<f32> = (0..15).map(|x| x as f32).collect();
        elev.write_slice(&blk, &[3, 2], &[3, 5]).unwrap();
        elev.resize(&[12, 13]).unwrap();
        assert!(elev.write_slice(&[1.0f32], &[12, 0], &[1, 1]).is_err());
        elev.write_slice(&[42.0f32], &[11, 12], &[1, 1]).unwrap();
    }
    let f = File::open(&p).unwrap();
    let rgb = f.dataset("z3/rgb").unwrap();
    assert_eq!(rgb.shape().unwrap(), vec![8, T, T, 3]);
    for i in 0..7usize {
        assert_eq!(rgb.read_slice::<u8>(&[i, 0, 0, 0], &[1, T, T, 3]).unwrap(), tile(i as u8), "tile {i}");
    }
    assert!(rgb.read_slice::<u8>(&[7, 0, 0, 0], &[1, T, T, 3]).unwrap().iter().all(|&b| b == 0));
    // Sub-block of a tile: pixel (row 10, col 20) of tile 3 for 2x2 pixels.
    let sub = rgb.read_slice::<u8>(&[3, 10, 20, 0], &[1, 2, 2, 3]).unwrap();
    let t3 = tile(3);
    let mut want = Vec::new();
    for r in 10..12 {
        for c in 20..22 {
            want.extend_from_slice(&t3[(r * T + c) * 3..(r * T + c) * 3 + 3]);
        }
    }
    assert_eq!(sub, want);
    // read_slice_into + type conversion (u8 -> f32).
    let mut out = vec![0f32; 3];
    rgb.read_slice_into(&[0, 0, 0, 0], &[1, 1, 1, 3], &mut out).unwrap();
    assert_eq!(out, tile(0)[..3].iter().map(|&b| b as f32).collect::<Vec<_>>());

    let elev = f.dataset("z3/elev").unwrap();
    assert_eq!(elev.shape().unwrap(), vec![12, 13]);
    let all = elev.read_all::<f32>().unwrap();
    for r in 0..12 {
        for c in 0..13 {
            let v = all[r * 13 + c];
            let want = if (3..6).contains(&r) && (2..7).contains(&c) {
                ((r - 3) * 5 + (c - 2)) as f32
            } else if (r, c) == (11, 12) {
                42.0
            } else {
                -9999.0
            };
            assert_eq!(v, want, "({r},{c})");
        }
    }
    assert_eq!(elev.read_slice::<f32>(&[4, 3], &[1, 2]).unwrap(), vec![6.0, 7.0]);
    // Empty selection is fine.
    assert!(elev.read_slice::<f32>(&[0, 0], &[0, 5]).unwrap().is_empty());
}

#[test]
fn raw_chunk_io() {
    let p = tmp("rawchunks.h5");
    let n_tiles = 4usize;
    {
        let f = File::create(&p).unwrap();
        // u8 tiles, pipeline [shuffle, deflate] (shuffle is identity for 1-byte types).
        let rgb = f
            .new_dataset::<u8>()
            .shape(&[0, T, T, 3])
            .max_shape(&[None, Some(T), Some(T), Some(3)])
            .chunk(&[1, T, T, 3])
            .shuffle(true)
            .deflate(6)
            .create("rgb")
            .unwrap();
        rgb.resize(&[n_tiles, T, T, 3]).unwrap();
        for i in 0..n_tiles {
            let c = zlib(&shuffle(&tile(i as u8), 1), 6);
            rgb.write_chunk_raw(&[i, 0, 0, 0], 0, &c).unwrap();
        }
        // Misaligned / out-of-extent chunk writes fail cleanly.
        assert!(rgb.write_chunk_raw(&[n_tiles, 0, 0, 0], 0, &[1, 2, 3]).is_err());
        assert!(rgb.write_chunk_raw(&[0, 0], 0, &[1, 2, 3]).is_err());
        assert!(rgb.write_chunk_raw(&[0, 1, 0, 0], 0, &[1, 2, 3]).is_err());

        // f32 elevation, deflate only.
        let e1 =
            f.new_dataset::<f32>().shape(&[0, 64, 64]).max_shape(&[None, Some(64), Some(64)]).chunk(&[1, 64, 64]).deflate(5).create("elev_deflate").unwrap();
        // f32 elevation, shuffle + deflate (shuffle matters for 4-byte types).
        let e2 = f
            .new_dataset::<f32>()
            .shape(&[0, 64, 64])
            .max_shape(&[None, Some(64), Some(64)])
            .chunk(&[1, 64, 64])
            .shuffle(true)
            .deflate(5)
            .create("elev_shuffle")
            .unwrap();
        // u16, shuffle + deflate, 2D chunks 32x32 inside a 64x96 dataset.
        let u = f.new_dataset::<u16>().shape(&[64, 96]).chunk(&[32, 32]).shuffle(true).deflate(9).create("u16").unwrap();
        e1.resize(&[2, 64, 64]).unwrap();
        e2.resize(&[2, 64, 64]).unwrap();
        for k in 0..2usize {
            let v: Vec<f32> = (0..64 * 64).map(|i| (i as f32).sin() * 1000.0 + k as f32).collect();
            let raw = as_bytes(&v);
            e1.write_chunk_raw(&[k, 0, 0], 0, &zlib(&raw, 5)).unwrap();
            e2.write_chunk_raw(&[k, 0, 0], 0, &zlib(&shuffle(&raw, 4), 5)).unwrap();
        }
        for cr in 0..2usize {
            for cc in 0..3usize {
                let v: Vec<u16> = (0..32 * 32).map(|i| (i * 13 + cr * 1000 + cc * 7) as u16).collect();
                u.write_chunk_raw(&[cr * 32, cc * 32], 0, &zlib(&shuffle(&as_bytes(&v), 2), 9)).unwrap();
            }
        }
        // A chunk written through the normal pipeline, for read_chunk_raw.
        let n = f.new_dataset::<i16>().shape(&[8, 8]).chunk(&[4, 8]).shuffle(true).deflate(3).create("normal").unwrap();
        let v: Vec<i16> = (0..64).map(|i| i * 100 - 3000).collect();
        n.write_all(&v).unwrap();
    }

    let f = File::open(&p).unwrap();
    let rgb = f.dataset("rgb").unwrap();
    for i in 0..n_tiles {
        assert_eq!(rgb.read_slice::<u8>(&[i, 0, 0, 0], &[1, T, T, 3]).unwrap(), tile(i as u8));
        let (mask, bytes) = rgb.read_chunk_raw(&[i, 0, 0, 0]).unwrap();
        assert_eq!(mask, 0);
        assert_eq!(bytes, zlib(&tile(i as u8), 6), "raw chunk bytes roundtrip");
        assert_eq!(unzlib(&bytes), tile(i as u8));
    }
    for name in ["elev_deflate", "elev_shuffle"] {
        let d = f.dataset(name).unwrap();
        for k in 0..2usize {
            let want: Vec<f32> = (0..64 * 64).map(|i| (i as f32).sin() * 1000.0 + k as f32).collect();
            assert_eq!(d.read_slice::<f32>(&[k, 0, 0], &[1, 64, 64]).unwrap(), want, "{name} {k}");
        }
    }
    let u = f.dataset("u16").unwrap();
    let all = u.read_all::<u16>().unwrap();
    for r in 0..64usize {
        for c in 0..96usize {
            let (cr, cc, i) = (r / 32, c / 32, (r % 32) * 32 + c % 32);
            assert_eq!(all[r * 96 + c], (i * 13 + cr * 1000 + cc * 7) as u16);
        }
    }
    // read_chunk_raw of a pipeline-written chunk: undo deflate + shuffle by hand.
    let n = f.dataset("normal").unwrap();
    let (mask, bytes) = n.read_chunk_raw(&[4, 0]).unwrap();
    assert_eq!(mask, 0);
    let raw = unshuffle(&unzlib(&bytes), 2);
    let want: Vec<i16> = (32..64).map(|i| i * 100 - 3000).collect();
    assert_eq!(raw, as_bytes(&want));
    // Unwritten chunk -> error.
    drop((f, rgb, u, n));
    let f = File::open_rw(&p).unwrap();
    let rgb = f.dataset("rgb").unwrap();
    rgb.resize(&[n_tiles + 1, T, T, 3]).unwrap();
    assert!(rgb.read_chunk_raw(&[n_tiles, 0, 0, 0]).is_err());
}

#[test]
fn concurrent_reads() {
    let p = tmp("concurrent.h5");
    let n = 32usize;
    {
        let f = File::create(&p).unwrap();
        let d = f.new_dataset::<u8>().shape(&[n, T, T, 3]).chunk(&[1, T, T, 3]).deflate(1).create("rgb").unwrap();
        for i in 0..n {
            d.write_slice(&tile(i as u8), &[i, 0, 0, 0], &[1, T, T, 3]).unwrap();
        }
    }
    let f = Arc::new(File::open(&p).unwrap());
    let d = Arc::new(f.dataset("rgb").unwrap());
    let threads: Vec<_> = (0..8usize)
        .map(|t| {
            let d = d.clone();
            let f = f.clone();
            std::thread::spawn(move || {
                for round in 0..4usize {
                    for i in (t..n).step_by(8) {
                        let i = (i + round) % n;
                        if round % 2 == 0 {
                            let v = d.read_slice::<u8>(&[i, 0, 0, 0], &[1, T, T, 3]).unwrap();
                            assert_eq!(v, tile(i as u8));
                        } else {
                            // Raw read under the lock, decompress outside it.
                            let (_, bytes) = d.read_chunk_raw(&[i, 0, 0, 0]).unwrap();
                            assert_eq!(unzlib(&bytes), tile(i as u8));
                        }
                        // Mix in metadata ops on other handles.
                        assert!(f.exists("rgb"));
                        let _ = f.dataset("rgb").unwrap().shape().unwrap();
                    }
                }
            })
        })
        .collect();
    for t in threads {
        t.join().unwrap();
    }
}

/// Writes a file for the manual h5py cross-check (see crate docs / report).
/// Path: $CARGO_TARGET_TMPDIR/h5-tests/pycheck.h5
#[test]
fn write_pycheck_file() {
    let f = File::create(tmp("pycheck.h5")).unwrap();
    f.set_attr_str("title", "tile store ✓").unwrap();
    f.set_attr("zoom", 3u32).unwrap();
    f.set_attr_array("bbox", &[-180.0f64, -85.0, 180.0, 85.0]).unwrap();
    let z = f.ensure_group("levels/3").unwrap();
    z.set_attr_str("crs", "EPSG:3857").unwrap();
    let a = z
        .new_dataset::<u8>()
        .shape(&[0, T, T, 3])
        .max_shape(&[None, Some(T), Some(T), Some(3)])
        .chunk(&[1, T, T, 3])
        .shuffle(true)
        .deflate(6)
        .create("rgb_pipeline")
        .unwrap();
    let b = z.new_dataset::<u8>().shape(&[0, T, T, 3]).max_shape(&[None, Some(T), Some(T), Some(3)]).chunk(&[1, T, T, 3]).deflate(6).create("rgb_raw").unwrap();
    for i in 0..3usize {
        a.resize(&[i + 1, T, T, 3]).unwrap();
        a.write_slice(&tile(i as u8), &[i, 0, 0, 0], &[1, T, T, 3]).unwrap();
        b.resize(&[i + 1, T, T, 3]).unwrap();
        b.write_chunk_raw(&[i, 0, 0, 0], 0, &zlib(&tile(i as u8), 6)).unwrap();
    }
    let e = z
        .new_dataset::<f32>()
        .shape(&[2, 64, 64])
        .max_shape(&[None, Some(64), Some(64)])
        .chunk(&[1, 64, 64])
        .shuffle(true)
        .deflate(5)
        .create("elev_raw_shuffle")
        .unwrap();
    for k in 0..2usize {
        let v: Vec<f32> = (0..64 * 64).map(|i| i as f32 * 0.5 + k as f32 * 10000.0).collect();
        e.write_chunk_raw(&[k, 0, 0], 0, &zlib(&shuffle(&as_bytes(&v), 4), 5)).unwrap();
    }
    e.set_attr_str("units", "m").unwrap();
}

/// Manual cross-check: reads a file written by h5py (path in $H5_PY_FILE),
/// containing attrs s="hello" (vlen str), fixed=b"abc" (fixed-length),
/// arr=int16[0..5] and rgb = uint8[2,256,256,3] chunked (1,256,256,3) with
/// gzip+shuffle holding tile(0), tile(1).
#[test]
#[ignore]
fn read_python_file() {
    let Ok(p) = std::env::var("H5_PY_FILE") else { panic!("set H5_PY_FILE") };
    let f = File::open(p).unwrap();
    assert_eq!(f.attr_str("s").unwrap(), "hello");
    assert_eq!(f.attr_str("fixed").unwrap(), "abc");
    assert_eq!(f.attr_array::<i16>("arr").unwrap(), vec![0, 1, 2, 3, 4]);
    let d = f.dataset("rgb").unwrap();
    assert_eq!(d.shape().unwrap(), vec![2, T, T, 3]);
    assert_eq!(d.max_shape().unwrap(), vec![None, Some(T), Some(T), Some(3)]);
    assert_eq!(d.chunk_shape().unwrap(), Some(vec![1, T, T, 3]));
    for i in 0..2usize {
        assert_eq!(d.read_slice::<u8>(&[i, 0, 0, 0], &[1, T, T, 3]).unwrap(), tile(i as u8));
        let (mask, bytes) = d.read_chunk_raw(&[i, 0, 0, 0]).unwrap();
        assert_eq!(mask, 0);
        assert_eq!(unshuffle(&unzlib(&bytes), 1), tile(i as u8));
    }
}
