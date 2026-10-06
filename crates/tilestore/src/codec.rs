//! Chunk codec matching the HDF5 filter pipeline [shuffle, deflate] (zlib stream).

use flate2::read::ZlibDecoder;
use flate2::write::ZlibEncoder;
use flate2::Compression;
use std::io::{Read, Write};

pub fn shuffle(data: &[u8], elem: usize) -> Vec<u8> {
    if elem <= 1 {
        return data.to_vec();
    }
    let n = data.len() / elem;
    let mut out = vec![0u8; data.len()];
    for b in 0..elem {
        let dst = &mut out[b * n..(b + 1) * n];
        for (i, d) in dst.iter_mut().enumerate() {
            *d = data[i * elem + b];
        }
    }
    // HDF5 leaves trailing bytes (len % elem) unshuffled at the end
    let tail = n * elem;
    out[tail..].copy_from_slice(&data[tail..]);
    out
}

pub fn unshuffle(data: &[u8], elem: usize) -> Vec<u8> {
    if elem <= 1 {
        return data.to_vec();
    }
    let n = data.len() / elem;
    let mut out = vec![0u8; data.len()];
    for b in 0..elem {
        let src = &data[b * n..(b + 1) * n];
        for (i, s) in src.iter().enumerate() {
            out[i * elem + b] = *s;
        }
    }
    let tail = n * elem;
    out[tail..].copy_from_slice(&data[tail..]);
    out
}

pub fn encode(data: &[u8], elem: usize, level: u32) -> Vec<u8> {
    let sh = shuffle(data, elem);
    let mut e = ZlibEncoder::new(Vec::with_capacity(data.len() / 3), Compression::new(level));
    e.write_all(&sh).expect("zlib write");
    e.finish().expect("zlib finish")
}

pub fn decode(bytes: &[u8], elem: usize, expected: usize) -> std::io::Result<Vec<u8>> {
    let mut d = ZlibDecoder::new(bytes);
    let mut out = Vec::with_capacity(expected);
    d.read_to_end(&mut out)?;
    Ok(unshuffle(&out, elem))
}

#[cfg(test)]
mod tests {
    #[test]
    fn roundtrip() {
        let data: Vec<u8> = (0..1027u32).map(|i| (i * 7 % 251) as u8).collect();
        for elem in [1, 2, 4] {
            let e = super::encode(&data, elem, 4);
            assert_eq!(super::decode(&e, elem, data.len()).unwrap(), data);
        }
    }
}
