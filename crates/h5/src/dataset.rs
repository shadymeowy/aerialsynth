use std::ffi::c_void;
use std::marker::PhantomData;
use std::ptr;

use hdf5_sys as sys;
use sys::h5::hsize_t;
use sys::h5i::hid_t;
use sys::h5p::H5P_DEFAULT;
use sys::h5s::{H5S_ALL, H5S_UNLIMITED};

use crate::error::{Error, Result};
use crate::group::Group;
use crate::raw::{check, cstr, describe, describe_child, lock, to_hsize, Handle};
use crate::types::H5Type;

/// An open HDF5 dataset.
///
/// Reads and writes convert between the stored datatype and the Rust element
/// type `T` (HDF5 numeric conversion), so e.g. a `u8` dataset can be read as
/// `f32`. For best performance use the stored type.
#[derive(Clone, Debug)]
pub struct Dataset {
    pub(crate) h: Handle,
}

/// Builder returned by [`Group::new_dataset`].
///
/// Filters are applied in the order shuffle → deflate (the standard HDF5 /
/// h5py order), which matters for [`Dataset::write_chunk_raw`] callers: a
/// pre-filtered chunk must be byte-shuffled first and then zlib-compressed.
#[must_use]
pub struct DatasetBuilder<'a, T: H5Type> {
    parent: &'a Group,
    shape: Option<Vec<usize>>,
    max_shape: Option<Vec<Option<usize>>>,
    chunk: Option<Vec<usize>>,
    deflate: Option<u8>,
    shuffle: bool,
    fill_value: Option<T>,
    _t: PhantomData<T>,
}

impl<'a, T: H5Type> DatasetBuilder<'a, T> {
    pub(crate) fn new(parent: &'a Group) -> Self {
        DatasetBuilder {
            parent,
            shape: None,
            max_shape: None,
            chunk: None,
            deflate: None,
            shuffle: false,
            fill_value: None,
            _t: PhantomData,
        }
    }

    /// Initial shape (required).
    pub fn shape(mut self, shape: &[usize]) -> Self {
        self.shape = Some(shape.to_vec());
        self
    }

    /// Maximum shape; `None` entries are unlimited. Defaults to `shape`.
    /// Anything larger than `shape` requires chunking.
    pub fn max_shape(mut self, max: &[Option<usize>]) -> Self {
        self.max_shape = Some(max.to_vec());
        self
    }

    /// Chunk shape (required for resizable or filtered datasets).
    pub fn chunk(mut self, chunk: &[usize]) -> Self {
        self.chunk = Some(chunk.to_vec());
        self
    }

    /// Enable the deflate (zlib) filter at `level` 0..=9.
    pub fn deflate(mut self, level: u8) -> Self {
        self.deflate = Some(level);
        self
    }

    /// Enable the byte-shuffle filter (placed before deflate).
    pub fn shuffle(mut self, on: bool) -> Self {
        self.shuffle = on;
        self
    }

    /// Fill value for unwritten elements (default: zero).
    pub fn fill_value(mut self, v: T) -> Self {
        self.fill_value = Some(v);
        self
    }

    /// Create the dataset. Intermediate groups in `name` are created as
    /// needed. Fails if the dataset already exists.
    pub fn create(self, name: &str) -> Result<Dataset> {
        let shape = self
            .shape
            .clone()
            .ok_or_else(|| Error::InvalidArgument(format!("dataset {name:?}: shape not set")))?;
        let rank = shape.len();
        let inval = |m: String| Error::InvalidArgument(format!("dataset {name:?}: {m}"));
        let maxdims: Vec<hsize_t> = match &self.max_shape {
            None => to_hsize(&shape),
            Some(m) => {
                if m.len() != rank {
                    return Err(inval(format!("max_shape rank {} != shape rank {rank}", m.len())));
                }
                m.iter().map(|d| d.map_or(H5S_UNLIMITED, |x| x as hsize_t)).collect()
            }
        };
        let needs_chunk = self.deflate.is_some()
            || self.shuffle
            || maxdims.iter().zip(&shape).any(|(&m, &s)| m != s as hsize_t);
        if let Some(c) = &self.chunk {
            if c.len() != rank {
                return Err(inval(format!("chunk rank {} != shape rank {rank}", c.len())));
            }
            if c.contains(&0) {
                return Err(inval("chunk dimensions must be > 0".into()));
            }
        } else if needs_chunk {
            return Err(inval("resizable or filtered datasets need .chunk(..)".into()));
        }
        if let Some(l) = self.deflate {
            if l > 9 {
                return Err(inval(format!("deflate level {l} > 9")));
            }
        }
        let c_name = cstr(name)?;
        let dims = to_hsize(&shape);
        let loc = self.parent.id();
        let ctx = || describe_child(loc, name);

        let _g = lock();
        // SAFETY: all calls under the lock with valid ids; array pointers
        // point to `rank` elements; `fill` is a valid T matching the native
        // type passed alongside it.
        unsafe {
            let space = Handle::check(
                sys::h5s::H5Screate_simple(rank as _, dims.as_ptr(), maxdims.as_ptr()),
                "H5Screate_simple",
                ctx,
            )?;
            let dcpl = Handle::check(
                sys::h5p::H5Pcreate(*sys::h5p::H5P_CLS_DATASET_CREATE),
                "H5Pcreate(dcpl)",
                ctx,
            )?;
            if let Some(c) = &self.chunk {
                let c = to_hsize(c);
                check(sys::h5p::H5Pset_chunk(dcpl.id(), rank as _, c.as_ptr()), "H5Pset_chunk", ctx)?;
            }
            if self.shuffle {
                check(sys::h5p::H5Pset_shuffle(dcpl.id()), "H5Pset_shuffle", ctx)?;
            }
            if let Some(l) = self.deflate {
                check(sys::h5p::H5Pset_deflate(dcpl.id(), l as _), "H5Pset_deflate", ctx)?;
            }
            if let Some(fill) = &self.fill_value {
                check(
                    sys::h5p::H5Pset_fill_value(
                        dcpl.id(),
                        T::native_type(),
                        fill as *const T as *const c_void,
                    ),
                    "H5Pset_fill_value",
                    ctx,
                )?;
            }
            let lcpl = Handle::check(
                sys::h5p::H5Pcreate(*sys::h5p::H5P_CLS_LINK_CREATE),
                "H5Pcreate(lcpl)",
                ctx,
            )?;
            check(
                sys::h5p::H5Pset_create_intermediate_group(lcpl.id(), 1),
                "H5Pset_create_intermediate_group",
                ctx,
            )?;
            let id = sys::h5d::H5Dcreate2(
                loc,
                c_name.as_ptr(),
                T::native_type(),
                space.id(),
                lcpl.id(),
                dcpl.id(),
                H5P_DEFAULT,
            );
            Ok(Dataset { h: Handle::check(id, "H5Dcreate2", ctx)? })
        }
    }
}

fn prod(v: &[usize]) -> usize {
    v.iter().product()
}

impl Dataset {
    /// Raw HDF5 id (for use with [`crate::sys`] under [`crate::lock`]).
    pub fn id(&self) -> hid_t {
        self.h.id()
    }

    /// Absolute path of this dataset inside its file.
    pub fn name(&self) -> String {
        crate::raw::obj_path(self.id())
    }

    fn ctx(&self) -> String {
        describe(self.id())
    }

    fn space(&self) -> Result<Handle> {
        let _g = lock();
        // SAFETY: valid dataset id under the lock.
        Handle::check(unsafe { sys::h5d::H5Dget_space(self.id()) }, "H5Dget_space", || self.ctx())
    }

    fn dims(&self) -> Result<(Vec<hsize_t>, Vec<hsize_t>)> {
        let _g = lock();
        let space = self.space()?;
        // SAFETY: valid dataspace id; buffers sized to the queried rank.
        unsafe {
            let rank = check(
                sys::h5s::H5Sget_simple_extent_ndims(space.id()),
                "H5Sget_simple_extent_ndims",
                || self.ctx(),
            )? as usize;
            let mut dims = vec![0 as hsize_t; rank];
            let mut max = vec![0 as hsize_t; rank];
            check(
                sys::h5s::H5Sget_simple_extent_dims(space.id(), dims.as_mut_ptr(), max.as_mut_ptr()),
                "H5Sget_simple_extent_dims",
                || self.ctx(),
            )?;
            Ok((dims, max))
        }
    }

    /// Current shape.
    pub fn shape(&self) -> Result<Vec<usize>> {
        Ok(self.dims()?.0.into_iter().map(|d| d as usize).collect())
    }

    /// Maximum shape (`None` = unlimited).
    pub fn max_shape(&self) -> Result<Vec<Option<usize>>> {
        Ok(self
            .dims()?
            .1
            .into_iter()
            .map(|d| if d == H5S_UNLIMITED { None } else { Some(d as usize) })
            .collect())
    }

    /// Chunk shape, or `None` for non-chunked layouts.
    pub fn chunk_shape(&self) -> Result<Option<Vec<usize>>> {
        let _g = lock();
        let rank = self.dims()?.0.len();
        // SAFETY: valid ids under the lock; `chunk` has room for `rank` dims.
        unsafe {
            let dcpl = Handle::check(
                sys::h5d::H5Dget_create_plist(self.id()),
                "H5Dget_create_plist",
                || self.ctx(),
            )?;
            let layout = sys::h5p::H5Pget_layout(dcpl.id());
            if layout != sys::h5d::H5D_layout_t::H5D_CHUNKED {
                return Ok(None);
            }
            let mut chunk = vec![0 as hsize_t; rank.max(1)];
            check(
                sys::h5p::H5Pget_chunk(dcpl.id(), rank as _, chunk.as_mut_ptr()),
                "H5Pget_chunk",
                || self.ctx(),
            )?;
            chunk.truncate(rank);
            Ok(Some(chunk.into_iter().map(|d| d as usize).collect()))
        }
    }

    /// Size in bytes of one stored element.
    pub fn dtype_size(&self) -> Result<usize> {
        let _g = lock();
        // SAFETY: valid ids under the lock.
        unsafe {
            let t = Handle::check(sys::h5d::H5Dget_type(self.id()), "H5Dget_type", || self.ctx())?;
            let n = sys::h5t::H5Tget_size(t.id());
            if n == 0 {
                return Err(crate::raw::hdf5_error("H5Tget_size", self.ctx()));
            }
            Ok(n)
        }
    }

    /// Change the current extent (within `max_shape`). Shrinking discards data.
    pub fn resize(&self, new_shape: &[usize]) -> Result<()> {
        let _g = lock();
        let rank = self.dims()?.0.len();
        if new_shape.len() != rank {
            return Err(Error::InvalidArgument(format!(
                "resize {}: rank {} != dataset rank {rank}",
                self.ctx(),
                new_shape.len()
            )));
        }
        let d = to_hsize(new_shape);
        // SAFETY: valid dataset id; `d` has `rank` elements.
        let r = unsafe { sys::h5d::H5Dset_extent(self.id(), d.as_ptr()) };
        check(r, "H5Dset_extent", || self.ctx())?;
        Ok(())
    }

    /// Validate a hyperslab and build (file space with selection, mem space).
    /// Returns `None` for an empty selection.
    fn select(&self, offset: &[usize], count: &[usize], buf_len: usize) -> Result<Option<(Handle, Handle)>> {
        let _g = lock();
        let (dims, _) = self.dims()?;
        let rank = dims.len();
        let bad = |m: String| Error::InvalidArgument(format!("{}: {m}", self.ctx()));
        if offset.len() != rank || count.len() != rank {
            return Err(bad(format!(
                "offset/count rank {}/{} != dataset rank {rank}",
                offset.len(),
                count.len()
            )));
        }
        if buf_len != prod(count) {
            return Err(bad(format!("buffer length {buf_len} != prod(count {count:?})")));
        }
        for i in 0..rank {
            if (offset[i] + count[i]) as hsize_t > dims[i] {
                return Err(bad(format!(
                    "selection offset {offset:?} count {count:?} exceeds shape {dims:?}"
                )));
            }
        }
        if buf_len == 0 {
            return Ok(None);
        }
        let (o, c) = (to_hsize(offset), to_hsize(count));
        // SAFETY: valid ids under the lock; arrays have `rank` elements.
        unsafe {
            let fspace = self.space()?;
            check(
                sys::h5s::H5Sselect_hyperslab(
                    fspace.id(),
                    sys::h5s::H5S_seloper_t::H5S_SELECT_SET,
                    o.as_ptr(),
                    ptr::null(),
                    c.as_ptr(),
                    ptr::null(),
                ),
                "H5Sselect_hyperslab",
                || self.ctx(),
            )?;
            let mspace = Handle::check(
                sys::h5s::H5Screate_simple(rank as _, c.as_ptr(), ptr::null()),
                "H5Screate_simple",
                || self.ctx(),
            )?;
            Ok(Some((fspace, mspace)))
        }
    }

    /// Write a C-order block `data` (`data.len() == prod(count)`) at `offset`.
    pub fn write_slice<T: H5Type>(&self, data: &[T], offset: &[usize], count: &[usize]) -> Result<()> {
        let _g = lock();
        let Some((fs, ms)) = self.select(offset, count, data.len())? else { return Ok(()) };
        // SAFETY: memory space has exactly data.len() elements of T's native type.
        let r = unsafe {
            sys::h5d::H5Dwrite(
                self.id(),
                T::native_type(),
                ms.id(),
                fs.id(),
                H5P_DEFAULT,
                data.as_ptr() as *const c_void,
            )
        };
        check(r, "H5Dwrite(hyperslab)", || self.ctx())?;
        Ok(())
    }

    /// Read a C-order block at `offset` of extent `count` into `out`
    /// (`out.len() == prod(count)`).
    pub fn read_slice_into<T: H5Type>(&self, offset: &[usize], count: &[usize], out: &mut [T]) -> Result<()> {
        let _g = lock();
        let Some((fs, ms)) = self.select(offset, count, out.len())? else { return Ok(()) };
        // SAFETY: memory space has exactly out.len() elements of T's native type.
        let r = unsafe {
            sys::h5d::H5Dread(
                self.id(),
                T::native_type(),
                ms.id(),
                fs.id(),
                H5P_DEFAULT,
                out.as_mut_ptr() as *mut c_void,
            )
        };
        check(r, "H5Dread(hyperslab)", || self.ctx())?;
        Ok(())
    }

    /// Read a C-order block at `offset` of extent `count`.
    pub fn read_slice<T: H5Type>(&self, offset: &[usize], count: &[usize]) -> Result<Vec<T>> {
        let mut v = vec![T::default(); prod(count)];
        self.read_slice_into(offset, count, &mut v)?;
        Ok(v)
    }

    /// Write the whole dataset (`data.len() == prod(shape)`).
    pub fn write_all<T: H5Type>(&self, data: &[T]) -> Result<()> {
        let _g = lock();
        let shape = self.shape()?;
        self.write_slice(data, &vec![0; shape.len()], &shape)
    }

    /// Read the whole dataset in C order.
    pub fn read_all<T: H5Type>(&self) -> Result<Vec<T>> {
        let _g = lock();
        let shape = self.shape()?;
        let mut v = vec![T::default(); prod(&shape)];
        if v.is_empty() {
            return Ok(v);
        }
        // SAFETY: H5S_ALL/H5S_ALL selects the full extent == v.len() elements.
        let r = unsafe {
            sys::h5d::H5Dread(
                self.id(),
                T::native_type(),
                H5S_ALL,
                H5S_ALL,
                H5P_DEFAULT,
                v.as_mut_ptr() as *mut c_void,
            )
        };
        check(r, "H5Dread(all)", || self.ctx())?;
        Ok(v)
    }

    /// Validate a chunk origin: right rank, aligned to the chunk shape and
    /// inside the current extent (HDF5 itself does not check the extent for
    /// direct chunk I/O and would happily record an out-of-bounds chunk).
    fn chunk_offset(&self, chunk_offset: &[usize]) -> Result<Vec<hsize_t>> {
        let _g = lock();
        let shape = self.shape()?;
        let bad = |m: String| Error::InvalidArgument(format!("{}: chunk offset {chunk_offset:?}: {m}", self.ctx()));
        if chunk_offset.len() != shape.len() {
            return Err(bad(format!("rank {} != dataset rank {}", chunk_offset.len(), shape.len())));
        }
        let chunk = self.chunk_shape()?.ok_or_else(|| bad("dataset is not chunked".into()))?;
        for i in 0..shape.len() {
            if !chunk_offset[i].is_multiple_of(chunk[i]) {
                return Err(bad(format!("not aligned to chunk shape {chunk:?}")));
            }
            if chunk_offset[i] >= shape[i] {
                return Err(bad(format!("outside current shape {shape:?}")));
            }
        }
        Ok(to_hsize(chunk_offset))
    }

    /// Write one already-filtered chunk directly, bypassing the filter
    /// pipeline (`H5Dwrite_chunk`). `chunk_offset` is the element coordinate
    /// of the chunk's origin (a multiple of the chunk shape) and must lie
    /// within the current extent. `filter_mask` bit *i* set means filter *i*
    /// of the pipeline was **not** applied (0 = all filters applied).
    ///
    /// With a pipeline `[shuffle, deflate]`, `bytes` must be
    /// `zlib(shuffle(raw_chunk_bytes))`; for `[deflate]` just `zlib(raw)`.
    /// Edge chunks are always full-size (padded) in storage.
    pub fn write_chunk_raw(&self, chunk_offset: &[usize], filter_mask: u32, bytes: &[u8]) -> Result<()> {
        let _g = lock();
        let off = self.chunk_offset(chunk_offset)?;
        // SAFETY: valid dataset id; `off` has rank elements; `bytes` is valid
        // for bytes.len() bytes.
        let r = unsafe {
            sys::h5d::H5Dwrite_chunk(
                self.id(),
                H5P_DEFAULT,
                filter_mask,
                off.as_ptr(),
                bytes.len(),
                bytes.as_ptr() as *const c_void,
            )
        };
        check(r, "H5Dwrite_chunk", || format!("{} chunk {chunk_offset:?}", self.ctx()))?;
        Ok(())
    }

    /// Read one stored chunk exactly as on disk (still filtered), returning
    /// `(filter_mask, bytes)`. Fails if the chunk has not been written.
    pub fn read_chunk_raw(&self, chunk_offset: &[usize]) -> Result<(u32, Vec<u8>)> {
        let _g = lock();
        let off = self.chunk_offset(chunk_offset)?;
        let ctx = || format!("{} chunk {chunk_offset:?}", self.ctx());
        let mut filters: u32 = 0;
        // SAFETY: H5Dread_chunk2 accepts buf == NULL with *buf_size == 0 and
        // then only reports the required size. The second call passes a
        // buffer of exactly that size.
        unsafe {
            let mut size: usize = 0;
            check(
                sys::h5d::H5Dread_chunk2(
                    self.id(),
                    H5P_DEFAULT,
                    off.as_ptr(),
                    &mut filters,
                    ptr::null_mut(),
                    &mut size,
                ),
                "H5Dread_chunk2(size)",
                ctx,
            )?;
            let mut buf = vec![0u8; size];
            let mut size2 = size;
            check(
                sys::h5d::H5Dread_chunk2(
                    self.id(),
                    H5P_DEFAULT,
                    off.as_ptr(),
                    &mut filters,
                    buf.as_mut_ptr() as *mut c_void,
                    &mut size2,
                ),
                "H5Dread_chunk2",
                ctx,
            )?;
            buf.truncate(size2.min(size));
            Ok((filters, buf))
        }
    }
}
