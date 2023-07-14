/*
# Copyright (c) 2023 uib GmbH <info@uib.de>
# This code is owned by the uib GmbH, Mainz, Germany (uib.de). All rights reserved.
# License: AGPL-3.0
*/
use chrono::prelude::*;
use log::{debug, info, warn};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::pyclass::CompareOp;
use pyo3::{types::PyBytes, wrap_pyfunction};
use sha1::{Digest, Sha1};
use sha2::Sha256;
use std::collections::{BTreeMap, HashSet};
use std::fs::{remove_file, File, OpenOptions};
use std::io::{prelude::*, BufReader};
use std::path::{Path, PathBuf};
use std::time::Instant;

mod md4;

const RSUM_SIZE: usize = 4;
const CHECKSUM_SIZE: usize = 16;
const ZSYNC_VERSION: &str = "0.6.2";
const PYZSYNC_VERSION: &str = env!("CARGO_PKG_VERSION");
const PRODUCER_NAME: &str = "pyzsync";
const SOURCE_REMOTE: i8 = -1;

#[derive(Debug, Clone)]
#[pyclass]
struct BlockInfo {
    block_id: u64,
    offset: u64,
    size: u16,
    rsum: u32,
    checksum: [u8; CHECKSUM_SIZE],
}

#[pymethods]
impl BlockInfo {
    #[new]
    fn new(
        block_id: u64,
        offset: u64,
        size: u16,
        rsum: u32,
        checksum: [u8; CHECKSUM_SIZE],
    ) -> Self {
        BlockInfo {
            block_id: block_id,
            offset: offset,
            size: size,
            rsum: rsum,
            checksum: checksum,
        }
    }
    #[getter]
    fn block_id(&self) -> PyResult<u64> {
        Ok(self.block_id)
    }
    #[getter]
    fn offset(&self) -> PyResult<u64> {
        Ok(self.offset)
    }
    #[getter]
    fn size(&self) -> PyResult<u16> {
        Ok(self.size)
    }
    #[getter]
    fn rsum(&self) -> PyResult<u32> {
        Ok(self.rsum)
    }
    #[getter]
    fn checksum(&self, py: Python<'_>) -> PyResult<PyObject> {
        Ok(PyBytes::new(py, &self.checksum).into())
    }
}

#[derive(Debug, Clone)]
#[pyclass]
struct ZsyncFileInfo {
    zsync: String,
    producer: String,
    filename: String,
    url: String,
    sha1: [u8; 20],
    sha256: [u8; 32],
    mtime: DateTime<Utc>,
    length: u64,
    block_size: u32,
    seq_matches: u8,
    rsum_bytes: u8,
    checksum_bytes: u8,
    block_info: Vec<BlockInfo>,
}

#[pymethods]
impl ZsyncFileInfo {
    #[new]
    fn new(
        zsync: String,
        producer: String,
        filename: String,
        url: String,
        sha1: [u8; 20],
        sha256: [u8; 32],
        mtime: DateTime<Utc>,
        length: u64,
        block_size: u32,
        seq_matches: u8,
        rsum_bytes: u8,
        checksum_bytes: u8,
        block_info: Vec<BlockInfo>,
    ) -> Self {
        ZsyncFileInfo {
            zsync: zsync,
            producer: producer,
            filename: filename,
            url: url,
            sha1: sha1,
            sha256: sha256,
            mtime: mtime,
            length: length,
            block_size: block_size,
            seq_matches: seq_matches,
            rsum_bytes: rsum_bytes,
            checksum_bytes: checksum_bytes,
            block_info: block_info,
        }
    }
    #[getter]
    fn zsync(&self) -> PyResult<String> {
        Ok(self.zsync.clone())
    }
    #[getter]
    fn producer(&self) -> PyResult<String> {
        Ok(self.producer.clone())
    }
    #[getter]
    fn filename(&self) -> PyResult<String> {
        Ok(self.filename.clone())
    }
    #[getter]
    fn url(&self) -> PyResult<String> {
        Ok(self.url.clone())
    }
    #[getter]
    fn sha1(&self, py: Python<'_>) -> PyResult<PyObject> {
        Ok(PyBytes::new(py, &self.sha1).into())
    }
    #[getter]
    fn sha256(&self, py: Python<'_>) -> PyResult<PyObject> {
        Ok(PyBytes::new(py, &self.sha256).into())
    }
    #[getter]
    fn mtime(&self) -> PyResult<DateTime<Utc>> {
        Ok(self.mtime)
    }
    #[getter]
    fn length(&self) -> PyResult<u64> {
        Ok(self.length)
    }
    #[getter]
    fn block_size(&self) -> PyResult<u32> {
        Ok(self.block_size)
    }
    #[getter]
    fn get_seq_matches(&self) -> PyResult<u8> {
        Ok(self.seq_matches)
    }
    #[setter]
    fn set_seq_matches(&mut self, value: u8) -> PyResult<()> {
        self.seq_matches = value;
        Ok(())
    }
    #[getter]
    fn rsum_bytes(&self) -> PyResult<u8> {
        Ok(self.rsum_bytes)
    }
    #[getter]
    fn checksum_bytes(&self) -> PyResult<u8> {
        Ok(self.checksum_bytes)
    }
    #[getter]
    fn block_info(&self) -> PyResult<Vec<BlockInfo>> {
        Ok(self.block_info.clone())
    }
}

#[derive(Debug, Clone)]
#[pyclass]
struct PatchInstruction {
    source: i8,
    source_offset: u64,
    target_offset: u64,
    size: u64,
}

#[pymethods]
impl PatchInstruction {
    #[new]
    fn new(source: i8, source_offset: u64, target_offset: u64, size: u64) -> Self {
        PatchInstruction {
            source: source,
            source_offset: source_offset,
            target_offset: target_offset,
            size: size,
        }
    }
    #[getter]
    fn source(&self) -> PyResult<i8> {
        Ok(self.source)
    }
    #[getter]
    fn source_offset(&self) -> PyResult<u64> {
        Ok(self.source_offset)
    }
    #[getter]
    fn target_offset(&self) -> PyResult<u64> {
        Ok(self.target_offset)
    }
    #[getter]
    fn size(&self) -> PyResult<u64> {
        Ok(self.size)
    }

    #[setter]
    fn set_source(&mut self, value: i8) -> PyResult<()> {
        self.source = value;
        Ok(())
    }
    #[setter]
    fn set_source_offset(&mut self, value: u64) -> PyResult<()> {
        self.source_offset = value;
        Ok(())
    }
    #[setter]
    fn set_target_offset(&mut self, value: u64) -> PyResult<()> {
        self.target_offset = value;
        Ok(())
    }
    #[setter]
    fn set_size(&mut self, value: u64) -> PyResult<()> {
        self.size = value;
        Ok(())
    }

    fn __repr__(&self) -> String {
        let mut src = format!("file#{}", self.source);
        if self.source == SOURCE_REMOTE {
            src = "remote".to_string();
        }
        format!(
            "PatchInstruction(source={}, source_offset={}, size={}, target_offset={})",
            src, self.source_offset, self.size, self.target_offset
        )
    }

    fn __richcmp__(&self, other: PyRef<PatchInstruction>, op: CompareOp) -> Py<PyAny> {
        let py = other.py();
        match op {
            CompareOp::Eq => (self.source == other.source
                && self.source_offset == other.source_offset
                && self.target_offset == other.target_offset
                && self.size == other.size)
                .into_py(py),
            CompareOp::Ne => (self.source != other.source
                || self.source_offset != other.source_offset
                || self.target_offset != other.target_offset
                || self.size != other.size)
                .into_py(py),
            _ => py.NotImplemented(),
        }
    }

    pub fn copy(&self) -> Self {
        self.clone()
    }

    pub fn __copy__(&self) -> Self {
        self.clone()
    }

    pub fn __deepcopy__(&self) -> Self {
        self.clone()
    }
}

#[pyfunction]
fn rs_version() -> PyResult<String> {
    Ok(PYZSYNC_VERSION.to_string())
}

/// Get the md4 hash of a block
fn _md4_part(
    block: &Vec<u8>,
    num_bytes: u8,
    offset: usize,
    len: usize,
) -> Result<[u8; CHECKSUM_SIZE], PyErr> {
    if num_bytes <= 0 || num_bytes > CHECKSUM_SIZE as u8 {
        return Err(PyValueError::new_err(format!(
            "num_bytes out of range: {}",
            num_bytes
        )));
    }
    let mut checksum: [u8; CHECKSUM_SIZE] = md4::md4(&block[offset..offset + len]);
    if (num_bytes as usize) < CHECKSUM_SIZE {
        for idx in (num_bytes as usize)..CHECKSUM_SIZE {
            checksum[idx] = 0u8;
        }
    }
    Ok(checksum)
}

fn _md4(block: &Vec<u8>, num_bytes: u8) -> Result<[u8; CHECKSUM_SIZE], PyErr> {
    _md4_part(block, num_bytes, 0, block.len())
}

#[pyfunction]
fn rs_md4(block: &PyBytes, num_bytes: u8, py: Python<'_>) -> PyResult<PyObject> {
    let res = _md4(block.as_bytes().to_vec().as_ref(), num_bytes)?;
    Ok(PyBytes::new(py, &res).into())
}

/// Get the rsum of a block
fn _rsum_part(block: &Vec<u8>, num_bytes: u8, offset: usize, len: usize) -> Result<u32, PyErr> {
    if num_bytes <= 0 || num_bytes > RSUM_SIZE as u8 {
        return Err(PyValueError::new_err(format!(
            "num_bytes out of range: {}",
            num_bytes
        )));
    }
    let mut a: u16 = 0;
    let mut b: u16 = 0;
    let mut rlen = len as u16;
    for idx in offset..offset + len {
        let c = u16::from(block[idx]);
        a += c;
        b += rlen * c;
        rlen -= 1;
    }
    let mut res: u32 = ((a as u32) << 16) + b as u32;
    if num_bytes < 4 {
        let mask = 0xffffffff >> (8 * (RSUM_SIZE as u8 - num_bytes));
        res &= mask;
    }
    Ok(res)
}

fn _rsum(block: &Vec<u8>, num_bytes: u8) -> Result<u32, PyErr> {
    _rsum_part(&block, num_bytes, 0, block.len())
}

#[pyfunction]
fn rs_rsum(block: &PyBytes, num_bytes: u8) -> PyResult<u32> {
    let res = _rsum(block.as_bytes().to_vec().as_ref(), num_bytes)?;
    Ok(res)
}

fn _update_rsum(rsum: u32, old_char: u8, new_char: u8, block_size: u16) -> u32 {
    // 1 << block_shift = block_size
    let block_shift = if block_size == 2048 { 11 } else { 12 };
    let old_char_u16 = u16::from(old_char);
    let new_char_u16 = u16::from(new_char);
    let mut a: u16 = ((rsum & 0xffff0000) >> 16) as u16;
    let mut b: u16 = rsum as u16 & 0xffff;
    a += new_char_u16 - old_char_u16;
    b += a - (old_char_u16 << block_shift);
    let res: u32 = ((a as u32) << 16) + b as u32;
    res
}

#[pyfunction]
fn rs_update_rsum(rsum: u32, old_char: u8, new_char: u8, block_size: u16) -> PyResult<u32> {
    let res = _update_rsum(rsum, old_char, new_char, block_size);
    Ok(res)
}

#[pyfunction]
fn rs_calc_block_size(file_size: u64) -> u16 {
    let block_size = if file_size < 100_000_000 { 2048 } else { 4096 };
    block_size
}

fn _calc_block_infos(
    file_path: &Path,
    block_size: u16,
    mut rsum_bytes: u8,
    mut checksum_bytes: u8,
) -> Result<(Vec<BlockInfo>, [u8; 20], [u8; 32]), PyErr> {
    if block_size != 2048 && block_size != 4096 {
        return Err(PyValueError::new_err(format!(
            "Invalid block_size: {}",
            block_size
        )));
    }

    if rsum_bytes < 1 {
        // rsum disabled
        rsum_bytes = 0;
    } else if rsum_bytes > RSUM_SIZE as u8 {
        return Err(PyValueError::new_err(format!(
            "rsum_bytes out of range: {}",
            rsum_bytes
        )));
    }

    if checksum_bytes < 1 {
        // checksum disabled
        checksum_bytes = 0;
    } else if checksum_bytes > CHECKSUM_SIZE as u8 {
        return Err(PyValueError::new_err(format!(
            "checksum_bytes out of range: {}",
            checksum_bytes
        )));
    }

    let mut sha1 = Sha1::new();
    let mut sha256 = Sha256::new();
    let file = File::open(file_path)?;
    let size = file_path.metadata()?.len();
    let block_count: u64 = (size + u64::from(block_size) - 1) / u64::from(block_size);
    let mut block_infos: Vec<BlockInfo> = Vec::new();
    let mut reader = BufReader::new(file);
    for block_id in 0..block_count {
        let offset = block_id * block_size as u64;
        let mut buf_size = block_size as usize;
        if block_id == block_count - 1 {
            buf_size = (size - offset) as usize;
        }
        let mut block = vec![0u8; buf_size];
        reader.read_exact(&mut block)?;
        sha1.update(&block);
        sha256.update(&block);
        if buf_size < block_size as usize {
            block.resize(block_size as usize, 0u8);
        }

        let mut checksum = [0u8; CHECKSUM_SIZE];
        if checksum_bytes > 0 {
            checksum = _md4(block.as_ref(), checksum_bytes)?;
        }

        let mut rsum: u32 = 0;
        if rsum_bytes > 0 {
            rsum = _rsum(block.as_ref(), rsum_bytes)?;
        }

        let block_info = BlockInfo {
            block_id: block_id,
            offset: offset,
            size: buf_size as u16,
            checksum: checksum,
            rsum: rsum,
        };
        block_infos.push(block_info);
    }
    let sha1_digest = sha1.finalize();
    let sha256_digest = sha256.finalize();
    Ok((block_infos, sha1_digest.into(), sha256_digest.into()))
}

#[pyfunction]
fn rs_calc_block_infos(
    file_path: PathBuf,
    block_size: u16,
    rsum_bytes: u8,
    checksum_bytes: u8,
) -> PyResult<Vec<BlockInfo>> {
    let result = _calc_block_infos(file_path.as_path(), block_size, rsum_bytes, checksum_bytes)?;
    Ok(result.0)
}

#[pyfunction]
fn rs_get_patch_instructions(
    py: Python<'_>,
    zsync_file_info: ZsyncFileInfo,
    file_paths: Vec<PathBuf>,
    progress_callback: PyObject,
) -> PyResult<Vec<PatchInstruction>> {
    // Get a list of instructions, based on the zsync file info,
    // to build the remote file, using as much as possible data from the local file.

    // If seq_matches is set to 2, the algorithm searches for two consecutive matching blocks.
    // This greatly speeds up the matching process.
    let seq_matches = zsync_file_info.seq_matches > 1;
    let checksum_bytes = zsync_file_info.checksum_bytes as usize;
    let block_size = zsync_file_info.block_size as usize;

    // There are 2 ** (rsum_bytes * 8) possible rsum hashes.
    let max_rsum_hashes = 2 << (zsync_file_info.rsum_bytes * 8 - 1);
    let rsum_mask: u32 = max_rsum_hashes - 1;

    // The rsum_list is used for fast negative checking.
    // If the value of the vector at index <rsum> is 0,
    // the hash does not exist in the block info.
    let mut rsum_list = vec![0u8; max_rsum_hashes as usize];

    // Arrange all blocks from the zsync file info in a map:
    // <block-rsum> => <block-md4> => <list of matching blocks>
    let mut map = BTreeMap::new();
    let mut sum_block_size: u64 = 0;

    for idx in 0..zsync_file_info.block_info.len() {
        let block_info = &zsync_file_info.block_info[idx];
        let mut hash = block_info.rsum;

        rsum_list[hash as usize] = 1;
        if seq_matches && idx + 1 < zsync_file_info.block_info.len() {
            // Combine rsum and rsum of following block
            let next_block_info = &zsync_file_info.block_info[idx + 1];
            hash ^= next_block_info.rsum << 16;
        }

        sum_block_size += block_info.size as u64;
        let checksum = &block_info.checksum[0..checksum_bytes];
        if !map.contains_key(&hash) {
            let map2 = BTreeMap::new();
            map.insert(hash, map2);
        }
        let map2 = map.get_mut(&hash).unwrap();
        if !map2.contains_key(&checksum) {
            let blocks: Vec<&BlockInfo> = Vec::new();
            map2.insert(checksum, blocks);
        }
        map2.get_mut(&checksum).unwrap().push(block_info);
    }

    if sum_block_size != zsync_file_info.length {
        return Err(PyValueError::new_err(format!(
            "Sum of block sizes {} does not match file info length {}",
            sum_block_size, zsync_file_info.length
        )));
    }

    let mut patch_instructions: Vec<PatchInstruction> = Vec::new();
    let mut block_ids_found: HashSet<u64> = HashSet::new();
    let mut rsum_list_lookups: u64 = 0;
    let mut rsum_map_lookups: u64 = 0;
    let mut checksum_lookups: u64 = 0;
    let mut checksum_matches: u64 = 0;

    for (file_id, file_path) in file_paths.iter().enumerate() {
        info!("Processing file #{}: {:?}", file_id, file_path);
        if !file_path.exists() {
            File::create(file_path)?;
            continue;
        }
        let file = File::open(file_path)?;
        let metadata = file.metadata()?;
        let file_size = metadata.len();
        if file_size == 0 {
            continue;
        }
        let reader = BufReader::new(file);
        let mut read_it = reader.bytes();
        let buffer_size = block_size * 2;
        let mut buf: Vec<u8> = Vec::with_capacity(buffer_size);
        let mut pos = 0;
        let end_pos: u64 =
            (((file_size as f64 / buffer_size as f64).floor() + 1.0) * buffer_size as f64) as u64;
        let mut percent: u8 = 0;
        let mut rsum = 0;
        let mut next_rsum = 0;
        let mut added_char = 0u8;
        let mut removed_char = 0u8;
        let start_time = Instant::now();
        // Continuously fill a buffer with bytes from the file and update the rsum.
        // If current rsum is found in map, also check md4.
        // Add matches to patch_instructions.
        loop {
            let new_percent: u8 = ((pos as f64 / end_pos as f64) * 100.0) as u8;
            if new_percent != percent || pos == 0 || pos >= end_pos {
                percent = new_percent;
                let duration = start_time.elapsed().as_millis();
                info!(
                    "{}/{} | {} % | {} ms | {} > {} > {} > {}",
                    pos,
                    end_pos,
                    percent,
                    duration,
                    rsum_list_lookups,
                    rsum_map_lookups,
                    checksum_lookups,
                    checksum_matches
                );
                if !progress_callback.is_none(py) {
                    let abort: PyObject = progress_callback
                        .call(py, (pos, end_pos), None)?
                        .extract(py)?;
                    if abort.is_true(py)? {
                        return Err(PyRuntimeError::new_err("Aborted by progress callback"));
                    }
                }
            }

            if pos >= end_pos as u64 {
                break;
            }
            // Let the Python interpreter a chance to process signals
            py.check_signals()?;

            // Fill the buffer with chars until the block size is reached.
            let add_chars = buffer_size - buf.len();
            while buf.len() < buffer_size {
                let _byte = read_it.next();
                if _byte.is_none() {
                    // End of file, fill buffer with zeroes
                    buf.push(0u8);
                } else {
                    added_char = _byte.unwrap().unwrap();
                    buf.push(added_char);
                }
                pos += 1;
            }

            if add_chars == 1 {
                // Only one char added, update the rolling checksum
                let char = buf[block_size - 1];
                rsum = _update_rsum(rsum, removed_char, char, block_size as u16);
                if seq_matches {
                    next_rsum = _update_rsum(next_rsum, char, added_char, block_size as u16);
                }
            } else {
                // More than one char added to buffer, calculate new rsum
                if next_rsum == 0 || add_chars != block_size {
                    // Exactly one block size added, reuse next rsum
                    rsum = _rsum_part(&buf, RSUM_SIZE as u8, 0, block_size)?;
                } else {
                    rsum = next_rsum;
                }
                if seq_matches {
                    next_rsum = _rsum_part(&buf, RSUM_SIZE as u8, block_size, block_size)?;
                }
            }

            // Reduce rsum for lookup to match rsum_bytes of block info
            let mut hash = rsum & rsum_mask;
            let mut rsum_list_match = false;

            // First look into the rsum_list (fast negative check)
            rsum_list_lookups += 1;
            if rsum_list[hash as usize] != 0 {
                if seq_matches {
                    let next_hash = next_rsum & rsum_mask;
                    hash ^= next_hash << 16;
                    if rsum_list[next_hash as usize] != 0 {
                        rsum_list_match = true;
                    }
                } else {
                    rsum_list_match = true;
                }
            }
            if rsum_list_match {
                let entry = map.get(&hash);
                rsum_map_lookups += 1;
                if entry.is_some() {
                    // Found some blocks wich match the rsum of the current buffer.
                    // Calculate md4 checksum of current buffer and reduce it to checksum_bytes.
                    let full_checksum = _md4_part(&buf, CHECKSUM_SIZE as u8, 0, block_size)?;
                    let checksum = &full_checksum[0..checksum_bytes];
                    let block_infos = entry.unwrap().get(&checksum);
                    checksum_lookups += 1;
                    if block_infos.is_some() {
                        // Found some blocks which also match the reduced md4 of the current buffer.
                        checksum_matches += 1;
                        //debug!("Matching md4: {:?}", block_infos);
                        for block_info in block_infos.unwrap() {
                            if pos - buffer_size as u64 + block_info.size as u64 <= file_size {
                                // A file can contain several blocks with matching md4 sums.
                                // Check if block ID was already handled in the instructions.
                                if !block_ids_found.contains(&block_info.block_id) {
                                    // Add current file offsets to instructions
                                    patch_instructions.push(PatchInstruction {
                                        source: file_id as i8,
                                        source_offset: pos - buffer_size as u64,
                                        target_offset: block_info.offset,
                                        size: block_info.size as u64,
                                    });

                                    // Add the ID of the block to the list of block IDs found
                                    block_ids_found.insert(block_info.block_id);
                                }
                            }
                        }
                        // Remove on block from buffer to read a full block size on next iteration
                        buf.drain(0..block_size);
                        continue;
                    }
                }
            }

            // No match found, remove first char from buffer
            removed_char = buf.drain(0..1).next().unwrap();
        }
        let duration = start_time.elapsed().as_millis();
        let mut rsum_efficiency = 1.0;
        if rsum_map_lookups > 0 {
            rsum_efficiency = 1.0
                - ((checksum_lookups as f64 - checksum_matches as f64) / rsum_map_lookups as f64);
        }
        debug!(
            "Statistics: duration={} ms, rsum_list_lookups={}, rsum_map_lookups={}, checksum_lookups={}, checksum_matches={}, rsum_efficiency={:.3} %",
            duration, rsum_list_lookups, rsum_map_lookups, checksum_lookups, checksum_matches, rsum_efficiency * 100.0
        );
        if rsum_efficiency < 0.4 {
            warn!(
                "Inefficient rsum (rsum_list_lookups={}, rsum_map_lookups={}, checksum_lookups={}, checksum_matches={}, rsum_efficiency={:.3} %)",
                rsum_list_lookups, rsum_map_lookups, checksum_lookups, checksum_matches, rsum_efficiency * 100.0
            );
        }
    }

    // Missing blocks need to be fetched from remote.
    // Add instructions to fetch all contiguous areas.
    let mut start_offset: i64 = -1;
    let mut end_offset: i64 = -1;
    let mut iter = zsync_file_info.block_info.iter().peekable();
    while let Some(block_info) = iter.next() {
        // Let the Python interpreter a chance to process signals
        py.check_signals()?;

        let is_last = iter.peek().is_none();
        if !block_ids_found.contains(&block_info.block_id) {
            let offset = block_info.offset as i64;
            if start_offset == -1 {
                start_offset = offset;
                end_offset = offset + block_info.size as i64;
                if !is_last {
                    continue;
                }
            } else if end_offset == offset {
                end_offset = offset + block_info.size as i64;
                if !is_last {
                    continue;
                }
            }
        }
        if start_offset == -1 {
            continue;
        }
        patch_instructions.push(PatchInstruction {
            source: SOURCE_REMOTE,
            source_offset: start_offset as u64,
            target_offset: start_offset as u64,
            size: (end_offset - start_offset) as u64,
        });
        start_offset = -1;
        end_offset = -1;
    }

    // Sort all instructions by target offset so they are in the right order to build the file.
    patch_instructions.sort_by_key(|inst| inst.target_offset);

    // Check instructions for obvious errors
    let mut pos: u64 = 0;
    for inst in &patch_instructions {
        if inst.target_offset != pos {
            return Err(PyRuntimeError::new_err(format!(
                "Gap in instructions: {} <> {}",
                pos, inst.target_offset
            )));
        }
        pos += inst.size as u64;
    }
    if pos != zsync_file_info.length {
        return Err(PyRuntimeError::new_err(format!(
            "Sum of instructions sizes {} does not match file info length {}",
            pos, zsync_file_info.length
        )));
    }

    Ok(patch_instructions)
}

fn _create_zsync_info(file_path: PathBuf, legacy_mode: bool) -> Result<ZsyncFileInfo, PyErr> {
    let metadata = file_path.metadata()?;
    let size = metadata.len();
    let mtime: DateTime<Utc> = metadata.modified()?.into();

    let block_size = rs_calc_block_size(size);

    let seq_matches = if size > (block_size as u64) { 2 } else { 1 };

    let mut rsum_bytes = ((((size as f64).ln() + (block_size as f64).ln()) / 2.0_f64.ln() - 8.6)
        / (seq_matches as f64)
        / 8.0_f64)
        .ceil() as u8;
    if rsum_bytes > 4 {
        rsum_bytes = 4;
    }
    if rsum_bytes < 2 {
        rsum_bytes = 2;
    }

    let mut checksum_bytes = ((20.0_f64
        + ((size as f64).ln() + (1.0_f64 + (size as f64) / (block_size as f64)).ln())
            / 2.0_f64.ln())
        / (seq_matches as f64)
        / 8.0_f64)
        .ceil() as u8;
    let checksum_bytes2 = ((7.9_f64
        + (20.0_f64 + (1.0_f64 + (size as f64) / (block_size as f64)).ln() / (2.0_f64).ln()))
        / 8.0_f64) as u8;
    if checksum_bytes < checksum_bytes2 {
        checksum_bytes = checksum_bytes2;
    }

    debug!(
        "block_size: {}, rsum_bytes: {}, checksum_bytes: {}",
        block_size, rsum_bytes, checksum_bytes
    );

    let (block_infos, sha1_digest, sha256_digest) =
        _calc_block_infos(file_path.as_path(), block_size, rsum_bytes, checksum_bytes)?;
    let zsync_file_info = ZsyncFileInfo {
        zsync: ZSYNC_VERSION.to_string(),
        producer: if legacy_mode {
            "".to_string()
        } else {
            format!("{} {}", PRODUCER_NAME, PYZSYNC_VERSION)
        },
        filename: file_path.file_name().unwrap().to_str().unwrap().to_string(),
        url: file_path.file_name().unwrap().to_str().unwrap().to_string(),
        sha1: sha1_digest,
        sha256: if legacy_mode {
            [0u8; 32]
        } else {
            sha256_digest
        },
        mtime: mtime,
        length: size,
        block_size: block_size as u32,
        seq_matches: seq_matches,
        rsum_bytes: rsum_bytes,
        checksum_bytes: checksum_bytes,
        block_info: block_infos,
    };
    Ok(zsync_file_info)
}

#[pyfunction]
fn rs_create_zsync_info(file_path: PathBuf, legacy_mode: bool) -> PyResult<ZsyncFileInfo> {
    let zsync_file_info = _create_zsync_info(file_path, legacy_mode)?;
    Ok(zsync_file_info)
}

fn _write_zsync_file(zsync_file_info: ZsyncFileInfo, zsync_file_path: PathBuf) -> PyResult<()> {
    info!("Writing zsync file {:?}", zsync_file_path);
    if zsync_file_path.is_file() {
        remove_file(&zsync_file_path)?;
    }
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(zsync_file_path)?;
    file.write_all(format!("zsync: {}\n", zsync_file_info.zsync.trim()).as_bytes())?;
    if zsync_file_info.producer.trim() != "" {
        file.write_all(format!("Producer: {}\n", zsync_file_info.producer.trim()).as_bytes())?;
    }
    file.write_all(format!("Filename: {}\n", zsync_file_info.filename.trim()).as_bytes())?;
    file.write_all(format!("MTime: {}\n", zsync_file_info.mtime.to_rfc2822()).as_bytes())?;
    file.write_all(format!("Blocksize: {}\n", zsync_file_info.block_size).as_bytes())?;
    file.write_all(format!("Length: {}\n", zsync_file_info.length).as_bytes())?;
    file.write_all(
        format!(
            "Hash-Lengths: {},{},{}\n",
            zsync_file_info.seq_matches, zsync_file_info.rsum_bytes, zsync_file_info.checksum_bytes
        )
        .as_bytes(),
    )?;
    file.write_all(format!("URL: {}\n", zsync_file_info.filename.trim()).as_bytes())?;
    file.write_all(format!("SHA-1: {}\n", hex::encode(zsync_file_info.sha1)).as_bytes())?;
    if zsync_file_info.sha256 != [0u8; 32] {
        file.write_all(format!("SHA-256: {}\n", hex::encode(zsync_file_info.sha256)).as_bytes())?;
    }
    file.write_all(b"\n")?;
    for block_info in zsync_file_info.block_info {
        // Write trailing rsum_bytes of the rsum
        let buf: [u8; 4] = [
            ((block_info.rsum >> 24) & 0xff) as u8,
            (block_info.rsum >> 16 & 0xff) as u8,
            ((block_info.rsum >> 8) & 0xff) as u8,
            (block_info.rsum & 0xff) as u8,
        ];
        file.write_all(&buf[RSUM_SIZE - (zsync_file_info.rsum_bytes as usize)..RSUM_SIZE])?;

        // Write leading checksum_bytes of the checksum
        file.write_all(&block_info.checksum[0..(zsync_file_info.checksum_bytes as usize)])?;
    }
    Ok(())
}

#[pyfunction]
fn rs_create_zsync_file(
    file_path: PathBuf,
    zsync_file_path: PathBuf,
    legacy_mode: bool,
) -> PyResult<()> {
    let zsync_file_info = _create_zsync_info(file_path, legacy_mode)?;
    _write_zsync_file(zsync_file_info, zsync_file_path)?;
    Ok(())
}

#[pyfunction]
fn rs_write_zsync_file(zsync_file_info: ZsyncFileInfo, zsync_file_path: PathBuf) -> PyResult<()> {
    _write_zsync_file(zsync_file_info, zsync_file_path)?;
    Ok(())
}

fn _read_zsync_file(zsync_file_path: PathBuf) -> Result<ZsyncFileInfo, PyErr> {
    let file = File::open(zsync_file_path)?;
    let mut zsync_file_info = ZsyncFileInfo {
        zsync: "".to_string(),
        producer: "".to_string(),
        filename: "".to_string(),
        url: "".to_string(),
        sha1: [0u8; 20],
        sha256: [0u8; 32],
        mtime: Utc::now(),
        length: 0,
        block_size: 4096,
        seq_matches: 1,
        rsum_bytes: 4,
        checksum_bytes: 16,
        block_info: Vec::new(),
    };
    let mut reader = BufReader::new(file);
    for line_res in reader.by_ref().lines() {
        let line = line_res?;
        if line == "" {
            break;
        }
        let mut splitter = line.splitn(2, ":");
        let opt = splitter.next().unwrap().trim();
        let val = splitter.next();
        if val.is_some() {
            let value = String::from(val.unwrap().trim());
            let option = String::from(opt).to_lowercase();
            debug!("{}={}", option, value);
            if option == "zsync" {
                zsync_file_info.zsync = value;
            } else if option == "producer" {
                zsync_file_info.producer = value;
            } else if option == "filename" {
                zsync_file_info.filename = value;
            } else if option == "url" {
                zsync_file_info.url = value;
            } else if option == "sha-1" {
                zsync_file_info.sha1 = match hex::decode(value) {
                    Ok(val) => val.try_into().map_err(|e| {
                        PyValueError::new_err(format!("Invalid SHA-1 value: {:?}", e))
                    })?,
                    Err(error) => {
                        return Err(PyValueError::new_err(format!(
                            "Invalid SHA-1 value: {}",
                            error
                        )));
                    }
                };
            } else if option == "sha-256" {
                zsync_file_info.sha256 = match hex::decode(value) {
                    Ok(val) => val.try_into().map_err(|e| {
                        PyValueError::new_err(format!("Invalid SHA-256 value: {:?}", e))
                    })?,
                    Err(error) => {
                        return Err(PyValueError::new_err(format!(
                            "Invalid SHA-256 value: {}",
                            error
                        )));
                    }
                };
            } else if option == "mtime" {
                zsync_file_info.mtime = match DateTime::parse_from_rfc2822(value.as_str()) {
                    Ok(val) => val.with_timezone(&Utc),
                    Err(error) => {
                        return Err(PyValueError::new_err(format!(
                            "Invalid MTime value: {}",
                            error
                        )));
                    }
                };
            } else if option == "blocksize" {
                zsync_file_info.block_size = value.parse()?;
            } else if option == "length" {
                zsync_file_info.length = value.parse()?;
            } else if option == "hash-lengths" {
                let hash_splitter = value.split(",");
                let val: Vec<&str> = hash_splitter.collect();
                if val.len() != 3 {
                    return Err(PyValueError::new_err(format!(
                        "Invalid Hash-Lengths value: {}",
                        value
                    )));
                }
                zsync_file_info.seq_matches = val[0].parse()?;
                zsync_file_info.rsum_bytes = val[1].parse()?;
                zsync_file_info.checksum_bytes = val[2].parse()?;
                if zsync_file_info.seq_matches < 1 || zsync_file_info.seq_matches > 2 {
                    return Err(PyValueError::new_err(format!(
                        "seq_matches out of range: {}",
                        zsync_file_info.seq_matches
                    )));
                }
                if zsync_file_info.rsum_bytes < 1 || zsync_file_info.rsum_bytes > RSUM_SIZE as u8 {
                    return Err(PyValueError::new_err(format!(
                        "rsum_bytes out of range: {}",
                        zsync_file_info.rsum_bytes
                    )));
                }
                if zsync_file_info.checksum_bytes < 3
                    || zsync_file_info.checksum_bytes > CHECKSUM_SIZE as u8
                {
                    return Err(PyValueError::new_err(format!(
                        "checksum_bytes out of range: {}",
                        zsync_file_info.checksum_bytes
                    )));
                }
            }
        }
    }

    let block_count: u64 = (zsync_file_info.length + u64::from(zsync_file_info.block_size) - 1)
        / u64::from(zsync_file_info.block_size);
    for block_id in 0..block_count {
        let mut buf = vec![0u8; zsync_file_info.rsum_bytes as usize];
        reader.read_exact(&mut buf)?;
        buf.resize(RSUM_SIZE, 0u8);
        buf.rotate_right(RSUM_SIZE - zsync_file_info.rsum_bytes as usize);
        let rsum: u32 =
            (buf[0] as u32) << 24 | (buf[1] as u32) << 16 | (buf[2] as u32) << 8 | (buf[3] as u32);

        let mut checksum = vec![0u8; zsync_file_info.checksum_bytes as usize];
        reader.read_exact(&mut checksum)?;
        checksum.resize(CHECKSUM_SIZE, 0u8);

        let offset = block_id * (zsync_file_info.block_size as u64);
        let mut b_size: u16 = zsync_file_info.block_size as u16;
        if block_id + 1 == block_count {
            b_size = (zsync_file_info.length - offset) as u16;
        }
        let block_info = BlockInfo {
            block_id: block_id,
            offset: offset,
            size: b_size,
            checksum: checksum.try_into().unwrap(),
            rsum: rsum,
        };
        zsync_file_info.block_info.push(block_info);
    }
    Ok(zsync_file_info)
}

#[pyfunction]
fn rs_read_zsync_file(zsync_file_path: PathBuf) -> PyResult<ZsyncFileInfo> {
    Ok(_read_zsync_file(zsync_file_path)?)
}

/// A Python module implemented in Rust.
#[pymodule]
fn pyzsync(_py: Python, m: &PyModule) -> PyResult<()> {
    pyo3_log::init();

    m.add_class::<BlockInfo>()?;
    m.add_class::<ZsyncFileInfo>()?;
    m.add_class::<PatchInstruction>()?;
    m.add_function(wrap_pyfunction!(rs_version, m)?)?;
    m.add_function(wrap_pyfunction!(rs_md4, m)?)?;
    m.add_function(wrap_pyfunction!(rs_rsum, m)?)?;
    m.add_function(wrap_pyfunction!(rs_update_rsum, m)?)?;
    m.add_function(wrap_pyfunction!(rs_calc_block_size, m)?)?;
    m.add_function(wrap_pyfunction!(rs_calc_block_infos, m)?)?;
    m.add_function(wrap_pyfunction!(rs_get_patch_instructions, m)?)?;
    m.add_function(wrap_pyfunction!(rs_read_zsync_file, m)?)?;
    m.add_function(wrap_pyfunction!(rs_write_zsync_file, m)?)?;
    m.add_function(wrap_pyfunction!(rs_create_zsync_info, m)?)?;
    m.add_function(wrap_pyfunction!(rs_create_zsync_file, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::_rsum;

    #[test]
    fn test_rsum() {
        let data = vec![0u8; 2048];
        let rsum = _rsum(&data, 4).unwrap();
        assert_eq!(rsum, 0);
    }
}
