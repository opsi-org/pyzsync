use std::fs::{OpenOptions, File, remove_file};
use std::io::{prelude::*, BufReader};
use std::path::{Path, PathBuf};
use std::collections::{HashSet, BTreeMap};
use chrono::prelude::*;
use sha1::{Sha1, Digest};
use pyo3::prelude::*;
use pyo3::{
	types::{PyBytes},
	wrap_pyfunction,
};
use log::{info, debug};

mod md4;
const RSUM_SIZE: usize = 4;
const CHECKSUM_SIZE: usize = 16;
const ZSYNC_VERSION: &str= "0.6.2";

#[derive(Debug,Clone)]
#[pyclass]
struct BlockInfo {
	block_id: u64,
	offset: u64,
	size: u16,
	rsum: u32,
	checksum: [u8; CHECKSUM_SIZE]
}

#[pymethods]
impl BlockInfo {
	#[new]
	fn new(
		block_id: u64,
		offset: u64,
		size: u16,
		rsum: u32,
		checksum: [u8; CHECKSUM_SIZE]
	) -> Self {
		BlockInfo {
			block_id: block_id,
			offset: offset,
			size: size,
			rsum: rsum,
			checksum: checksum
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

#[derive(Debug,Clone)]
#[pyclass]
struct ZsyncFileInfo {
	zsync: String,
	filename: String,
	url: String,
	sha1: [u8; 20],
	mtime: DateTime<Utc>,
	length: u64,
	block_size: u32,
	seq_matches: u8,
	rsum_bytes: u8,
	checksum_bytes: u8,
	block_info: Vec<BlockInfo>
}


#[pymethods]
impl ZsyncFileInfo {
	#[new]
	fn new(
		zsync: String,
		filename: String,
		url: String,
		sha1: [u8; 20],
		mtime: DateTime<Utc>,
		length: u64,
		block_size: u32,
		seq_matches: u8,
		rsum_bytes: u8,
		checksum_bytes: u8,
		block_info: Vec<BlockInfo>
	) -> Self {
		ZsyncFileInfo {
			zsync: zsync,
			filename: filename,
			url: url,
			sha1: sha1,
			mtime: mtime,
			length: length,
			block_size: block_size,
			seq_matches: seq_matches,
			rsum_bytes: rsum_bytes,
			checksum_bytes: checksum_bytes,
			block_info: block_info
		}
	}
	#[getter]
	fn zsync(&self) -> PyResult<String> {
		Ok(self.zsync.clone())
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
	fn seq_matches(&self) -> PyResult<u8> {
		Ok(self.seq_matches)
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


#[derive(Debug,Clone)]
#[pyclass]
enum Source {
	Local = 1,
	Remote = 2
}

#[derive(Debug,Clone)]
#[pyclass]
struct PatchInstruction {
	source: Source,
	source_offset: u64,
	target_offset: u64,
	size: u64,
}

#[pymethods]
impl PatchInstruction {
	#[getter]
	fn source(&self) -> PyResult<Source> {
		Ok(self.source.clone())
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
}

/// Get the md4 hash of a block
fn _md4(block: &Vec<u8>, num_bytes: u8) -> [u8; CHECKSUM_SIZE] {
	if num_bytes <= 0 || num_bytes > CHECKSUM_SIZE as u8 {
		panic!("num_bytes out of range: {}", num_bytes);
	}
	let mut checksum: [u8; CHECKSUM_SIZE] = md4::md4(block);
	if (num_bytes as usize) < CHECKSUM_SIZE {
		for idx in (num_bytes as usize) .. CHECKSUM_SIZE {
			checksum[idx] = 0u8;
		}
	}
	checksum
}

#[pyfunction]
fn rs_md4(block: &PyBytes, num_bytes: u8, py: Python<'_>) -> PyResult<PyObject> {
	let res = _md4(block.as_bytes().to_vec().as_ref(), num_bytes);
	Ok(PyBytes::new(py, &res).into())
}

/// Get the weak hash of a block
fn _rsum(block: &Vec<u8>, num_bytes: u8) -> u32 {
	if num_bytes <= 0 || num_bytes > RSUM_SIZE as u8 {
		panic!("num_bytes out of range: {}", num_bytes);
	}
	let mut a: u16 = 0;
	let mut b: u16 = 0;
	let len = block.len();
	let mut rlen = len as u16;
	for idx in 0..len {
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
	res
}

#[pyfunction]
fn rs_rsum(block: &PyBytes, num_bytes: u8) -> PyResult<u32> {
	let res = _rsum(block.as_bytes().to_vec().as_ref(), num_bytes);
	Ok(res)
}

fn _update_rsum(rsum: u32, old_char: u8, new_char: u8) -> u32 {
	let old_char_u16 = u16::from(old_char);
	let new_char_u16 = u16::from(new_char);
	let mut a: u16 = ((rsum & 0xffff0000) >> 16) as u16;
	let mut b: u16 = rsum as u16 & 0xffff;
	a += new_char_u16 - old_char_u16;
	b += a - (old_char_u16 << 11);
	let res: u32 = ((a as u32) << 16) + b as u32;
	res
}

#[pyfunction]
fn rs_update_rsum(rsum: u32, old_char: u8, new_char: u8) -> PyResult<u32> {
	let res = _update_rsum(rsum, old_char, new_char);
	Ok(res)
}

#[pyfunction]
fn rs_calc_block_size(file_size: u64) -> u16 {
	let block_size = if file_size < 100_000_000 { 2048 } else { 4096 };
	block_size
}

fn _calc_block_infos(file_path: &Path, block_size: u16, mut rsum_bytes: u8, mut checksum_bytes: u8) -> Result<(Vec<BlockInfo>, [u8; 20]), std::io::Error> {
	if rsum_bytes < 1 {
		// rsum disabled
		rsum_bytes = 0;
	}
	else if rsum_bytes > RSUM_SIZE as u8 {
		panic!("rsum_bytes out of range: {}", rsum_bytes);
	}

	if checksum_bytes < 1 {
		// checksum disabled
		checksum_bytes = 0;
	}
	else if checksum_bytes > CHECKSUM_SIZE as u8 {
		panic!("checksum_bytes out of range: {}", checksum_bytes);
	}

	let mut sha1 = Sha1::new();
	let file = File::open(file_path)?;
	let size = file_path.metadata()?.len();
	let block_count: u64 = (size + u64::from(block_size) - 1) / u64::from(block_size);
	let mut block_infos :Vec<BlockInfo> = Vec::new();
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
		if buf_size < block_size as usize {
			block.resize(block_size as usize, 0u8);
		}

		let mut checksum = [0u8; CHECKSUM_SIZE];
		if checksum_bytes > 0 {
			checksum = _md4(block.as_ref(), checksum_bytes);
		}

		let mut rsum: u32 = 0;
		if rsum_bytes > 0 {
			rsum = _rsum(block.as_ref(), rsum_bytes);
		}

		let block_info = BlockInfo {
			block_id: block_id,
			offset: offset,
			size: buf_size as u16,
			checksum: checksum,
			rsum: rsum
		};
		block_infos.push(block_info);
	}
	let digest = sha1.finalize();
	Ok((block_infos, digest.into()))
}

#[pyfunction]
fn rs_calc_block_infos(file_path: PathBuf, block_size: u16, rsum_bytes: u8, checksum_bytes: u8) -> PyResult<Vec<BlockInfo>> {
	let result = _calc_block_infos(file_path.as_path(), block_size, rsum_bytes, checksum_bytes)?;
	Ok(result.0)
}

#[pyfunction]
fn rs_get_patch_instructions(zsync_file_info: ZsyncFileInfo, file_path: PathBuf) -> PyResult<Vec<PatchInstruction>> {
	let checksum_bytes = zsync_file_info.checksum_bytes as usize;
	let mut map = BTreeMap::new();
	let mut iter = zsync_file_info.block_info.iter();
	let mut sum_block_size: u64 = 0;
	while let Some(block_info) = iter.next() {
		sum_block_size += block_info.size as u64;
		let checksum = &block_info.checksum[0..checksum_bytes];
		if ! map.contains_key(&block_info.rsum) {
			let map2 = BTreeMap::new();
			map.insert(block_info.rsum, map2);
		}
		let map2 = map.get_mut(&block_info.rsum).unwrap();
		if ! map2.contains_key(&checksum) {
			let blocks :Vec<&BlockInfo> = Vec::new();
			map2.insert(checksum, blocks);
		}
		map2.get_mut(&checksum).unwrap().push(block_info);
	}

	let file = File::open(file_path)?;
	let metadata = file.metadata()?;
	let size = metadata.len();

	if sum_block_size != zsync_file_info.length {
		panic!("Sum of block sizes {} does not match file info length {}", sum_block_size, zsync_file_info.length);
	}

	let mut patch_instructions :Vec<PatchInstruction> = Vec::new();
	let reader = BufReader::new(file);
	let mut read_it = reader.bytes();
	let mut buf: Vec<u8> = Vec::with_capacity(zsync_file_info.block_size as usize);
	let mut pos = 0;
	let mut block_ids_found: HashSet<u64> = HashSet::new();
	let mut percent: u8 = 0;
	let mut rsum = 0;
	let mut old_char = 0u8;
	let rsum_mask = 0xffffffff >> (8 * (RSUM_SIZE as u8 - zsync_file_info.rsum_bytes));

	let mut rsum_lookups: u64 = 0;
	let mut checksum_lookups: u64 = 0;
	let mut checksum_matches: u64 = 0;

	while pos < size {
		let new_percent: u8 = ((pos as f64 / size as f64) * 100.0).ceil() as u8;
		if new_percent != percent {
			percent = new_percent;
			info!("pos: {}/{} ({} %)", pos, zsync_file_info.length, percent);
		}
		let add_bytes = zsync_file_info.block_size - buf.len() as u32;
		while buf.len() < zsync_file_info.block_size as usize {
			let _byte = read_it.next();
			if _byte.is_none() {
				buf.push(0u8);
				continue;
			}
			let new_char = _byte.unwrap().unwrap();
			buf.push(new_char);
			pos += 1;
			if add_bytes == 1 {
				rsum = _update_rsum(rsum, old_char, new_char);
			}
		}
		if add_bytes > 1 {
			// Calc new rsum
			rsum = _rsum(&buf, RSUM_SIZE as u8);
		}

		let rsum_key = rsum & rsum_mask;
		let entry = map.get(&rsum_key);
		rsum_lookups += 1;
		if entry.is_some() {
			//debug!("Matching rsum");
			let full_checksum = _md4(&buf, CHECKSUM_SIZE as u8);
			let checksum = &full_checksum[0..checksum_bytes];
			let block_infos = entry.unwrap().get(&checksum);
			checksum_lookups += 1;
			if block_infos.is_some() {
				checksum_matches += 1;
				//debug!("Matching md4: {:?}", block_infos);
				for block_info in block_infos.unwrap() {
					if ! block_ids_found.contains(&block_info.block_id) {
						patch_instructions.push(
							PatchInstruction {
								source: Source::Local,
								source_offset: pos - block_info.size as u64,
								target_offset: block_info.offset,
								size: block_info.size as u64,
							}
						);
						block_ids_found.insert(block_info.block_id);
					}
				}
				buf.clear();
				continue;
			}
		}
		old_char = buf.drain(0..1).next().unwrap();
	}

	debug!(
		"Statistics: rsum_lookups={}, checksum_lookups={}, checksum_matches={}",
		rsum_lookups, checksum_lookups, checksum_matches
	);

	let mut start_offset: i64 = -1;
	let mut end_offset: i64 = -1;
	let mut iter = zsync_file_info.block_info.iter().peekable();
	while let Some(block_info) = iter.next() {
		let is_last = iter.peek().is_none();
		if ! block_ids_found.contains(&block_info.block_id) {
			let offset = block_info.offset as i64;
			if start_offset == -1 {
				start_offset = offset;
				end_offset = offset + block_info.size as i64;
				if !is_last {
					continue;
				}
			}
			else if end_offset == offset {
				end_offset = offset + block_info.size as i64;
				if !is_last {
					continue;
				}
			}
		}
		if start_offset == -1 {
			continue;
		}
		patch_instructions.push(
			PatchInstruction {
				source: Source::Remote,
				source_offset: start_offset as u64,
				target_offset: start_offset as u64,
				size: (end_offset - start_offset) as u64,
			}
		);
		start_offset = -1;
		end_offset = -1;
	}
	patch_instructions.sort_by_key(|inst| inst.target_offset);

	let mut pos: u64 = 0;
	for inst in &patch_instructions {
		if inst.target_offset != pos {
			panic!("Gap in instructions: {} <> {}", pos, inst.target_offset);
		}
		pos += inst.size as u64;
	}
	if pos != zsync_file_info.length {
		panic!("Sum of instructions sizes {} does not match file info length {}", pos, zsync_file_info.length);
	}
	Ok(patch_instructions)

}


fn _create_zsync_info(file_path: PathBuf) -> Result<ZsyncFileInfo, Box<dyn std::error::Error>> {
	let metadata = file_path.metadata()?;
	let size = metadata.len();
	let mtime: DateTime<Utc> = metadata.modified()?.into();

	let block_size = rs_calc_block_size(size);

	let seq_matches = if size > (block_size as u64) { 2 } else { 1 };

	let mut rsum_bytes = ((((size as f64).ln() + (block_size as f64).ln()) / 2.0_f64.ln() - 8.6) / (seq_matches as f64) / 8.0_f64).ceil() as u8;
	if rsum_bytes > 4 { rsum_bytes = 4; }
	if rsum_bytes < 2 { rsum_bytes = 2; }

	let mut checksum_bytes = (
		(20.0_f64 + ((size as f64).ln() + (1.0_f64 + (size as f64) / (block_size as f64)).ln()) / 2.0_f64.ln()) / (seq_matches as f64) / 8.0_f64
	).ceil() as u8;
	let checksum_bytes2 = ((7.9_f64 + (20.0_f64 + (1.0_f64 + (size as f64) / (block_size as f64)).ln() / (2.0_f64).ln())) / 8.0_f64) as u8;
	if checksum_bytes < checksum_bytes2 {
		checksum_bytes = checksum_bytes2;
	}

	debug!("block_size: {}, rsum_bytes: {}, checksum_bytes: {}", block_size, rsum_bytes, checksum_bytes);

	let (block_infos, sha1_digest) = _calc_block_infos(file_path.as_path(), block_size, rsum_bytes, checksum_bytes)?;
	let zsync_file_info = ZsyncFileInfo {
		zsync: ZSYNC_VERSION.to_string(),
		filename: file_path.file_name().unwrap().to_str().unwrap().to_string(),
		url: file_path.file_name().unwrap().to_str().unwrap().to_string(),
		sha1: sha1_digest,
		mtime: mtime,
		length: size,
		block_size: block_size as u32,
		seq_matches: seq_matches,
		rsum_bytes: rsum_bytes,
		checksum_bytes: checksum_bytes,
		block_info: block_infos
	};
	Ok(zsync_file_info)
}

#[pyfunction]
fn rs_create_zsync_info(file_path: PathBuf) -> PyResult<ZsyncFileInfo> {
	let zsync_file_info = _create_zsync_info(file_path).unwrap();
	Ok(zsync_file_info)
}

#[pyfunction]
fn rs_create_zsync_file(file_path: PathBuf, zsync_file_path: PathBuf) -> PyResult<()> {
	let zsync_file_info = _create_zsync_info(file_path).unwrap();
	_write_zsync_file(zsync_file_info, zsync_file_path)?;
	Ok(())
}

#[pyfunction]
fn rs_write_zsync_file(zsync_file_info: ZsyncFileInfo, zsync_file_path: PathBuf) -> PyResult<()> {
	_write_zsync_file(zsync_file_info, zsync_file_path)?;
	Ok(())
}

fn _write_zsync_file(zsync_file_info: ZsyncFileInfo, zsync_file_path: PathBuf) -> PyResult<()> {
	if zsync_file_path.is_file() {
		remove_file(&zsync_file_path).unwrap();
	}
	let mut file = OpenOptions::new()
		.create_new(true)
		.write(true)
		.open(zsync_file_path)
		.unwrap();
		file.write_all(format!("zsync: {}\n", zsync_file_info.zsync).as_bytes())?;
		file.write_all(format!("Filename: {}\n", zsync_file_info.filename).as_bytes())?;
		file.write_all(format!("MTime: {}\n", zsync_file_info.mtime.to_rfc2822()).as_bytes())?;
		file.write_all(format!("Blocksize: {}\n", zsync_file_info.block_size).as_bytes())?;
		file.write_all(format!("Length: {}\n", zsync_file_info.length).as_bytes())?;
		file.write_all(format!("Hash-Lengths: {},{},{}\n", zsync_file_info.seq_matches, zsync_file_info.rsum_bytes, zsync_file_info.checksum_bytes).as_bytes())?;
		file.write_all(format!("URL: {}\n", zsync_file_info.filename).as_bytes())?;
		file.write_all(format!("SHA-1: {}\n", hex::encode(zsync_file_info.sha1)).as_bytes())?;
		file.write_all(b"\n")?;
		for block_info in zsync_file_info.block_info {
			// Write trailing rsum_bytes of the rsum
			let buf: [u8; 4] = [
				((block_info.rsum >> 24) & 0xff) as u8, (block_info.rsum >> 16 & 0xff) as u8,
				((block_info.rsum >> 8) & 0xff) as u8, (block_info.rsum & 0xff) as u8
			];
			file.write_all(&buf[RSUM_SIZE - (zsync_file_info.rsum_bytes as usize) .. RSUM_SIZE])?;

			// Write leading checksum_bytes of the checksum
			file.write_all(&block_info.checksum[0 .. (zsync_file_info.checksum_bytes as usize)])?;
		}
	Ok(())
}

fn _read_zsync_file(zsync_file_path: PathBuf) -> Result<ZsyncFileInfo, Box<dyn std::error::Error>> {
	let file = File::open(zsync_file_path)?;
	let mut zsync_file_info = ZsyncFileInfo {
		zsync: "".to_string(),
		filename: "".to_string(),
		url: "".to_string(),
		sha1: [0u8; 20],
		mtime: Utc::now(),
		length: 0,
		block_size: 4096,
		seq_matches: 1,
		rsum_bytes: 4,
		checksum_bytes: 16,
		block_info: Vec::new()
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
			}
			else if option == "filename" {
				zsync_file_info.filename = value;
			}
			else if option == "url" {
				zsync_file_info.url = value;
			}
			else if option == "sha-1" {
				zsync_file_info.sha1 = hex::decode(value).unwrap().try_into().unwrap();
			}
			else if option == "mtime" {
				zsync_file_info.mtime = DateTime::parse_from_rfc2822(value.as_str()).unwrap().with_timezone(&Utc);
			}
			else if option == "blocksize" {
				zsync_file_info.block_size = value.parse()?;
			}
			else if option == "length" {
				zsync_file_info.length = value.parse()?;
			}
			else if option == "hash-lengths" {
				let mut hash_splitter = value.splitn(3, ",");
				zsync_file_info.seq_matches = hash_splitter.next().unwrap().parse()?;
				zsync_file_info.rsum_bytes = hash_splitter.next().unwrap().parse()?;
				zsync_file_info.checksum_bytes = hash_splitter.next().unwrap().parse()?;
				if zsync_file_info.seq_matches < 1 || zsync_file_info.seq_matches > 2 {
					panic!("seq_matches out of range: {}", zsync_file_info.seq_matches);
				}
				if zsync_file_info.rsum_bytes < 1 || zsync_file_info.rsum_bytes > RSUM_SIZE as u8 {
					panic!("rsum_bytes out of range: {}", zsync_file_info.rsum_bytes);
				}
				if zsync_file_info.checksum_bytes < 3 || zsync_file_info.checksum_bytes > CHECKSUM_SIZE as u8 {
					panic!("checksum_bytes out of range: {}", zsync_file_info.checksum_bytes);
				}
			}
		}
	}

	let block_count: u64 = (zsync_file_info.length + u64::from(zsync_file_info.block_size) - 1) / u64::from(zsync_file_info.block_size);
	for block_id in 0..block_count {
		let mut buf = vec![0u8; zsync_file_info.rsum_bytes as usize];
		reader.read_exact(&mut buf)?;
		buf.resize(RSUM_SIZE, 0u8);
		buf.rotate_right(RSUM_SIZE - zsync_file_info.rsum_bytes as usize);
		let rsum: u32 = (buf[0] as u32) << 24 | (buf[1] as u32) << 16 | (buf[2] as u32) << 8 | (buf[3] as u32);

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
			rsum: rsum
		};
		zsync_file_info.block_info.push(block_info);
	}
	Ok(zsync_file_info)
}

#[pyfunction]
fn rs_read_zsync_file(zsync_file_path: PathBuf) -> PyResult<ZsyncFileInfo> {
	Ok(_read_zsync_file(zsync_file_path).unwrap())
}

/// A Python module implemented in Rust.
#[pymodule]
fn pyzsync(_py: Python, m: &PyModule) -> PyResult<()> {
	pyo3_log::init();

	m.add_class::<BlockInfo>()?;
	m.add_class::<ZsyncFileInfo>()?;
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