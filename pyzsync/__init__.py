from pathlib import Path
from datetime import datetime
from typing import Literal, Callable
from dataclasses import dataclass
from enum import IntEnum
import hashlib

from pyzsync.pyzsync import (
	rs_md4, rs_rsum, rs_update_rsum, rs_calc_block_infos,
	rs_read_zsync_file, rs_write_zsync_file, rs_create_zsync_file,
	rs_calc_block_size, rs_get_patch_instructions, BlockInfo, ZsyncFileInfo
)
"""
class BlockInfo:
	block_id: int
	offset: int
	rsum: int
	checksum: bytes

class ZsyncFileInfo:
	zsync: str
	filename: str
	url: str
	sha1: bytes
	mtime: datetime
	length: int
	block_size: int
	seq_matches: int
	rsum_bytes: int
	checksum_bytes: int
	block_info: list[BlockInfo]
"""
class Source(IntEnum):
	Local = 1
	Remote = 2

class PatchInstruction:
	source: Source
	source_offset: int
	target_offset: int
	size: int

@dataclass(kw_only=True, slots=True)
class PatchInstruction:
	"""
	Position are zero-indexed & inclusive
	"""
	source: Literal["local", "remote"]
	source_offset: int
	target_offset: int
	size: int

def md4(block: bytes, num_bytes: int = 16) -> bytes:
	return rs_md4(block, num_bytes)

def rsum(block: bytes, num_bytes: int = 4) -> int:
	return rs_rsum(block, num_bytes)

def update_rsum(rsum: int, old_char: int, new_char: int) -> int:
	return rs_update_rsum(rsum, old_char, new_char)

def calc_block_size(file_size: int) -> int:
	return rs_calc_block_size(file_size)

def calc_block_infos(file: Path, block_size: int, rsum_bytes: int = 4, checksum_bytes: int = 16) -> list[BlockInfo]:
	return rs_calc_block_infos(file, block_size, rsum_bytes, checksum_bytes)

def read_zsync_file(zsync_file: Path) -> ZsyncFileInfo:
	return rs_read_zsync_file(zsync_file)

def write_zsync_file(zsync_info: ZsyncFileInfo, zsync_file: Path) -> None:
	return rs_write_zsync_file(zsync_info, zsync_file)

def create_zsync_file(file: Path, zsync_file: Path) -> None:
	return rs_create_zsync_file(file, zsync_file)

def get_patch_instructions(zsync_info: ZsyncFileInfo, file: Path) -> list[PatchInstruction]:
	return rs_get_patch_instructions(zsync_info, file)

def optimize_instructions_for_http(instructions: list[PatchInstruction]) -> list[PatchInstruction]:
	# TODO
	return instructions

def patch_file(file: Path, instructions: list[PatchInstruction], fetch_function: Callable) -> bytes:
	"""
	Returns SHA-1 digest
	"""
	sha1 = hashlib.new("sha1")
	tmp_file = file.with_name(f"{file.name}.zsync-tmp")
	with (
		open(file, "rb") as lfile,
		open(tmp_file, "wb") as tfile
	):
		for instruction in instructions:
			if instruction.source == Source.Local:
				lfile.seek(instruction.source_offset)
				data = lfile.read(instruction.size)
			else:
				data = fetch_function(instruction.source_offset, instruction.size)
			tfile.write(data)
			sha1.update(data)
	file.unlink()
	tmp_file.rename(file)
	return sha1.digest()
