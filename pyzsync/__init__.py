# Copyright (c) 2023 uib GmbH <info@uib.de>
# This code is owned by the uib GmbH, Mainz, Germany (uib.de). All rights reserved.
# License: AGPL-3.0

import hashlib
import time
from contextlib import ExitStack
from io import BytesIO
from pathlib import Path
from typing import BinaryIO, Callable, Literal, NamedTuple, cast

from pyzsync.pyzsync import (
	BlockInfo,
	PatchInstruction,
	ZsyncFileInfo,
	rs_calc_block_infos,
	rs_calc_block_size,
	rs_create_zsync_file,
	rs_create_zsync_info,
	rs_get_patch_instructions,
	rs_md4,
	rs_read_zsync_file,
	rs_rsum,
	rs_update_rsum,
	rs_version,
	rs_write_zsync_file,
)

SOURCE_REMOTE = -1

__version__ = rs_version()


class Range(NamedTuple):
	"""Range (zero-indexed & inclusive). 0-1023 = first 1024 bytes"""

	start: int
	end: int


class FileRangeReader(BytesIO):
	"""File-like reader that reads chunks of bytes from a file controlled by a list of ranges."""

	def __init__(self, file: Path, ranges: list[Range]) -> None:
		self.file = file
		self.ranges = sorted(ranges, key=lambda r: r.start)
		self.range_index = 0
		self.range_pos = self.ranges[0].start

	def read(self, size: int | None = None) -> bytes:
		bytes_needed = size
		if not bytes_needed:
			bytes_needed = self.ranges[self.range_index].end - self.range_pos
			bytes_needed += sum([r.end - r.start + 1 for r in self.ranges[self.range_index + 1 :]])
		data = b""
		with self.file.open("rb") as file:
			while bytes_needed != 0:
				range = self.ranges[self.range_index]
				range_size = range.end + 1 - self.range_pos
				read_size = bytes_needed
				start = self.range_pos
				if range_size <= read_size:
					read_size = range_size
					if self.range_index < len(self.ranges) - 1:
						self.range_index += 1
						self.range_pos = self.ranges[self.range_index].start
				else:
					self.range_pos += read_size
				file.seek(start)
				data += file.read(read_size)
				bytes_needed -= read_size
		return data


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


def create_zsync_file(file: Path, zsync_file: Path, *, legacy_mode: bool = False) -> None:
	return rs_create_zsync_file(file, zsync_file, legacy_mode)


def create_zsync_info(file: Path, *, legacy_mode: bool = False) -> ZsyncFileInfo:
	return rs_create_zsync_info(file, legacy_mode)


def get_patch_instructions(zsync_info: ZsyncFileInfo, files: Path | list[Path]) -> list[PatchInstruction]:
	if not isinstance(files, list):
		files = [files]
	return rs_get_patch_instructions(zsync_info, files)


def patch_file(
	files: Path | list[Path],
	instructions: list[PatchInstruction],
	fetch_function: Callable,
	*,
	output_file: Path | None = None,
	delete_files: bool = True,
	return_hash: Literal["sha1", "sha256"] | None = "sha1",
) -> bytes:
	"""
	Patches file by given instructions.
	Remote instructions are fetched via fetch_function.

	:param files: A local file or a list of local files to read the supplied local instructions from.
	:param instructions: List of patch instructions.
	:param fetch_function: A function that must accept a list of Ranges and return a file-like object with read method.
	:param output_file: Output file. If `None` first supplied file will be used as output file.
	:param delete_files: Delete supplied extra files after patching.
	:param return_hash: Type of hash to return. If `None` no hash will be returned.
	:return: Hash of patched file.
	"""
	if not isinstance(files, list):
		files = [files]

	if not output_file:
		output_file = files[0]

	timestamp_millis = int(1000 * time.time())
	tmp_file = output_file.with_name(f"{output_file.name}.zsync-tmp-{timestamp_millis}").resolve()
	for idx, file in enumerate(files):
		files[idx] = file.resolve()
		if files[idx] == tmp_file:
			raise ValueError(f"Invalid filename {files[idx]}")

	_hash = None
	if return_hash:
		_hash = hashlib.new(return_hash)

	remote_ranges: list[Range] = [Range(i.source_offset, i.source_offset + i.size - 1) for i in instructions if i.source == SOURCE_REMOTE]
	stream: BinaryIO | None = None
	if remote_ranges:
		stream = fetch_function(remote_ranges)
		if not stream or not hasattr(stream, "read"):
			raise ValueError("fetch_function must return a file-like object")
	stream = cast(BinaryIO, stream)

	with ExitStack() as stack:
		fhs = [stack.enter_context(open(file, "rb")) for file in files]
		with open(tmp_file, "wb") as fht:
			for instruction in instructions:
				if instruction.source == SOURCE_REMOTE:
					data = stream.read(instruction.size)
				else:
					fhs[instruction.source].seek(instruction.source_offset)
					data = fhs[instruction.source].read(instruction.size)
				fht.write(data)
				if _hash:
					_hash.update(data)

	if output_file.exists():
		output_file.unlink()
	tmp_file.rename(output_file)

	if delete_files:
		for file in files:
			if file.exists() and file != output_file:
				file.unlink()

	if _hash:
		return _hash.digest()
	return b""
