# Copyright (c) 2023 uib GmbH <info@uib.de>
# This code is owned by the uib GmbH, Mainz, Germany (uib.de). All rights reserved.
# License: AGPL-3.0

import hashlib
from pathlib import Path
from typing import Callable, Literal

from pyzsync.pyzsync import (
	BlockInfo,
	PatchInstruction,
	Source,
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
	rs_write_zsync_file,
)


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


def create_zsync_info(file: Path) -> ZsyncFileInfo:
	return rs_create_zsync_info(file)


def get_patch_instructions(zsync_info: ZsyncFileInfo, file: Path) -> list[PatchInstruction]:
	return rs_get_patch_instructions(zsync_info, file)


def patch_file(
	file: Path,
	instructions: list[PatchInstruction],
	fetch_function: Callable,
	*,
	output_file: Path | None = None,
	return_hash: Literal["sha1", "sha256"] | None = "sha1",
) -> bytes:
	"""
	Returns SHA-1 digest
	"""
	if not output_file:
		output_file = file
	_hash = None
	if return_hash:
		_hash = hashlib.new(return_hash)
	tmp_file = output_file.with_name(f"{output_file.name}.zsync-tmp")
	with (open(file, "rb") as lfile, open(tmp_file, "wb") as tfile):
		for instruction in instructions:
			if instruction.source == Source.Local:
				lfile.seek(instruction.source_offset)
				data = lfile.read(instruction.size)
			else:
				data = fetch_function(instruction.source_offset, instruction.size)
			tfile.write(data)
			if _hash:
				_hash.update(data)
	if output_file.exists():
		output_file.unlink()
	tmp_file.rename(output_file)
	if _hash:
		return _hash.digest()
	return b""
