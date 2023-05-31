# Copyright (c) 2023 uib GmbH <info@uib.de>
# This code is owned by the uib GmbH, Mainz, Germany (uib.de). All rights reserved.
# License: AGPL-3.0

import hashlib
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Callable, Literal

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
	Returns SHA-1 digest
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

	with ExitStack() as stack:
		fhs = [stack.enter_context(open(file, "rb")) for file in files]
		with open(tmp_file, "wb") as fht:
			for instruction in instructions:
				if instruction.source == SOURCE_REMOTE:
					data = fetch_function(instruction.source_offset, instruction.size)
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
