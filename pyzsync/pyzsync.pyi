# Copyright (c) 2023 uib GmbH <info@uib.de>
# This code is owned by the uib GmbH, Mainz, Germany (uib.de). All rights reserved.
# License: AGPL-3.0

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

@dataclass
class BlockInfo:
	block_id: int
	offset: int
	size: int
	rsum: int
	checksum: bytes

@dataclass
class ZsyncFileInfo:
	zsync: str
	producer: str
	filename: str
	url: str
	sha1: bytes
	sha256: bytes
	mtime: datetime
	length: int
	block_size: int
	seq_matches: int
	rsum_bytes: int
	checksum_bytes: int
	block_info: list[BlockInfo]

@dataclass
class PatchInstruction:
	"""
	Position are zero-indexed & inclusive
	"""

	source: int
	source_offset: int
	target_offset: int
	size: int

	def copy(self) -> PatchInstruction: ...

def rs_version() -> str: ...
def rs_md4(block: bytes, num_bytes: int) -> bytes: ...
def rs_rsum(block: bytes, num_bytes: int) -> int: ...
def rs_update_rsum(rsum: int, old_char: int, new_char: int, block_size: int) -> int: ...
def rs_calc_block_size(file_size: int) -> int: ...
def rs_calc_block_infos(file: Path, block_size: int, rsum_bytes: int, checksum_bytes: int) -> list[BlockInfo]: ...
def rs_read_zsync_file(zsync_file: Path) -> ZsyncFileInfo: ...
def rs_write_zsync_file(zsync_info: ZsyncFileInfo, zsync_file: Path) -> None: ...
def rs_create_zsync_file(file: Path, zsync_file: Path, legacy_mode: bool) -> None: ...
def rs_create_zsync_info(file: Path, legacy_mode: bool) -> ZsyncFileInfo: ...
def rs_get_patch_instructions(
	zsync_info: ZsyncFileInfo, files: list[Path], progress_callback: Callable | None = None
) -> list[PatchInstruction]: ...
