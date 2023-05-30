# Copyright (c) 2023 uib GmbH <info@uib.de>
# This code is owned by the uib GmbH, Mainz, Germany (uib.de). All rights reserved.
# License: AGPL-3.0

from enum import IntEnum
from datetime import datetime

class BlockInfo:
	block_id: int
	offset: int
	size: int
	rsum: int
	checksum: bytes

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

class Source(IntEnum):
	Local = 1
	Remote = 2

class PatchInstruction:
	"""
	Position are zero-indexed & inclusive
	"""
	source: Source
	source_offset: int
	target_offset: int
	size: int