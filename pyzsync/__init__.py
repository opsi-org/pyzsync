# Copyright (c) 2023 uib GmbH <info@uib.de>
# This code is owned by the uib GmbH, Mainz, Germany (uib.de). All rights reserved.
# License: AGPL-3.0

import hashlib
import time
from contextlib import ExitStack
from http.client import HTTPConnection, HTTPSConnection
from io import BytesIO
from logging import getLogger
from pathlib import Path
from typing import BinaryIO, Callable, Literal, NamedTuple, cast
from urllib.parse import urlparse

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

logger = getLogger("pyzsync")


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


class HTTPRangeReader(BytesIO):
	"""File-like reader that reads chunks of bytes over HTTP controlled by a list of ranges."""

	def __init__(self, url: str, ranges: list[Range], *, headers: dict[str, str] | None = None) -> None:
		self.url = urlparse(url)
		self.ranges = sorted(ranges, key=lambda r: r.start)
		self.headers = cast(dict[str, str], headers or {})

		byte_ranges = ", ".join(f"{r.start}-{r.end}" for r in self.ranges)
		self.headers["Range"] = f"bytes={byte_ranges}"
		self.headers["Accept-Encoding"] = "identity"

		conn_class = HTTPConnection if self.url.scheme == "http" else HTTPSConnection
		self.connection = conn_class(self.url.netloc, timeout=8 * 3600, blocksize=65536)
		logger.info("Sending GET request to %s", self.url.geturl())
		self.connection.request("GET", self.url.path, headers=self.headers)
		self.response = self.connection.getresponse()
		logger.debug("Received response: %r, headers: %r", self.response.status, dict(self.response.headers))
		if self.response.status < 200 or self.response.status > 299:
			raise RuntimeError(f"Failed to fetch ranges from {self.url.geturl()}: {self.response.status} - {self.response.read()}")

		self.total_size = int(self.response.headers["Content-Length"])
		self.position = 0
		self.percentage = -1
		self.raw_data = b""
		self.data = b""
		self.in_body = False
		self.boundary = b""
		ctype = self.response.headers["Content-Type"]
		if ctype.startswith("multipart/byteranges"):
			boundary = [p.split("=", 1)[1].strip() for p in ctype.split(";") if p.strip().startswith("boundary=")]
			if not boundary:
				raise ValueError("No boundary found in Content-Type")
			self.boundary = boundary[0].encode("ascii")

	def read(self, size: int | None = None) -> bytes:
		if not size:
			size = self.total_size - self.position
		return_data = b""
		if self.boundary:
			while len(self.data) < size:
				self.raw_data += self.response.read(65536)
				if not self.in_body:
					idx = self.raw_data.find(self.boundary)
					if idx == -1:
						raise RuntimeError("Failed to read multipart")
					idx = self.raw_data.find(b"\n\n", idx)
					if idx == -1:
						raise RuntimeError("Failed to read multipart")
					self.raw_data = self.raw_data[idx + 2 :]
					self.in_body = True

				idx = self.raw_data.find(b"\n--" + self.boundary)
				if idx == -1:
					self.data += self.raw_data
				else:
					self.data += self.raw_data[:idx]
					self.raw_data = self.raw_data[idx:]
					self.in_body = False
			return_data = self.data[:size]
			self.data = self.data[size:]
		else:
			return_data = self.response.read(size)

		self.position += len(return_data)

		percentage = int(self.position * 100 / self.total_size)
		if percentage > self.percentage:
			self.percentage = percentage
			logger.info(
				"zsync %r: %s%% (%0.2f/%0.2f MB)",
				self.url.geturl(),
				self.percentage,
				self.position / 1_000_000,
				self.total_size / 1_000_000,
			)
		return return_data


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
	chunk_size = 65536
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
				bytes_read = 0
				if instruction.source != SOURCE_REMOTE:
					fhs[instruction.source].seek(instruction.source_offset)
				while bytes_read < instruction.size:
					read_size = instruction.size - bytes_read
					if read_size > chunk_size:
						read_size = chunk_size
					if instruction.source == SOURCE_REMOTE:
						data = stream.read(read_size)
					else:
						data = fhs[instruction.source].read(read_size)
					fht.write(data)
					if _hash:
						_hash.update(data)
					bytes_read += read_size
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
