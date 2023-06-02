# Copyright (c) 2023 uib GmbH <info@uib.de>
# This code is owned by the uib GmbH, Mainz, Germany (uib.de). All rights reserved.
# License: AGPL-3.0

from __future__ import annotations

import hashlib
import time
from collections.abc import MutableMapping
from contextlib import ExitStack
from http.client import HTTPConnection, HTTPSConnection
from io import BytesIO
from logging import getLogger
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Iterator, Literal, NamedTuple, cast
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


# Based on requests CaseInsensitiveDict
class CaseInsensitiveDict(MutableMapping):
	"""A case-insensitive ``dict``-like object"""

	def __init__(self, data: dict[str, Any] | None = None, **kwargs: Any) -> None:
		self._store: dict[str, tuple[str, Any]] = {}
		self.update(data or {}, **kwargs)

	def __setitem__(self, key: str, value: Any) -> None:
		self._store[key.lower()] = (key, value)

	def __getitem__(self, key: str) -> Any:
		return self._store[key.lower()][1]

	def __delitem__(self, key: str) -> None:
		del self._store[key.lower()]

	def __iter__(self) -> Iterator[str]:
		return (cased_key for cased_key, _ in self._store.values())

	def __len__(self) -> int:
		return len(self._store)

	def __repr__(self) -> str:
		return str(dict(self.items()))


class Range(NamedTuple):
	"""Range (zero-indexed & inclusive). 0-1023 = first 1024 bytes"""

	start: int
	end: int


class ProgressListener:
	def progress_changed(self, reader: RangeReader, position: int, total: int, per_second: int) -> None:
		pass


class RangeReader(BytesIO):
	def __init__(self, ranges: list[Range]) -> None:
		self.total_position = 0
		self.total_size = sum(r.end - r.start + 1 for r in ranges)
		self.per_second = 0
		self._ps_last_time = time.time()
		self._ps_last_position = 0
		self._progress_listeners: list[ProgressListener] = []
		self._progress_listener_lock = Lock()

	def register_progress_listener(self, listener: ProgressListener) -> None:
		with self._progress_listener_lock:
			if listener not in self._progress_listeners:
				self._progress_listeners.append(listener)

	def unregister_progress_listener(self, listener: ProgressListener) -> None:
		with self._progress_listener_lock:
			if listener in self._progress_listeners:
				self._progress_listeners.remove(listener)

	def _call_progress_listeners(self) -> None:
		now = time.time()
		elapsed = now - self._ps_last_time
		per_second = (self.total_position - self._ps_last_position) / elapsed if elapsed else 0.0
		self.per_second = int(self.per_second * 0.7 + per_second * 0.3)
		self._last_time = now

		with self._progress_listener_lock:
			for progress_listener in self._progress_listeners:
				try:
					progress_listener.progress_changed(self, self.total_position, self.total_size, self.per_second)
				except Exception as err:
					logger.warning(err)

	def read(self, size: int | None = None) -> bytes:
		raise NotImplementedError("Not implemented")


class FileRangeReader(RangeReader):
	"""File-like reader that reads chunks of bytes from a file controlled by a list of ranges."""

	def __init__(self, file: Path, ranges: list[Range]) -> None:
		super().__init__(ranges)

		self.file = file
		self.ranges = sorted(ranges, key=lambda r: r.start)
		self.range_index = 0
		self.range_pos = self.ranges[0].start

	def read(self, size: int | None = None) -> bytes:
		bytes_needed = size
		if not bytes_needed:
			bytes_needed = self.total_size - self.total_position

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

		self.total_position += len(data)

		self._call_progress_listeners()

		return data


class HTTPRangeReader(RangeReader):
	"""File-like reader that reads chunks of bytes over HTTP controlled by a list of ranges."""

	_chunk_size = 128 * 1000

	def __init__(
		self,
		url: str,
		ranges: list[Range],
		*,
		headers: CaseInsensitiveDict | dict[str, str] | None = None,
		max_ranges_per_request: int = 100,
		read_timeout: int = 8 * 3600,
	) -> None:
		super().__init__(ranges)
		self._url = urlparse(url)
		self._headers = headers if isinstance(headers, CaseInsensitiveDict) else CaseInsensitiveDict(headers)
		self._headers["Accept-Encoding"] = "identity"
		self._read_timeout = read_timeout

		# Max header size ~4k
		ranges = sorted(ranges, key=lambda r: r.start)
		self._requests = [ranges[x : x + max_ranges_per_request] for x in range(0, len(ranges), max_ranges_per_request)]
		self._request_index = -1
		self._content_size = 0
		self._content_position = 0
		self._part_index = 0
		self._raw_data = b""
		self._data = b""
		self._in_body = False
		self._boundary = b""
		self._response: Any = None
		self._session: Any = None

	@property
	def url(self) -> str:
		return self._url.geturl()

	def _send_request(self) -> tuple[int, CaseInsensitiveDict]:
		conn_class = HTTPConnection if self._url.scheme == "http" else HTTPSConnection
		self._session = conn_class(self._url.netloc, timeout=self._read_timeout, blocksize=self._chunk_size)
		self._session.request("GET", self._url.path, headers=self._headers)
		self._response = self._session.getresponse()
		return self._response.status, CaseInsensitiveDict(dict(self._response.getheaders()))

	def _read_response_data(self, size: int | None = None) -> bytes:
		return self._response.read(size)

	def _parse_content_range(self, content_range: str) -> list[Range]:
		unit, range = content_range.split(" ", 1)
		if unit.strip() != "bytes":
			raise RuntimeError(f"Invalid Content-Range unit {unit}")
		try:
			return [Range(int(r.split("-")[0].strip()), int(r.split("-")[1].strip())) for r in range.split("/", 1)[0].split(",")]
		except Exception as err:
			raise RuntimeError(f"Failed to parse Content-Range: {err}") from err

	def _request(self) -> None:
		self._request_index += 1
		self._content_position = 0
		self._part_index = -1
		self._boundary = b""
		self._in_body = True
		byte_ranges = ", ".join(f"{r.start}-{r.end}" for r in self._requests[self._request_index])
		self._headers["Range"] = f"bytes={byte_ranges}"

		logger.info("Sending GET request #%d to %s", self._request_index, self._url.geturl())
		logger.debug("Sending GET request with headers: %r", self._headers)
		response_code, response_headers = self._send_request()
		logger.debug("Received response: %r, headers: %r", response_code, response_headers)
		if response_code < 200 or response_code > 299:
			raise RuntimeError(
				f"Failed to fetch ranges from {self._url.geturl()}: "
				f"{response_code} - {self._read_response_data(self._chunk_size).decode('utf-8', 'replace')}"
			)

		content_range = response_headers.get("Content-Range")
		if content_range:
			# Content-Range can also be placed in multipart header
			ranges = self._parse_content_range(content_range)
			if ranges != self._requests[self._request_index]:
				raise RuntimeError(f"Content-Range {content_range} does not match requested ranges {self._requests[self._request_index]}")

		self._content_size = int(response_headers["Content-Length"])
		ctype = response_headers["Content-Type"]
		if ctype.startswith("multipart/byteranges"):
			boundary = [p.split("=", 1)[1].strip() for p in ctype.split(";") if p.strip().startswith("boundary=")]
			if not boundary:
				raise RuntimeError("No boundary found in Content-Type")
			self._boundary = boundary[0].encode("ascii")
			self._in_body = False

	def read(self, size: int | None = None) -> bytes:
		if not size:
			size = self.total_size - self.total_position

		return_data = b""
		while len(self._data) < size:
			if self._content_position >= self._content_size and self._request_index < len(self._requests) - 1:
				self._request()

			chunk = self._read_response_data(self._chunk_size)
			self._content_position += len(chunk)

			if self._boundary:
				self._raw_data += chunk
				if not self._in_body:
					idx = self._raw_data.find(b"\r\n--" + self._boundary)
					if idx == -1:
						raise RuntimeError("Failed to read multipart")
					idx2 = self._raw_data.find(b"\r\n\r\n", len(self._boundary) + 4)
					if idx2 == -1:
						raise RuntimeError("Failed to read multipart")

					self._part_index += 1
					part_headers = CaseInsensitiveDict()
					for header in self._raw_data[idx:idx2].split(b"\r\n"):
						if b":" in header:
							name, value = header.decode("utf-8", "replace").split(":", 1)
							part_headers[name.strip()] = value.strip()
					logger.debug("Multipart headers: %r", part_headers)

					content_range = part_headers.get("Content-Range")
					if content_range:
						ranges = self._parse_content_range(content_range)
						cur_range = self._requests[self._request_index][self._part_index]
						if ranges[0] != cur_range:
							raise RuntimeError(f"Content-Range {content_range} does not match range {cur_range}")

					self._raw_data = self._raw_data[idx2 + 4 :]
					self._in_body = True

				idx = self._raw_data.find(b"\r\n--" + self._boundary)
				if idx == -1:
					self._data += self._raw_data
					self._raw_data = b""
				else:
					self._data += self._raw_data[:idx]
					self._raw_data = self._raw_data[idx:]
					self._in_body = False
			else:
				self._data += chunk

		return_data = self._data[:size]
		self._data = self._data[size:]
		self.total_position += len(return_data)

		self._call_progress_listeners()

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
	range_reader_factory: Callable,
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
	:param range_reader_factory: A function that must return a RangeReader object.
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
	range_reader: RangeReader | None = None
	if remote_ranges:
		range_reader = cast(RangeReader, range_reader_factory(remote_ranges))

	with ExitStack() as stack:
		fhs = [stack.enter_context(open(file, "rb")) for file in files]
		with open(tmp_file, "wb") as fht:
			for instruction in instructions:
				bytes_read = 0
				if instruction.source == SOURCE_REMOTE:
					logger.debug(
						"Processing remote instruction %d/%d size=%d",
						instruction.source_offset,
						instruction.source_offset + instruction.size - 1,
						instruction.size,
					)
				if instruction.source != SOURCE_REMOTE:
					fhs[instruction.source].seek(instruction.source_offset)
				while bytes_read < instruction.size:
					read_size = instruction.size - bytes_read
					if read_size > chunk_size:
						read_size = chunk_size
					if instruction.source == SOURCE_REMOTE:
						assert range_reader
						data = range_reader.read(read_size)
					else:
						data = fhs[instruction.source].read(read_size)
					fht.write(data)
					if _hash:
						_hash.update(data)
					bytes_read += len(data)
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
