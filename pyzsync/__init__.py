# Copyright (c) 2023 uib GmbH <info@uib.de>
# This code is owned by the uib GmbH, Mainz, Germany (uib.de). All rights reserved.
# License: AGPL-3.0

from __future__ import annotations

import hashlib
import time
from collections.abc import MutableMapping
from contextlib import ExitStack
from http.client import HTTPConnection, HTTPSConnection
from logging import getLogger
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Generator, Iterator, Literal, NamedTuple, cast
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


class Chunk(NamedTuple):
	"""A chunk of data with a range info"""

	range: Range
	data: bytes


class ProgressListener:
	def progress_changed(self, reader: RangeReader, position: int, total: int, per_second: int) -> None:
		pass


class RangeReader:
	chunk_size = 65536

	def __init__(self, ranges: list[Range]) -> None:
		self._ranges = ranges
		self.total_position = 0
		self.total_size = sum(r.end - r.start + 1 for r in self._ranges)
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

	def read(self) -> Generator[Chunk, None, None]:
		"""
		Read ranges in chunks.
		The generated chunks are not necessarily ordered by start offset.
		"""
		raise NotImplementedError("Not implemented")


class FileRangeReader(RangeReader):
	"""File-like reader that reads chunks of bytes from a file controlled by a list of ranges."""

	def __init__(self, file: Path, ranges: list[Range]) -> None:
		super().__init__(ranges)
		self._file = file

	def read(self) -> Generator[Chunk, None, None]:
		with self._file.open("rb") as file:
			for rng in self._ranges:
				file.seek(rng.start)
				start = rng.start
				end = rng.end
				while start < rng.end:
					read_size = end - start + 1
					if read_size > self.chunk_size:
						read_size = self.chunk_size

					data = file.read(read_size)
					data_len = len(data)
					if read_size < data_len:
						raise RuntimeError(f"Failed to read bytes {start}-{end} from {self._file}")

					yield Chunk(range=Range(start, end), data=data)
					self.total_position += data_len
					self._call_progress_listeners()

					start += data_len
					end += data_len
					if end > rng.end:
						end = rng.end


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
		self._requests = [self._ranges[x : x + max_ranges_per_request] for x in range(0, len(self._ranges), max_ranges_per_request)]
		self._request_index = -1
		self._content_range: Range | None = None
		self._part_index = -1
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

	def _parse_content_range(self, content_range: str) -> Range:
		unit, range = content_range.split(" ", 1)
		if unit.strip() != "bytes":
			raise RuntimeError(f"Invalid Content-Range unit {unit}")
		try:
			start_end = range.split("/", 1)[0].split("-")
			return Range(int(start_end[0].strip()), int(start_end[1].strip()))
		except Exception as err:
			raise RuntimeError(f"Failed to parse Content-Range: {err}") from err

	def _request(self) -> None:
		self._request_index += 1
		self._content_range = None
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

		ctype = response_headers["Content-Type"]
		if ctype.startswith("multipart/byteranges"):
			boundary = [p.split("=", 1)[1].strip() for p in ctype.split(";") if p.strip().startswith("boundary=")]
			if not boundary:
				raise RuntimeError("No boundary found in Content-Type")
			self._boundary = boundary[0].encode("ascii")
			self._in_body = False
			# Content-Range will be read from multipart header
		else:
			content_range = response_headers.get("Content-Range")
			if not content_range:
				raise RuntimeError("Content-Range header missing")

			self._content_range = self._parse_content_range(content_range)
			if self._content_range != self._requests[self._request_index][0]:
				raise RuntimeError(f"Content-Range {content_range} does not match requested ranges {self._requests[self._request_index]}")

	def read(self) -> Generator[Chunk, None, None]:
		raw_data = b""
		range_pos = -1
		while True:
			if not self._content_range or range_pos > self._content_range.end:
				if self._request_index >= len(self._requests) - 1:
					return
				self._request()
				raw_data = b""
				range_pos = -1

			raw_data += self._read_response_data(self._chunk_size)
			data = b""

			if self._boundary:
				if not self._in_body:
					idx = raw_data.find(b"\r\n--" + self._boundary)
					if idx == -1:
						raise RuntimeError("Failed to read multipart")
					idx2 = raw_data.find(b"\r\n\r\n", len(self._boundary) + 4)
					if idx2 == -1:
						raise RuntimeError("Failed to read multipart")

					self._part_index += 1
					part_headers = CaseInsensitiveDict()
					for header in raw_data[idx:idx2].split(b"\r\n"):
						if b":" in header:
							name, value = header.decode("utf-8", "replace").split(":", 1)
							part_headers[name.strip()] = value.strip()
					logger.debug("Multipart headers: %r", part_headers)

					# https://datatracker.ietf.org/doc/html/rfc7233#section-4.1
					#   When multiple ranges are requested, a server MAY coalesce any of the
					#   ranges that overlap, or that are separated by a gap that is smaller
					#   than the overhead of sending multiple parts, regardless of the order
					#   in which the corresponding byte-range-spec appeared in the received
					#   Range header field.
					content_range = part_headers.get("Content-Range")
					if not content_range:
						raise RuntimeError(f"Content-Range header missing in part #{self._part_index}")

					self._content_range = self._parse_content_range(content_range)
					raw_data = raw_data[idx2 + 4 :]
					self._in_body = True

				idx = raw_data.find(b"\r\n--" + self._boundary)
				if idx == -1:
					data = raw_data
					raw_data = b""
				else:
					data = raw_data[:idx]
					raw_data = raw_data[idx:]
					self._in_body = False
			else:
				data = raw_data
				raw_data = b""

			assert isinstance(self._content_range, Range)
			if range_pos == -1:
				range_pos = self._content_range.start

			data_len = len(data)
			if not data:
				raise RuntimeError("Failed to read data")
			chunk = Chunk(range=Range(range_pos, range_pos + data_len - 1), data=data)
			yield chunk

			range_pos += data_len
			self.total_position += data_len
			if self.total_position > self.total_size:
				self.total_position = self.total_size

			self._call_progress_listeners()


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


def create_zsync_file(file: Path, zsync_file: Path, *, legacy_mode: bool = True) -> None:
	return rs_create_zsync_file(file, zsync_file, legacy_mode)


def create_zsync_info(file: Path, *, legacy_mode: bool = True) -> ZsyncFileInfo:
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

	remote_ranges: list[Range] = [Range(i.source_offset, i.source_offset + i.size - 1) for i in instructions if i.source == SOURCE_REMOTE]
	range_reader: RangeReader | None = None
	if remote_ranges:
		range_reader = cast(RangeReader, range_reader_factory(remote_ranges))

	chunk_size = 65536
	with ExitStack() as stack:
		fhs = [stack.enter_context(open(file, "rb")) for file in files]
		with open(tmp_file, "wb") as fht:
			# lseek() allows the file offset to be set beyond the end of the
			# file (but this does not change the size of the file).  If data is
			# later written at this point, subsequent reads of the data in the
			# gap (a "hole") return null bytes ('\0') until data is actually
			# written into the gap.
			for instruction in instructions:
				if instruction.source == SOURCE_REMOTE:
					continue
				bytes_read = 0
				while bytes_read < instruction.size:
					read_size = instruction.size - bytes_read
					if read_size > chunk_size:
						read_size = chunk_size
					fhs[instruction.source].seek(instruction.source_offset)
					data = fhs[instruction.source].read(read_size)
					fht.seek(instruction.target_offset)
					fht.write(data)
					bytes_read += len(data)

			if range_reader:
				for chunk in range_reader.read():
					fht.seek(chunk.range.start)
					fht.write(chunk.data)

	if output_file.exists():
		output_file.unlink()
	tmp_file.rename(output_file)

	if delete_files:
		for file in files:
			if file.exists() and file != output_file:
				file.unlink()

	if not return_hash:
		return b""

	_hash = hashlib.new(return_hash)
	with open(output_file, "rb") as fht:
		while data := fht.read(chunk_size):
			_hash.update(data)
	return _hash.digest()
