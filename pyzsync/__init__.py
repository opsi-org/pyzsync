# Copyright (c) 2023 uib GmbH <info@uib.de>
# This code is owned by the uib GmbH, Mainz, Germany (uib.de). All rights reserved.
# License: AGPL-3.0

from __future__ import annotations

import hashlib
import time
from collections import defaultdict
from collections.abc import MutableMapping
from http.client import HTTPConnection, HTTPSConnection
from logging import getLogger
from pathlib import Path
from threading import Lock
from typing import Any, BinaryIO, Callable, Iterator, Literal, NamedTuple
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

	@property
	def size(self) -> int:
		return self.end - self.start + 1


class ProgressListener:
	def progress_changed(self, patcher: Patcher, position: int, total: int, per_second: int) -> None:
		pass


def instructions_by_source_range(instructions: list[PatchInstruction]) -> dict[Range, list[PatchInstruction]]:
	result = defaultdict(list)
	for inst in instructions:
		rng = Range(start=inst.source_offset, end=inst.source_offset + inst.size - 1)
		result[rng].append(inst)
	return result


class Patcher:
	chunk_size = 65536

	def __init__(self, instructions: list[PatchInstruction], target_file: BinaryIO) -> None:
		self._instructions = instructions
		self._target_file = target_file
		self.total_position = 0
		self.total_size = sum(inst.size for inst in self._instructions)
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

	def run(self) -> None:
		raise NotImplementedError("Not implemented")


class FilePatcher(Patcher):
	def __init__(self, instructions: list[PatchInstruction], target_file: BinaryIO, source_file: Path) -> None:
		super().__init__(instructions, target_file)
		self._source_file = source_file

	def run(self) -> None:
		with self._source_file.open("rb") as sfh:
			for source_range, instructions in instructions_by_source_range(self._instructions).items():
				sfh.seek(source_range.start)
				bytes_read = 0
				while bytes_read < source_range.size:
					read_size = source_range.size - bytes_read
					if read_size > self.chunk_size:
						read_size = self.chunk_size
					data = sfh.read(read_size)
					data_len = len(data)
					if data_len < read_size:
						raise RuntimeError(f"Failed to read bytes {source_range.start}-{source_range.end} from {self._source_file}")

					for inst in instructions:
						self._target_file.seek(inst.target_offset + bytes_read)
						self._target_file.write(data)

					self.total_position += data_len
					bytes_read += data_len
					self._call_progress_listeners()


class HTTPPatcher(Patcher):
	def __init__(
		self,
		instructions: list[PatchInstruction],
		target_file: BinaryIO,
		url: str,
		*,
		headers: CaseInsensitiveDict | dict[str, str] | None = None,
		# Max header size ~4k
		max_ranges_per_request: int = 50,
		read_timeout: int = 8 * 3600,
	) -> None:
		super().__init__(instructions, target_file)
		self._url = urlparse(url)
		self._headers = headers if isinstance(headers, CaseInsensitiveDict) else CaseInsensitiveDict(headers)
		self._headers["Accept-Encoding"] = "identity"
		self._read_timeout = read_timeout

		self._requests: list[dict[Range, list[PatchInstruction]]] = []
		for source_range, instructions in instructions_by_source_range(self._instructions).items():
			if not self._requests or len(self._requests[-1]) == max_ranges_per_request:
				self._requests.append({source_range: instructions})
			else:
				self._requests[-1][source_range] = instructions
		self._instructions_processed: list[PatchInstruction] = []
		self._request_index = -1
		self._content_range: Range | None = None
		self._current_instructions: dict[Range, list[PatchInstruction]] = {}
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
		self._session = conn_class(self._url.netloc, timeout=self._read_timeout, blocksize=self.chunk_size)
		self._session.request("GET", self._url.path, headers=self._headers)
		self._response = self._session.getresponse()
		return self._response.status, CaseInsensitiveDict(dict(self._response.getheaders()))

	def _read_response_data(self, size: int | None = None) -> bytes:
		return self._response.read(size)

	def _set_content_range(self, content_range: str) -> None:
		unit, range = content_range.split(" ", 1)
		if unit.strip() != "bytes":
			raise RuntimeError(f"Invalid Content-Range unit {unit}")
		try:
			start_end = range.split("/", 1)[0].split("-")
			self._content_range = Range(int(start_end[0].strip()), int(start_end[1].strip()))
			self._current_instructions = {
				source_range: instructions
				for source_range, instructions in self._requests[self._request_index].items()
				if source_range.start >= self._content_range.start and source_range.end <= self._content_range.end
			}
			if not self._current_instructions:
				raise RuntimeError(
					f"Content-Range {content_range} does not match any requested range {self._requests[self._request_index]}"
				)
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

		logger.info("Sending GET request #%d to %s", self._request_index + 1, self._url.geturl())
		logger.debug("Sending GET request with headers: %r", self._headers)
		response_code, response_headers = self._send_request()
		logger.debug("Received response: %r, headers: %r", response_code, response_headers)

		if response_code < 200 or response_code > 299:
			raise RuntimeError(
				f"Failed to fetch ranges from {self._url.geturl()}: "
				f"{response_code} - {self._read_response_data(self.chunk_size).decode('utf-8', 'replace')}"
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

			self._set_content_range(content_range)

	def run(self) -> None:
		raw_data = b""
		range_pos = -1
		boundary_len = 0

		while True:
			if not raw_data and (not self._content_range or range_pos >= self._content_range.end):
				if self._current_instructions:
					missed_instructions = [
						inst for insts in self._current_instructions.values() for inst in insts if inst not in self._instructions_processed
					]
					if missed_instructions:
						raise RuntimeError(
							f"The following instructions have not been processed: {missed_instructions} (requests: {self._requests})"
						)

				if self._request_index >= len(self._requests) - 1:
					break
				self._request()
				boundary_len = len(self._boundary)
				raw_data = b""
				range_pos = self._content_range.start if self._content_range else -1

			raw_data += self._read_response_data(self.chunk_size)
			data = b""

			if self._boundary:
				# https://datatracker.ietf.org/doc/html/rfc7233#section-4.1
				#   When multiple ranges are requested, a server MAY coalesce any of the
				#   ranges that overlap, or that are separated by a gap that is smaller
				#   than the overhead of sending multiple parts, regardless of the order
				#   in which the corresponding byte-range-spec appeared in the received
				#   Range header field.
				if not self._in_body:
					idx = raw_data.find(b"\r\n--" + self._boundary)
					if idx == -1:
						raise RuntimeError("Failed to read multipart")
					if raw_data[idx + boundary_len + 4 : idx + boundary_len + 6] == b"--":
						raw_data = b""
						continue
					idx2 = raw_data.find(b"\r\n\r\n", boundary_len + 4)
					if idx2 == -1:
						raise RuntimeError("Failed to read multipart")

					self._part_index += 1
					part_headers = CaseInsensitiveDict()
					for header in raw_data[idx:idx2].split(b"\r\n"):
						if b":" in header:
							name, value = header.decode("utf-8", "replace").split(":", 1)
							part_headers[name.strip()] = value.strip()
					logger.debug("Multipart headers: %r", part_headers)

					content_range = part_headers.get("Content-Range")
					if not content_range:
						raise RuntimeError(f"Content-Range header missing in part #{self._part_index}")

					self._set_content_range(content_range)
					assert isinstance(self._content_range, Range)

					raw_data = raw_data[idx2 + 4 :]
					self._in_body = True
					range_pos = self._content_range.start

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

			data_len = len(data)
			if not data:
				raise RuntimeError("Failed to read data")

			# TODO: optimize! memoryview?
			data_start = range_pos
			data_end = range_pos + data_len - 1
			for rng, instructions in self._current_instructions.items():
				s_diff = rng.start - data_start
				e_diff = rng.end - data_end

				rng_data = data
				if s_diff > 0:
					rng_data = rng_data[s_diff:]
				if e_diff < 0:
					rng_data = rng_data[:e_diff]

				for inst in instructions:
					seek = inst.target_offset
					if s_diff < 0:
						seek += s_diff * -1
					self._target_file.seek(seek)
					self._target_file.write(rng_data)

					if seek + len(rng_data) == inst.target_offset + inst.size:
						logger.debug("Instruction %r was processed", inst)
						if inst in self._instructions_processed:
							logger.warning("Instruction %r was processed more than once", inst)
						else:
							self._instructions_processed.append(inst)

			range_pos += data_len
			self.total_position += data_len
			if self.total_position > self.total_size:
				self.total_position = self.total_size

			self._call_progress_listeners()

		missed_instructions = [inst for inst in self._instructions if inst not in self._instructions_processed]
		if missed_instructions:
			raise RuntimeError(f"The following instructions have not been processed: {missed_instructions}")


def md4(block: bytes, num_bytes: int = 16) -> bytes:
	return rs_md4(block, num_bytes)


def rsum(block: bytes, num_bytes: int = 4) -> int:
	return rs_rsum(block, num_bytes)


def update_rsum(rsum: int, old_char: int, new_char: int, block_size: int) -> int:
	return rs_update_rsum(rsum, old_char, new_char, block_size)


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
	patcher_factory: Callable,
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
	:param patcher_factory: A function that must return a Patcher object.
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
	tmp_file = output_file.with_name(f"{output_file.name}.zsync-tmp-{timestamp_millis}").absolute()
	for idx, file in enumerate(files):
		files[idx] = file.absolute()
		if files[idx] == tmp_file:
			raise ValueError(f"Invalid filename {files[idx]}")

	instructions_by_source = defaultdict(list)
	for inst in instructions:
		instructions_by_source[inst.source].append(inst)

	logger.debug("Patching temp file '%s'", tmp_file)
	with open(tmp_file, "wb") as fht:
		for source in sorted(instructions_by_source):
			logger.debug("Processing %d instructions for source %r", len(instructions_by_source[source]), source)
			if source == SOURCE_REMOTE:
				patcher = patcher_factory(instructions_by_source[source], target_file=fht)
			else:
				patcher = FilePatcher(instructions_by_source[source], target_file=fht, source_file=files[source])
			patcher.run()
	logger.debug("Temp file '%s' patching completed", tmp_file)

	if output_file.exists():
		logger.debug("Removing output file '%s'", output_file)
		output_file.unlink()

	logger.debug("Moving temp file '%s' to output file '%s'", tmp_file, output_file)
	tmp_file.rename(output_file)

	if delete_files:
		for file in files:
			if file.exists() and not file.resolve().samefile(output_file.resolve()):
				logger.debug("Deleting file '%s'", file)
				file.unlink()

	if not return_hash:
		return b""

	logger.debug("Calculating %r hash digest of output file '%s'", return_hash, output_file)
	_hash = hashlib.new(return_hash)
	with open(output_file, "rb") as fht:
		while data := fht.read(65536):
			_hash.update(data)
	return _hash.digest()
