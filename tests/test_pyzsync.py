# Copyright (c) 2023 uib GmbH <info@uib.de>
# This code is owned by the uib GmbH, Mainz, Germany (uib.de). All rights reserved.
# License: AGPL-3.0

import hashlib
import platform
import shutil
import socket
import time
from collections import Counter
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from http.server import HTTPServer
from pathlib import Path
from random import randbytes
from socketserver import ThreadingMixIn
from statistics import mean
from subprocess import run
from threading import Thread
from typing import Any, BinaryIO, Generator, Optional, cast

import pytest
from RangeHTTPServer import RangeRequestHandler  # type: ignore[import]

from pyzsync import (
	SOURCE_REMOTE,
	BlockInfo,
	CaseInsensitiveDict,
	FilePatcher,
	HTTPPatcher,
	Patcher,
	PatchInstruction,
	ProgressListener,
	Range,
	ZsyncFileInfo,
	calc_block_infos,
	calc_block_size,
	create_zsync_file,
	create_zsync_info,
	get_patch_instructions,
	instructions_by_source_range,
	md4,
	optimize_instructions,
	patch_file,
	read_zsync_file,
	rsum,
	update_rsum,
	write_zsync_file,
)


@contextmanager
def http_server(directory: Path) -> Generator[int, None, None]:
	# Select free port
	with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
		sock.bind(("", 0))
		sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
		port = sock.getsockname()[1]

	class Handler(RangeRequestHandler):
		def __init__(self, *args: Any, **kwargs: Any) -> None:
			super().__init__(*args, directory=str(directory), **kwargs)

	class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
		pass

	server = ThreadingHTTPServer(("", port), Handler)
	thread = Thread(target=server.serve_forever)
	thread.daemon = True
	thread.start()
	try:
		# Wait for server to start
		for _ in range(10):
			with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:  # pylint: disable=dotted-import-in-loop
				sock.settimeout(1)
				res = sock.connect_ex(("127.0.0.1", port))
				if res == 0:
					break
		yield port
	finally:
		server.socket.close()
		thread.join(3)


def test_instructions_by_source_range() -> None:
	instructions = [
		PatchInstruction(source=SOURCE_REMOTE, source_offset=0, target_offset=0, size=100),
		PatchInstruction(source=SOURCE_REMOTE, source_offset=0, target_offset=100, size=100),
		PatchInstruction(source=SOURCE_REMOTE, source_offset=100, target_offset=200, size=100),
		PatchInstruction(source=SOURCE_REMOTE, source_offset=100, target_offset=300, size=100),
		PatchInstruction(source=SOURCE_REMOTE, source_offset=200, target_offset=400, size=100),
		PatchInstruction(source=SOURCE_REMOTE, source_offset=200, target_offset=500, size=200),
		PatchInstruction(source=SOURCE_REMOTE, source_offset=400, target_offset=700, size=100),
	]
	result = instructions_by_source_range(instructions)
	assert len(result) == 5
	assert result[Range(start=0, end=99)][0].target_offset == 0
	assert result[Range(start=0, end=99)][1].target_offset == 100
	assert result[Range(start=100, end=199)][0].target_offset == 200
	assert result[Range(start=100, end=199)][1].target_offset == 300
	assert result[Range(start=200, end=299)][0].target_offset == 400
	assert result[Range(start=200, end=399)][0].target_offset == 500
	assert result[Range(start=400, end=499)][0].target_offset == 700


def test_md4() -> None:
	block = b"x" * 2048
	assert md4(block) == bytes.fromhex("f3b20ba5cf00653e13fcf03f85bd0224")


def test_rsum() -> None:
	block = ""
	for char in range(0, 1088):
		block += chr(char)
	bblock = block.encode("utf-8")
	assert len(bblock) == 2048
	assert hex(rsum(bblock, 4)) == "0x67a0efa0"
	assert hex(rsum(bblock, 3)) == "0xa0efa0"
	assert hex(rsum(bblock, 2)) == "0xefa0"
	assert hex(rsum(bblock, 1)) == "0xa0"


@pytest.mark.parametrize(
	"rsum_bytes, block_size",
	(
		(2, 2048),
		(3, 2048),
		(4, 2048),
		(2, 4096),
		(3, 4096),
		(4, 4096),
	),
)
def test_update_rsum(rsum_bytes: int, block_size: int) -> None:
	block = ""
	for char in range(2, 1089):
		block += chr(char)
	if block_size == 4096:
		block += block
	bblock = block.encode("utf-8")
	assert len(bblock) == block_size
	_rsum = rsum(bblock, 4)
	assert hex(_rsum) == "0xd1e0f202" if block_size == 4096 else "0x68f0b901"

	rsum_mask = (2 << (rsum_bytes * 8 - 1)) - 1
	for _ in range(2048):
		old_char = bblock[0]
		bblock = bblock[1:] + b"\03"
		new_char = bblock[-1]
		new_rsum = rsum(bblock, rsum_bytes)
		_rsum = update_rsum(_rsum, old_char, new_char, block_size)
		# print(hex(rsum))
		assert _rsum & rsum_mask == new_rsum


def test_calc_block_size() -> None:
	assert calc_block_size(1) == 2048
	assert calc_block_size(1_000_000_000) == 4096
	assert calc_block_size(2_000_000_000) == 4096


def test_hash_speed(tmp_path: Path) -> None:
	test_file = tmp_path / "local"
	file_size = 1_000_000_000
	block_size = 4096
	block_count = int((file_size + block_size - 1) / block_size)
	with open(test_file, "wb") as file:
		for _ in range(block_count):
			file.write(randbytes(block_size))

	rsum_start = time.time()
	with open(test_file, "rb") as file:
		while block := file.read(block_size):
			rsum(block)
	rsum_time = time.time() - rsum_start

	md4_start = time.time()
	with open(test_file, "rb") as file:
		while block := file.read(block_size):
			md4(block)
	md4_time = time.time() - md4_start

	print(block_count, rsum_time, md4_time)
	assert rsum_time < 5
	assert md4_time < 20

	shutil.rmtree(tmp_path)


@pytest.mark.posix
def test_get_patch_instructions(tmp_path: Path) -> None:
	remote_file = Path("tests/data/test.small")
	local_file = tmp_path / "test.small"
	zsync_file = Path("tests/data/test.small.zsync")
	info = read_zsync_file(zsync_file)
	data = remote_file.read_bytes()

	def patcher_factory(instructions: list[PatchInstruction], target_file: BinaryIO) -> Patcher:
		return FilePatcher(instructions=instructions, target_file=target_file, source_file=remote_file)

	info.seq_matches = 1
	local_file.write_bytes(data[:2048])
	instructions = get_patch_instructions(info, local_file)
	# print(instructions)

	assert instructions[0].source == 0
	assert instructions[0].source_offset == 0
	assert instructions[0].target_offset == 0
	assert instructions[0].size == 2048
	assert instructions[1].source == SOURCE_REMOTE
	assert instructions[1].source_offset == 2048
	assert instructions[1].target_offset == 2048
	assert instructions[1].size == 6961
	sha1 = patch_file(local_file, instructions, patcher_factory)
	assert sha1.hex() == info.sha1.hex()

	# blocks: L R R L L
	local_file.write_bytes(data[:2048] + data[6144:])
	instructions = get_patch_instructions(info, local_file)
	assert instructions[0].source == 0
	assert instructions[0].source_offset == 0
	assert instructions[0].target_offset == 0
	assert instructions[0].size == 2048
	assert instructions[1].source == SOURCE_REMOTE
	assert instructions[1].source_offset == 2048
	assert instructions[1].target_offset == 2048
	assert instructions[1].size == 4096
	assert instructions[2].source == 0
	assert instructions[2].source_offset == 2048
	assert instructions[2].target_offset == 6144
	assert instructions[2].size == 2048
	assert instructions[3].source == 0
	assert instructions[3].source_offset == 4096
	assert instructions[3].target_offset == 8192
	assert instructions[3].size == 817

	sha1 = patch_file(local_file, instructions, patcher_factory)
	assert sha1.hex() == info.sha1.hex()

	# Test callback
	positions = []

	def progress_callback(pos: int, total: int) -> None:
		nonlocal positions
		positions.append((pos, total))

	instructions = get_patch_instructions(info, local_file, progress_callback=progress_callback)
	assert positions == [(0, 12288), (4096, 12288), (6144, 12288), (8192, 12288), (10240, 12288), (12288, 12288)]

	def progress_callback_abort(pos: int, total: int) -> bool:
		return pos > 0

	with pytest.raises(RuntimeError, match="Aborted by progress callback"):
		instructions = get_patch_instructions(info, local_file, progress_callback=progress_callback_abort)

	# Test with seq_matches = 2
	info.seq_matches = 2
	local_file.write_bytes(data[:2048])
	instructions = get_patch_instructions(info, local_file)
	# print(instructions)

	assert instructions[0].source == SOURCE_REMOTE
	assert instructions[0].size == 9009
	assert instructions[0].source_offset == 0
	assert instructions[0].target_offset == 0
	sha1 = patch_file(local_file, instructions, patcher_factory)
	assert sha1.hex() == info.sha1.hex()


@pytest.mark.parametrize(
	"file_size_remote, file_size_local",
	(
		(1, 0),
		(100, 0),
		(1024, 0),
		(2048, 0),
		(3000, 0),
		(4096, 0),
		(5000, 0),
		(1, 1),
		(1, 100),
		(1, 2048),
		(100, 100),
		(1024, 100),
		(2048, 2048),
		(3000, 2048),
		(4096, 4096),
		(5000, 100),
	),
)
def test_patch_instructions_zero_file(tmp_path: Path, file_size_remote: int, file_size_local: int) -> None:
	remote_file = tmp_path / "remote"
	local_file = tmp_path / "local"
	zsync_file = tmp_path / "remote.zsync"

	remote_file.write_bytes(b"\0" * file_size_remote)
	local_file.write_bytes(b"\0" * file_size_local)

	create_zsync_file(remote_file, zsync_file)
	info = read_zsync_file(zsync_file)
	instructions = get_patch_instructions(info, local_file)

	def patcher_factory(instructions: list[PatchInstruction], target_file: BinaryIO) -> Patcher:
		return FilePatcher(instructions=instructions, target_file=target_file, source_file=remote_file)

	sha1 = patch_file(local_file, instructions, patcher_factory)
	assert sha1 == info.sha1
	assert local_file.read_bytes() == remote_file.read_bytes()


@pytest.mark.posix
def test_calc_block_infos(tmp_path: Path) -> None:
	# git config --global core.autocrlf false
	test_file = Path("tests/data/test.small")

	block_info = calc_block_infos(test_file, 2048, 4, 16)
	assert len(block_info) == 5
	assert sum([i.size for i in block_info]) == test_file.stat().st_size

	assert block_info[0].block_id == 0
	assert block_info[0].offset == 0
	assert block_info[0].size == 2048

	assert block_info[0].checksum == bytes.fromhex("56bd0a0924aafee3def128b5844b3058")
	assert block_info[0].rsum == 0x8BF6804D

	assert block_info[3].block_id == 3
	assert block_info[3].offset == 6144
	assert block_info[3].checksum == bytes.fromhex("709f54a2fcbc61c01177f6426d58a9b5")
	assert block_info[3].rsum == 0xBCFEE5B5

	assert block_info[4].block_id == 4
	assert block_info[4].offset == 8192
	assert block_info[4].checksum == bytes.fromhex("35a0c669ac8c646e70c02bd1ddd90042")
	assert block_info[4].rsum == 0xB5BA7A78
	assert block_info[4].size == 817


@pytest.mark.posix
def test_read_zsync_file(tmp_path: Path) -> None:
	# git config --global core.autocrlf false
	test_file = Path("tests/data/test.small")
	zsync_file = Path("tests/data/test.small.zsync")
	# Original zsync rsum.a rsum.b:
	# 0 32845
	# 0 15789
	# 0 2597
	# 0 58805
	# 0 31352

	digest = hashlib.sha1(test_file.read_bytes()).hexdigest()
	assert digest == "bfb8611ca38c187cea650072898ff4381ed2b465"

	info = read_zsync_file(zsync_file)

	assert info.zsync == "0.6.2"
	assert info.filename == "test.small"
	assert info.url == "test.small"
	assert info.sha1 == bytes.fromhex(digest)
	assert info.mtime == datetime.fromisoformat("2023-05-26T10:30:14+00:00")
	assert info.length == 9009
	assert info.block_size == 2048
	assert info.seq_matches == 2
	assert info.rsum_bytes == 2
	assert info.checksum_bytes == 3

	assert len(info.block_info) == 5

	assert info.block_info[0].block_id == 0
	assert info.block_info[0].offset == 0
	assert info.block_info[0].checksum == bytes.fromhex("56bd0a00000000000000000000000000")
	assert info.block_info[0].rsum == 0x804D

	assert info.block_info[3].block_id == 3
	assert info.block_info[3].offset == 6144
	assert info.block_info[3].checksum == bytes.fromhex("709f5400000000000000000000000000")
	assert info.block_info[3].rsum == 0xE5B5

	assert info.block_info[4].block_id == 4
	assert info.block_info[4].offset == 8192
	assert info.block_info[4].checksum == bytes.fromhex("35a0c600000000000000000000000000")
	assert info.block_info[4].rsum == 0x7A78


def test_read_zsync_file_umlauts() -> None:
	zsync_file = Path("tests/data/äöü.zsync")
	info = read_zsync_file(zsync_file)
	assert info.zsync == "0.6.2"
	assert info.filename == "äöü"
	assert info.url == "äöü"


@pytest.mark.parametrize(
	"rsum_bytes",
	(2, 3, 4),
)
def test_write_zsync_file(tmp_path: Path, rsum_bytes: int) -> None:
	zsync_file = tmp_path / "test.zsync"
	file_info = ZsyncFileInfo(
		zsync="0.6.4",
		producer="pyzsync 1.2.3",
		filename="test",
		url="test",
		sha1=bytes.fromhex("bfb8611ca38c187cea650072898ff4381ed2b465"),
		sha256=bytes.fromhex("db5a54534ed83189736c93a04b3d5805f84651ceaf323fcc0d06dd773559ddfc"),
		mtime=datetime.fromisoformat("2023-05-25T18:37:04+00:00"),
		length=8192,
		block_size=2048,
		seq_matches=2,
		rsum_bytes=rsum_bytes,
		checksum_bytes=3,
		block_info=[
			BlockInfo(block_id=0, offset=0, size=2048, rsum=0x12345678, checksum=bytes.fromhex("11223344556677889900112233445566")),
			BlockInfo(block_id=1, offset=2048, size=2048, rsum=0x123456, checksum=bytes.fromhex("112233445566778899001122334455aa")),
			BlockInfo(block_id=2, offset=4096, size=2048, rsum=0x1234, checksum=bytes.fromhex("112233445566778899001122334455bb")),
			BlockInfo(block_id=3, offset=6144, size=2048, rsum=0x12, checksum=bytes.fromhex("112233445566778899001122334455cc")),
		],
	)
	write_zsync_file(file_info, zsync_file)
	r_file_info = read_zsync_file(zsync_file)

	assert r_file_info.zsync == file_info.zsync
	assert r_file_info.producer == file_info.producer
	assert r_file_info.filename == file_info.filename
	assert r_file_info.url == file_info.url
	assert r_file_info.sha1 == file_info.sha1
	assert r_file_info.sha256 == file_info.sha256
	assert r_file_info.mtime == file_info.mtime
	assert r_file_info.length == file_info.length
	assert r_file_info.block_size == file_info.block_size
	assert r_file_info.seq_matches == file_info.seq_matches
	assert r_file_info.rsum_bytes == file_info.rsum_bytes
	assert r_file_info.checksum_bytes == file_info.checksum_bytes

	assert len(r_file_info.block_info) == 4
	hash_mask = (2 << (rsum_bytes * 8 - 1)) - 1
	for idx, block_info in enumerate(r_file_info.block_info):
		# print(block_info.block_id)
		print(hex(block_info.rsum))
		print(hex((file_info.block_info[idx].rsum & hash_mask)))
		assert block_info.rsum == file_info.block_info[idx].rsum & hash_mask

	shutil.rmtree(tmp_path)


def test_big_zsync_file(tmp_path: Path) -> None:
	test_file = tmp_path / "test.big"

	sha1 = hashlib.new("sha1")
	with open(test_file, "wb") as file:
		data = ""
		for char in range(0x0000, 0x9FFF):
			data += chr(char)
		dbytes = data.encode("utf-8")
		for _ in range(8_000):
			sha1.update(dbytes)
			file.write(dbytes)
	digest = sha1.hexdigest()
	assert digest == "6c1c5f44448f4799298ad7372d6cabcf9c8750fe"

	zsync_file = Path("tests/data/test.big.zsync")

	for _ in range(2):
		info = read_zsync_file(zsync_file)
		assert info.zsync == "0.6.2"
		assert info.filename == "test.big"
		assert info.url == "test.big"
		assert info.sha1 == bytes.fromhex(digest)
		assert info.length == 965608000
		assert info.block_size == 4096
		assert info.seq_matches == 2
		assert info.rsum_bytes == 3
		assert info.checksum_bytes == 5

		assert len(info.block_info) == 235745

		assert info.block_info[0].block_id == 0
		assert info.block_info[0].offset == 0
		assert info.block_info[0].checksum == bytes.fromhex("919f75694e0000000000000000000000")
		assert info.block_info[0].rsum == 0x9DC2FF

		assert info.block_info[10000].checksum == bytes.fromhex("1e84be73a10000000000000000000000")
		assert info.block_info[10000].rsum == 0xF9AEAB

		assert info.block_info[235744].checksum == bytes.fromhex("109328c12e0000000000000000000000")
		assert info.block_info[235744].rsum == 0xDD4EE3

		# Create zsync file and read again
		zsync_file = Path(tmp_path / "test.big.zsync")
		start = time.time()
		create_zsync_file(test_file, zsync_file)
		duration = time.time() - start
		assert duration < 16

	shutil.rmtree(tmp_path)


@pytest.mark.posix
@pytest.mark.parametrize(
	"legacy_mode",
	(False, True),
)
def test_create_zsync_file(tmp_path: Path, legacy_mode: bool) -> None:
	zsync_file = tmp_path / "test.small.zsync"
	test_file = Path("tests/data/test.small")

	sha1_digest = hashlib.sha1(test_file.read_bytes()).hexdigest()
	assert sha1_digest == "bfb8611ca38c187cea650072898ff4381ed2b465"
	sha256_digest = hashlib.sha256(test_file.read_bytes()).hexdigest()
	assert sha256_digest == "9699e10d36d6b54e74fe3174d1beeb3c21eb2aa30928696ad2ad0072ce66f002"

	create_zsync_file(test_file, zsync_file, legacy_mode=legacy_mode)

	info = read_zsync_file(zsync_file)
	assert info.zsync == "0.6.2"
	assert info.producer == "" if legacy_mode else "pyzsync 0.1"
	assert info.filename == "test.small"
	assert info.url == "test.small"
	assert info.sha1 == bytes.fromhex(sha1_digest)
	assert info.sha256 == bytes.fromhex(
		"0000000000000000000000000000000000000000000000000000000000000000" if legacy_mode else sha256_digest
	)
	assert info.mtime == datetime.fromtimestamp(int(test_file.stat().st_mtime), tz=timezone.utc)
	assert info.length == 9009
	assert info.block_size == 2048
	assert info.seq_matches == 2
	assert info.rsum_bytes == 2
	assert info.checksum_bytes == 3

	assert len(info.block_info) == 5

	assert info.block_info[0].block_id == 0
	assert info.block_info[0].offset == 0
	assert info.block_info[0].checksum == bytes.fromhex("56bd0a00000000000000000000000000")
	assert info.block_info[0].rsum == 0x0000804D

	assert info.block_info[3].block_id == 3
	assert info.block_info[3].offset == 6144
	assert info.block_info[3].checksum == bytes.fromhex("709f5400000000000000000000000000")
	assert info.block_info[3].rsum == 0x0000E5B5

	assert info.block_info[4].block_id == 4
	assert info.block_info[4].offset == 8192
	assert info.block_info[4].checksum == bytes.fromhex("35a0c600000000000000000000000000")
	assert info.block_info[4].rsum == 0x00007A78

	shutil.rmtree(tmp_path)


@pytest.mark.parametrize(
	"mode, block_size, rsum_bytes, exp_max, exp_mean",
	(
		("random", 2048, 1, 71, 40),
		("random", 2048, 2, 6, 1.1),
		("random", 2048, 3, 2, 1.001),
		("random", 2048, 4, 2, 1.0002),
		("random", 4096, 1, 68, 40),
		("random", 4096, 2, 6, 1.1),
		("random", 4096, 3, 2, 1.001),
		("random", 4096, 4, 2, 1.00011),
		("repeat", 2048, 1, 256, 256),
		("repeat", 2048, 2, 4, 4),
		("repeat", 2048, 3, 4, 4),
		("repeat", 2048, 4, 4, 4),
		("repeat", 4096, 1, 256, 256),
		("repeat", 4096, 2, 8, 8),
		("repeat", 4096, 3, 8, 8),
		("repeat", 4096, 4, 8, 8),
	),
)
def test_rsum_collisions(tmp_path: Path, mode: str, block_size: int, rsum_bytes: int, exp_max: int, exp_mean: int) -> None:
	data_file = tmp_path / "data"
	with open(data_file, "wb") as file:
		if mode == "random":
			file.write(randbytes(block_size * 10000))
		elif mode == "repeat":
			for block_id in range(0xFF + 1):
				block_data = bytes([block_id]) * block_size
				file.write(block_data)
		else:
			raise ValueError(f"Unknown mode: {mode}")

	block_infos = calc_block_infos(data_file, block_size, rsum_bytes, 16)
	hashes = [b.rsum for b in block_infos]
	counter = Counter(hashes)
	occurences = list(counter.values())
	h_mean = mean(occurences)
	h_max = max(occurences)
	print(block_size, rsum_bytes, "is:", h_max, h_mean, "exp:", exp_max, exp_mean)
	# for hash in counter.keys():
	# 	print(hash.hex())
	assert h_mean <= exp_mean
	assert h_max <= exp_max

	shutil.rmtree(tmp_path)


@pytest.mark.parametrize(
	"mode, block_size, checksum_bytes, exp_max, exp_mean",
	(
		("random", 2048, 3, 2, 1.001),
		("random", 2048, 4, 1, 1.00011),
		("random", 2048, 5, 1, 1),
		("random", 4096, 3, 2, 1.001),
		("random", 4096, 4, 2, 1.00011),
		("random", 4096, 5, 1, 1),
		("repeat", 2048, 3, 1, 1),
		("repeat", 2048, 4, 2, 1.00011),
		("repeat", 2048, 5, 1, 1),
		("repeat", 4096, 3, 1, 1),
		("repeat", 4096, 4, 1, 1),
		("repeat", 4096, 5, 1, 1),
	),
)
def test_checksum_collisions(tmp_path: Path, mode: str, block_size: int, checksum_bytes: int, exp_max: int, exp_mean: int) -> None:
	data_file = tmp_path / "data"
	with open(data_file, "wb") as file:
		if mode == "random":
			file.write(randbytes(block_size * 10000))
		elif mode == "repeat":
			for block_id in range(0xFF + 1):
				block_data = bytes([block_id]) * block_size
				file.write(block_data)
		else:
			raise ValueError(f"Unknown mode: {mode}")

	block_infos = calc_block_infos(data_file, block_size, 4, checksum_bytes)
	hashes = [b.checksum for b in block_infos]
	counter = Counter(hashes)
	occurences = list(counter.values())
	h_mean = mean(occurences)
	h_max = max(occurences)
	print(block_size, checksum_bytes, "is:", h_max, h_mean, "exp:", exp_max, exp_mean)
	assert h_mean <= exp_mean
	assert h_max <= exp_max

	shutil.rmtree(tmp_path)


@pytest.mark.parametrize(
	"chunk_size",
	(100, 1_000, 10_000, 100_000),
)
def test_http_patcher(tmp_path: Path, chunk_size: int) -> None:
	target_file = tmp_path / "local"

	class MockHTTPPatcher(HTTPPatcher):
		data = (
			b"\r\n"
			b"--3d7d9f7d709b\r\n"
			b"Content-Type: text/plain\r\n"
			b"Content-Range: bytes 0-29/110\r\n"
			b"\r\n"
			b"aaaaabbbbbcccccdddddeeeeefffff\r\n"
			b"--3d7d9f7d709b\r\n"
			b"Content-Type: text/plain\r\n"
			b"Content-Range: bytes 100-109/110\r\n"
			b"\r\n"
			b"ggggghhhhh\r\n"
			b"--3d7d9f7d709b--\r\n"
		)
		pos = 0

		def _send_request(self) -> tuple[int, CaseInsensitiveDict]:
			return 206, CaseInsensitiveDict({"Content-Type": "multipart/byteranges; boundary=3d7d9f7d709b"})

		def _read_response_data(self, size: Optional[int] = None) -> bytes:
			size = size or len(self.data) - self.pos
			dat = self.data[self.pos : self.pos + size]
			self.pos += size
			return dat

	instructions = [
		PatchInstruction(source=SOURCE_REMOTE, source_offset=0, target_offset=0, size=10),
		PatchInstruction(source=SOURCE_REMOTE, source_offset=20, target_offset=21, size=9),
		PatchInstruction(source=SOURCE_REMOTE, source_offset=100, target_offset=40, size=10),
	]

	with open(target_file, "wb") as tfh:
		patcher = MockHTTPPatcher(instructions=instructions, target_file=tfh, url="https://localhost:12345/mock")
		patcher.chunk_size = chunk_size
		patcher.run()

	data = target_file.read_bytes()
	assert data == b"aaaaabbbbb\0\0\0\0\0\0\0\0\0\0\0eeeeeffff\0\0\0\0\0\0\0\0\0\0ggggghhhhh"


def test_patch_file_local(tmp_path: Path) -> None:
	remote_file = tmp_path / "remote"
	remote_zsync_file = tmp_path / "remote.zsync"
	local_file1 = tmp_path / "local1"
	local_file2 = tmp_path / "local2"
	local_file3 = tmp_path / "local3"

	file_size = 200_000_000
	block_size = calc_block_size(file_size)
	block_count = int((file_size + block_size - 1) / block_size)
	with (
		open(remote_file, "wb") as rfile,
		open(local_file1, "wb") as lfile1,
		open(local_file2, "wb") as lfile2,
		open(local_file3, "wb") as lfile3,
	):
		cnt = 1
		for block_id in range(block_count):
			data_size = block_size
			if block_id == block_count - 1:
				data_size = file_size - block_id * block_size
				assert data_size <= block_size
			block_data = randbytes(data_size)
			rfile.write(block_data)
			if cnt == 1:
				lfile1.write(block_data + b"\0\0\0\0\0")
			elif cnt == 2:
				lfile2.write(block_data + b"\0\0")
			elif cnt == 3:
				lfile3.write(block_data)
			cnt = cnt + 1 if cnt < 4 else 1

	assert remote_file.stat().st_size == file_size

	# Create zsync file
	create_zsync_file(remote_file, remote_zsync_file, legacy_mode=False)

	# Start sync
	files = [local_file1, local_file2, local_file3]
	zsync_info = read_zsync_file(remote_zsync_file)
	zsync_info.seq_matches = 1
	instructions = get_patch_instructions(zsync_info, files)
	# for inst in instructions:
	# 	print(inst.source, inst.source_offset, inst.size, "=>", inst.target_offset)

	def patcher_factory(instructions: list[PatchInstruction], target_file: BinaryIO) -> Patcher:
		return FilePatcher(instructions=instructions, target_file=target_file, source_file=remote_file)

	output_file = tmp_path / "out"

	sha1 = patch_file(files, instructions, patcher_factory, output_file=output_file, return_hash="sha1", delete_files=False)
	assert zsync_info.sha1 == hashlib.sha1(remote_file.read_bytes()).digest()
	assert remote_file.stat().st_size == output_file.stat().st_size
	# assert remote_file.read_bytes() == output_file.read_bytes()
	assert sha1 == zsync_info.sha1

	sha256 = patch_file(files, instructions, patcher_factory, output_file=output_file, return_hash="sha256", delete_files=True)
	assert zsync_info.sha256 == hashlib.sha256(remote_file.read_bytes()).digest()
	assert remote_file.stat().st_size == output_file.stat().st_size
	# assert remote_file.read_bytes() == output_file.read_bytes()
	assert sha256 == zsync_info.sha256

	local_bytes = sum(i.size for i in instructions if i.source != SOURCE_REMOTE)
	speedup = local_bytes * 100 / zsync_info.length
	print(f"Speedup: {speedup} %")
	assert round(speedup) == 75

	shutil.rmtree(tmp_path)


def test_patch_file_http(tmp_path: Path) -> None:
	remote_file = tmp_path / "remote"
	remote_zsync_file = tmp_path / "remote.zsync"
	local_file = tmp_path / "local"
	local_file_bak = tmp_path / "local.bak"

	block_count = 10
	block_size = 2048
	with open(remote_file, "wb") as rfile, open(local_file, "wb") as lfile:
		for block_id in range(block_count):
			block_data = randbytes(int(block_size / 2) if block_id == block_count - 1 else block_size)
			rfile.write(block_data)
			if block_id in (1, 2, 3, 7):
				lfile.write(block_data)
	shutil.copy(local_file, local_file_bak)

	create_zsync_file(remote_file, remote_zsync_file, legacy_mode=False)
	zsync_info = read_zsync_file(remote_zsync_file)
	zsync_info.seq_matches = 1

	instructions = get_patch_instructions(zsync_info, local_file, optimized=True)
	for inst in instructions:
		print(inst.source, inst.source_offset, inst.size, "=>", inst.target_offset)

	# R:0
	assert len(instructions) == 1
	assert instructions[0].source == SOURCE_REMOTE
	assert instructions[0].source_offset == 0
	assert instructions[0].target_offset == 0
	assert instructions[0].size == 19456

	instructions = get_patch_instructions(zsync_info, local_file)
	for inst in instructions:
		print(inst.source, inst.source_offset, inst.size, "=>", inst.target_offset)

	# R:0, L:1, L:2, L:3, R:4-6, L:7, R:8-9
	assert len(instructions) == 7
	assert instructions[0].source == SOURCE_REMOTE
	assert instructions[1].source != SOURCE_REMOTE
	assert instructions[2].source != SOURCE_REMOTE
	assert instructions[3].source != SOURCE_REMOTE
	assert instructions[4].source == SOURCE_REMOTE
	assert instructions[5].source != SOURCE_REMOTE
	assert instructions[6].source == SOURCE_REMOTE

	remote_bytes = sum(i.size for i in instructions if i.source == SOURCE_REMOTE)

	class MyProgressListener(ProgressListener):
		abort_position = -1

		def __init__(self) -> None:
			self.position = 0
			self.total = 0

		def progress_changed(self, patcher: Patcher, position: int, total: int, per_second: int) -> None:
			self.position = position
			self.total = total
			print(f"{self.position}/{self.total} ({per_second/1000:.2f} kB/s)")
			if self.abort_position > 0 and self.position >= self.abort_position:
				patcher.abort()

	with http_server(tmp_path) as port:
		progress_listener = MyProgressListener()
		http_patcher: Optional[HTTPPatcher] = None

		def patcher_factory(instructions: list[PatchInstruction], target_file: BinaryIO) -> Patcher:
			nonlocal http_patcher
			http_patcher = HTTPPatcher(
				instructions=instructions, target_file=target_file, url=f"http://localhost:{port}/remote", max_ranges_per_request=1
			)
			http_patcher.register_progress_listener(progress_listener)
			return http_patcher

		sha256 = patch_file(local_file, instructions, patcher_factory, return_hash="sha256")
		assert progress_listener.total == remote_bytes
		assert progress_listener.position == progress_listener.total

		assert isinstance(http_patcher, HTTPPatcher)
		http_patcher = cast(HTTPPatcher, http_patcher)
		http_patcher.unregister_progress_listener(progress_listener)

		assert remote_file.stat().st_size == local_file.stat().st_size
		assert sha256 == zsync_info.sha256
		assert zsync_info.sha256 == hashlib.sha256(remote_file.read_bytes()).digest()
		# assert remote_file.read_bytes() == local_file.read_bytes()

	speedup = (zsync_info.length - remote_bytes) * 100 / zsync_info.length
	print(f"Speedup: {speedup} %")
	assert round(speedup) == 42

	# Restore the original local_file and test abort
	shutil.copy(local_file_bak, local_file)

	with http_server(tmp_path) as port:
		progress_listener = MyProgressListener()
		progress_listener.abort_position = int(block_size / block_count / 2)
		http_patcher = None

		def patcher_factory(instructions: list[PatchInstruction], target_file: BinaryIO) -> Patcher:
			nonlocal http_patcher
			http_patcher = HTTPPatcher(
				instructions=instructions, target_file=target_file, url=f"http://localhost:{port}/remote", max_ranges_per_request=1
			)
			http_patcher.register_progress_listener(progress_listener)
			return http_patcher

		with pytest.raises(RuntimeError, match="Aborted"):
			patch_file(local_file, instructions, patcher_factory, return_hash="sha256")

	shutil.rmtree(tmp_path)


def test_optimize_instructions() -> None:
	instructions = [
		PatchInstruction(source=SOURCE_REMOTE, source_offset=0, target_offset=0, size=2048),
		PatchInstruction(source=SOURCE_REMOTE, source_offset=2048, target_offset=2048, size=2048),
		PatchInstruction(source=SOURCE_REMOTE, source_offset=4096, target_offset=4096, size=2048),
	]
	assert optimize_instructions(instructions, min_remote_gap=16384) == [
		PatchInstruction(source=SOURCE_REMOTE, source_offset=0, target_offset=0, size=6144),
	]

	instructions = [
		PatchInstruction(source=SOURCE_REMOTE, source_offset=0, target_offset=0, size=2048),
		PatchInstruction(source=0, source_offset=2048, target_offset=2048, size=2048),
		PatchInstruction(source=SOURCE_REMOTE, source_offset=4096, target_offset=4096, size=2048),
	]
	assert optimize_instructions(instructions, min_remote_gap=16384) == [
		PatchInstruction(source=SOURCE_REMOTE, source_offset=0, target_offset=0, size=6144),
	]

	instructions = [
		PatchInstruction(source=SOURCE_REMOTE, source_offset=0, target_offset=0, size=2048),
		PatchInstruction(source=0, source_offset=2048, target_offset=2048, size=2048),
		PatchInstruction(source=0, source_offset=4096, target_offset=4096, size=4096),
		PatchInstruction(source=SOURCE_REMOTE, source_offset=8192, target_offset=8192, size=1000),
	]
	assert optimize_instructions(instructions, min_remote_gap=16384) == [
		PatchInstruction(source=SOURCE_REMOTE, source_offset=0, target_offset=0, size=9192),
	]
	assert optimize_instructions(instructions, min_remote_gap=2048) == instructions
	assert optimize_instructions(instructions, min_remote_gap=1) == instructions


@pytest.mark.targz_available
def test_patch_tar(tmp_path: Path) -> None:
	remote_file = tmp_path / "remote"
	remote_zsync_file = tmp_path / "remote.zsync"
	local_file = tmp_path / "local"
	file_size = 100_000

	src = tmp_path / "src"
	src.mkdir()
	for num in range(1, 100):
		file = src / f"file{num}"
		file.write_bytes(randbytes(file_size))

	run(f"tar c src | gzip --rsyncable > {local_file}", shell=True, cwd=tmp_path)

	for num in range(101, 110):
		file = src / f"file{num}"
		file.write_bytes(randbytes(file_size))

	for num in range(1, 10):
		file = src / f"file{num}"
		file.unlink()

	run(f"tar c src | gzip --rsyncable > {remote_file}", shell=True, cwd=tmp_path)

	print(remote_file.stat().st_size)
	print(local_file.stat().st_size)

	# Create zsync file
	create_zsync_file(remote_file, remote_zsync_file)

	# Start sync
	zsync_info = read_zsync_file(remote_zsync_file)
	instructions = get_patch_instructions(zsync_info, local_file)
	instructions = sorted(instructions, key=lambda r: r.target_offset)
	# for inst in instructions:
	# 	print(inst.source, inst.source_offset, inst.size, "=>", inst.target_offset)

	def patcher_factory(instructions: list[PatchInstruction], target_file: BinaryIO) -> Patcher:
		return FilePatcher(instructions=instructions, target_file=target_file, source_file=remote_file)

	sha1 = patch_file(files=local_file, instructions=instructions, patcher_factory=patcher_factory)

	assert remote_file.stat().st_size == local_file.stat().st_size
	# assert remote_file.read_bytes() == local_file.read_bytes()
	assert sha1 == zsync_info.sha1

	local_bytes = sum(i.size for i in instructions if i.source != SOURCE_REMOTE)
	speedup = local_bytes * 100 / zsync_info.length
	print(f"Speedup: {speedup} %")
	assert round(speedup) >= 86

	shutil.rmtree(tmp_path)


@pytest.mark.parametrize(
	"file_size",
	(1_000_000, 100_000_000, 1_000_000_000),
)
@pytest.mark.zsyncmake_available
def test_original_zsyncmake_compatibility(tmp_path: Path, file_size: int) -> None:
	# Test different file sizes which result in different block_sizes, rsum_bytes and checksum_bytes
	remote_file = tmp_path / "remote"
	remote_zsync_file = tmp_path / "remote.zsync"
	local_file = tmp_path / "local"

	block_size = calc_block_size(file_size)
	block_count = int((file_size + block_size - 1) / block_size)
	with open(remote_file, "wb") as rfile, open(local_file, "wb") as lfile:
		for block_id in range(block_count):
			data_size = block_size
			if block_id == block_count - 1:
				data_size = file_size - block_id * block_size
				assert data_size <= block_size
			block_data = randbytes(data_size)
			rfile.write(block_data)
			if block_id % 10 == 0:
				lfile.write(b"\0\0\0")
			else:
				lfile.write(block_data)

	cmd = ["zsyncmake", "-Z", "-o", str(remote_zsync_file.name), str(remote_file)]
	run(cmd, cwd=tmp_path)

	zsync_info = read_zsync_file(remote_zsync_file)
	assert zsync_info.sha1 == hashlib.sha1(remote_file.read_bytes()).digest()

	block_infos = calc_block_infos(remote_file, zsync_info.block_size, zsync_info.rsum_bytes, zsync_info.checksum_bytes)
	assert len(zsync_info.block_info) == len(block_infos)
	for idx, block_info in enumerate(zsync_info.block_info):
		assert block_info.rsum == block_infos[idx].rsum
		assert block_info.checksum == block_infos[idx].checksum

	instructions = get_patch_instructions(zsync_info, local_file)

	def patcher_factory(instructions: list[PatchInstruction], target_file: BinaryIO) -> Patcher:
		return FilePatcher(instructions=instructions, target_file=target_file, source_file=remote_file)

	sha1 = patch_file(local_file, instructions, patcher_factory)

	assert sha1 == zsync_info.sha1

	local_bytes = sum(i.size for i in instructions if i.source != SOURCE_REMOTE)
	speedup = local_bytes * 100 / zsync_info.length
	print(f"Speedup: {speedup} %")
	assert round(speedup) == 80

	shutil.rmtree(tmp_path)


@pytest.mark.zsync_available
@pytest.mark.parametrize(
	"file_size",
	(2_000_000, 200_000_000),
)
def test_original_zsync_compatibility(tmp_path: Path, file_size: int) -> None:
	remote_dir = tmp_path / "remote"
	local_dir = tmp_path / "local"
	remote_dir.mkdir()
	local_dir.mkdir()

	remote_file = remote_dir / "testfile"
	remote_zsync_file = remote_dir / "testfile.zsync"
	local_file = local_dir / "testfile"

	block_size = calc_block_size(file_size)
	remote_blocks = int(file_size / block_size / 2) * 2
	local_blocks = int(remote_blocks / 2)
	local_bytes = local_blocks * block_size
	with open(remote_file, "wb") as rfile, open(local_file, "wb") as lfile:
		for _ in range(remote_blocks - local_blocks):
			data = randbytes(block_size)
			rfile.write(data)
			lfile.write(data)
		for _ in range(local_blocks):
			data = randbytes(block_size)
			rfile.write(data)

	create_zsync_file(remote_file, remote_zsync_file, legacy_mode=True)

	with http_server(remote_dir) as port:
		cmd = ["zsync", "-i", str(local_file), f"http://localhost:{port}/{remote_zsync_file.name}"]
		out = run(cmd, cwd=local_dir, check=True, timeout=30, capture_output=True, encoding="utf-8").stdout
		if out:
			print(out)
			assert f"used {local_bytes} local" in out

	assert hashlib.sha1(local_file.read_bytes()).digest() == hashlib.sha1(remote_file.read_bytes()).digest()

	shutil.rmtree(tmp_path)


def test_create_zsync_file_multi_thread_no_crash(tmp_path: Path) -> None:
	remote_file = tmp_path / "remotefile"
	remote_zsync_file = tmp_path / "remotefile.zsync"

	with open(remote_file, "wb") as rfile:
		for _ in range(1_000):
			rfile.write(randbytes(10_000))

	def task() -> None:
		time.sleep(0.1)
		create_zsync_file(remote_file, remote_zsync_file, legacy_mode=False)

	threads = []
	for _ in range(25):
		thread = Thread(target=task)
		thread.daemon = True
		threads.append(thread)

	for thread in threads:
		thread.start()

	for thread in threads:
		thread.join(10)


def test_errors(tmp_path: Path) -> None:
	# Windows error messages are localized
	is_windows = platform.system().lower() == "windows"

	some_file = tmp_path / "some_file"
	some_file.write_bytes(b"data")
	some_zsync_file = tmp_path / "some_file.zsync"
	create_zsync_file(some_file, some_zsync_file)
	some_zsync_info = read_zsync_file(some_zsync_file)

	with pytest.raises(ValueError, match="num_bytes out of range"):
		md4(block=b"data", num_bytes=18)

	with pytest.raises(ValueError, match="num_bytes out of range"):
		rsum(block=b"data", num_bytes=5)

	with pytest.raises(ValueError, match="Invalid block_size"):
		calc_block_infos(file=some_file, block_size=1)

	with pytest.raises(ValueError, match="rsum_bytes out of range"):
		calc_block_infos(file=some_file, block_size=2048, rsum_bytes=10)

	with pytest.raises(ValueError, match="checksum_bytes out of range"):
		calc_block_infos(file=some_file, block_size=2048, checksum_bytes=33)

	with pytest.raises(FileNotFoundError, match="" if is_windows else "No such file or directory"):
		calc_block_infos(file=Path("nonexistent"), block_size=2048)

	with pytest.raises(FileNotFoundError, match="" if is_windows else "No such file or directory"):
		create_zsync_info(Path("nonexistent"))

	with pytest.raises(FileNotFoundError, match="" if is_windows else "No such file or directory"):
		create_zsync_file(Path("nonexistent"), tmp_path / "some.zsync")

	with pytest.raises(FileNotFoundError, match="" if is_windows else "No such file or directory"):
		write_zsync_file(zsync_info=some_zsync_info, zsync_file=Path("/tmp/no/such/path"))

	zsync_file = tmp_path / "fail.zsync"

	zsync_file.write_text("SHA-1: 123\n", encoding="utf-8")
	with pytest.raises(ValueError, match="Invalid SHA-1 value"):
		read_zsync_file(zsync_file=zsync_file)

	zsync_file.write_text("SHA-256: 123\n", encoding="utf-8")
	with pytest.raises(ValueError, match="Invalid SHA-256 value"):
		read_zsync_file(zsync_file=zsync_file)

	zsync_file.write_text("SHA-256: 00112233\n", encoding="utf-8")
	with pytest.raises(ValueError, match="Invalid SHA-256 value"):
		read_zsync_file(zsync_file=zsync_file)

	zsync_file.write_text("MTime: 123\n", encoding="utf-8")
	with pytest.raises(ValueError, match="Invalid MTime value"):
		read_zsync_file(zsync_file=zsync_file)

	zsync_file.write_text("Blocksize: fail\n", encoding="utf-8")
	with pytest.raises(ValueError, match="invalid digit found"):
		read_zsync_file(zsync_file=zsync_file)

	zsync_file.write_text("Length: fail\n", encoding="utf-8")
	with pytest.raises(ValueError, match="invalid digit found"):
		read_zsync_file(zsync_file=zsync_file)

	zsync_file.write_text("Hash-Lengths: fail\n", encoding="utf-8")
	with pytest.raises(ValueError, match="Invalid Hash-Lengths value"):
		read_zsync_file(zsync_file=zsync_file)

	zsync_file.write_text("Hash-Lengths: 1,2,3,4\n", encoding="utf-8")
	with pytest.raises(ValueError, match="Invalid Hash-Lengths value"):
		read_zsync_file(zsync_file=zsync_file)

	zsync_file.write_text("Hash-Lengths: 9,2,2\n", encoding="utf-8")
	with pytest.raises(ValueError, match="seq_matches out of range"):
		read_zsync_file(zsync_file=zsync_file)

	zsync_file.write_text("Hash-Lengths: 2,10,2\n", encoding="utf-8")
	with pytest.raises(ValueError, match="rsum_bytes out of range"):
		read_zsync_file(zsync_file=zsync_file)

	zsync_file.write_text("Hash-Lengths: 2,2,20\n", encoding="utf-8")
	with pytest.raises(ValueError, match="checksum_bytes out of range"):
		read_zsync_file(zsync_file=zsync_file)
