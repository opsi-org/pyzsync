# Copyright (c) 2023 uib GmbH <info@uib.de>
# This code is owned by the uib GmbH, Mainz, Germany (uib.de). All rights reserved.
# License: AGPL-3.0

import hashlib
import shutil
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from random import randbytes
from statistics import mean
from subprocess import run

import pytest

from pyzsync import (
	BlockInfo,
	Source,
	ZsyncFileInfo,
	calc_block_infos,
	calc_block_size,
	create_zsync_file,
	get_patch_instructions,
	md4,
	patch_file,
	read_zsync_file,
	rsum,
	update_rsum,
	write_zsync_file,
)


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


def test_update_rsum() -> None:
	block = ""
	for char in range(2, 1089):
		block += chr(char)
	bblock = block.encode("utf-8")
	assert len(bblock) == 2048
	_rsum = rsum(bblock, 4)
	assert hex(_rsum) == "0x68f0b901"
	for _ in range(2048):
		old_char = bblock[0]
		bblock = bblock[1:] + b"\03"
		new_char = bblock[-1]
		new_rsum = rsum(bblock, 4)
		_rsum = update_rsum(_rsum, old_char, new_char)
		# print(hex(rsum))
		assert _rsum == new_rsum


def test_calc_block_size() -> None:
	assert calc_block_size(1) == 2048
	assert calc_block_size(1_000_000_000) == 4096
	assert calc_block_size(2_000_000_000) == 4096


def test_hash_speed(tmp_path: Path):
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
	assert rsum_time < 3
	assert md4_time < 15

	shutil.rmtree(tmp_path)


def test_calc_block_infos(tmp_path: Path) -> None:
	# Ensure correct line endings (git windows)
	data = Path("tests/data/test.small").read_bytes().replace(b"\r\n", b"\n")
	test_file = tmp_path / "test.small"
	test_file.write_bytes(data)

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


def test_read_zsync_file(tmp_path: Path) -> None:
	# Ensure correct line endings (git windows)
	data = Path("tests/data/test.small").read_bytes().replace(b"\r\n", b"\n")
	test_file = tmp_path / "test.small"
	test_file.write_bytes(data)
	zsync_file = Path("tests/data/test.small.zsync")

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

	# zsyncmake test.big  2,96s user 0,24s system 99% cpu 3,206 total
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
		assert duration < 15

	shutil.rmtree(tmp_path)


def test_create_zsync_file(tmp_path: Path) -> None:
	zsync_file = tmp_path / "test.small.zsync"
	test_file = Path("tests/data/test.small")

	digest = hashlib.sha1(test_file.read_bytes()).hexdigest()
	assert digest == "bfb8611ca38c187cea650072898ff4381ed2b465"

	create_zsync_file(test_file, zsync_file)

	info = read_zsync_file(zsync_file)
	assert info.zsync == "0.6.2"
	assert info.producer == "pyzsync 0.1"
	assert info.filename == "test.small"
	assert info.url == "test.small"
	assert info.sha1 == bytes.fromhex(digest)
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
		("random", 2048, 1, 66, 40),
		("random", 2048, 2, 5, 1.1),
		("random", 2048, 3, 2, 1.001),
		("random", 2048, 4, 2, 1.00011),
		("random", 4096, 1, 66, 40),
		("random", 4096, 2, 5, 1.1),
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
		("random", 2048, 4, 1, 1),
		("random", 2048, 5, 1, 1),
		("random", 4096, 3, 2, 1.001),
		("random", 4096, 4, 1, 1),
		("random", 4096, 5, 1, 1),
		("repeat", 2048, 3, 1, 1),
		("repeat", 2048, 4, 1, 1),
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


def test_patch_file_local(tmp_path: Path):
	remote_file = tmp_path / "remote"
	remote_zsync_file = tmp_path / "remote.zsync"
	local_file = tmp_path / "local"

	file_size = 100_000_000
	# file_size = 2048 * 1 + 1
	block_size = calc_block_size(file_size)
	block_count = int((file_size + block_size - 1) / block_size)
	with (open(remote_file, "wb") as rfile, open(local_file, "wb") as lfile):
		for block_id in range(block_count):
			data_size = block_size
			if block_id == block_count - 1:
				data_size = file_size - block_id * block_size
				assert data_size <= block_size
			block_data = randbytes(data_size)
			rfile.write(block_data)
			if block_id % 5 == 0:
				continue
			elif block_id % 10 == 0:
				lfile.write(block_data + b"\0\0\0\0\0")
			else:
				lfile.write(block_data)

	assert remote_file.stat().st_size == file_size

	# Create zsync file
	create_zsync_file(remote_file, remote_zsync_file)

	# Start sync
	zsync_info = read_zsync_file(remote_zsync_file)
	instructions = get_patch_instructions(zsync_info, local_file)
	# for inst in instructions:
	# 	print(inst.source, inst.source_offset, inst.size, "=>", inst.target_offset)

	def fetch_function(offset: int, size: int) -> bytes:
		with open(remote_file, "rb") as rfile:
			rfile.seek(offset)
			return rfile.read(size)

	output_file = tmp_path / "out"

	sha1 = patch_file(local_file, instructions, fetch_function, output_file=output_file, return_hash="sha1")
	assert zsync_info.sha1 == hashlib.sha1(remote_file.read_bytes()).digest()
	assert remote_file.read_bytes() == output_file.read_bytes()
	assert sha1 == zsync_info.sha1

	sha256 = patch_file(local_file, instructions, fetch_function, output_file=output_file, return_hash="sha256")
	assert zsync_info.sha256 == hashlib.sha256(remote_file.read_bytes()).digest()
	assert remote_file.read_bytes() == output_file.read_bytes()
	assert sha256 == zsync_info.sha256

	local_bytes = sum([i.size for i in instructions if i.source == Source.Local])
	speedup = local_bytes * 100 / zsync_info.length
	print(f"Speedup: {speedup}%")
	assert round(speedup) == 80

	shutil.rmtree(tmp_path)


import socket
from contextlib import closing, contextmanager
from http.client import HTTPConnection
from socketserver import TCPServer
from threading import Thread

from RangeHTTPServer import RangeRequestHandler


@contextmanager
def http_server(directory: Path):
	# Select free port
	with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
		sock.bind(("", 0))
		sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
		port = sock.getsockname()[1]

	class Handler(RangeRequestHandler):
		def __init__(self, *args, **kwargs):
			super().__init__(*args, directory=str(directory), **kwargs)

	server = TCPServer(("", port), Handler)
	thread = Thread(target=server.serve_forever)
	thread.daemon = True
	thread.start()
	try:
		yield port
	finally:
		server.socket.close()
		thread.join(3)


def test_patch_file_http(tmp_path: Path):
	remote_file = tmp_path / "remote"
	remote_zsync_file = tmp_path / "remote.zsync"
	local_file = tmp_path / "local"

	block_count = 10
	block_size = 2048
	with (open(remote_file, "wb") as rfile, open(local_file, "wb") as lfile):
		for block_id in range(block_count):
			block_data = randbytes(int(block_size / 2) if block_id == block_count - 1 else block_size)
			rfile.write(block_data)
			if block_id in (1, 2, 3, 7):
				lfile.write(block_data)

	create_zsync_file(remote_file, remote_zsync_file)
	zsync_info = read_zsync_file(remote_zsync_file)
	instructions = get_patch_instructions(zsync_info, local_file)
	# R:0, L:1, L:2, L:3, R:4-6, L:7, R:8-9
	assert len(instructions) == 7
	assert instructions[0].source == Source.Remote
	assert instructions[1].source == Source.Local
	assert instructions[2].source == Source.Local
	assert instructions[3].source == Source.Local
	assert instructions[4].source == Source.Remote
	assert instructions[5].source == Source.Local
	assert instructions[6].source == Source.Remote

	with http_server(tmp_path) as port:
		conn = HTTPConnection("localhost", port)

		def fetch_function(offset: int, size: int) -> bytes:
			conn.request("GET", "/remote", headers={"Range": f"bytes={offset}-{offset + size - 1}"})
			response = conn.getresponse()
			# print(response.status, response.reason)
			return response.read()

		sha256 = patch_file(local_file, instructions, fetch_function, return_hash="sha256")
		assert zsync_info.sha256 == hashlib.sha256(remote_file.read_bytes()).digest()
		assert remote_file.read_bytes() == local_file.read_bytes()
		assert sha256 == zsync_info.sha256

	local_bytes = sum([i.size for i in instructions if i.source == Source.Local])
	speedup = local_bytes * 100 / zsync_info.length
	print(f"Speedup: {speedup}%")
	assert round(speedup) == 42

	shutil.rmtree(tmp_path)


@pytest.mark.targz_available
def test_patch_tar(tmp_path: Path):
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
	# for inst in instructions:
	# 	print(inst.source, inst.source_offset, inst.size, "=>", inst.target_offset)

	def fetch_function(offset: int, size: int) -> bytes:
		with open(remote_file, "rb") as rfile:
			rfile.seek(offset)
			return rfile.read(size)

	sha1 = patch_file(local_file, instructions, fetch_function)
	assert sha1 == zsync_info.sha1

	local_bytes = sum([i.size for i in instructions if i.source == Source.Local])
	speedup = local_bytes * 100 / zsync_info.length
	print(f"Speedup: {speedup}%")
	assert round(speedup) >= 87

	shutil.rmtree(tmp_path)


def test_get_instructions(tmp_path: Path) -> None:
	# TODO
	remote_file = tmp_path / "remote"
	remote_zsync_file = tmp_path / "remote.zsync"
	local_file = tmp_path / "local"

	# file_size = 200_000_000
	file_size = 100_000
	block_size = calc_block_size(file_size)
	block_count = int((file_size + block_size - 1) / block_size)
	with (open(remote_file, "wb") as rfile, open(local_file, "wb") as lfile):
		for block_id in range(block_count):
			data_size = block_size
			if block_id == block_count - 1:
				data_size = file_size - block_id * block_size
				assert data_size <= block_size
			block_data = randbytes(data_size)
			rfile.write(block_data)
			if block_id % 2 == 0:
				lfile.write(block_data)
			else:
				lfile.write(b"\0\0\0")
			# elif block_id % 2 == 0:
			# 	lfile.write(block_data[:-4] + b"\0\0\0\0")
			# else:
			# 	lfile.write(block_data)

	assert remote_file.stat().st_size == file_size

	create_zsync_file(remote_file, remote_zsync_file)
	zsync_info = read_zsync_file(remote_zsync_file)

	for inst in get_patch_instructions(zsync_info, local_file):
		print(inst.source, inst.source == Source.Remote)

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
	with (open(remote_file, "wb") as rfile, open(local_file, "wb") as lfile):
		for block_id in range(block_count):
			data_size = block_size
			if block_id == block_count - 1:
				data_size = file_size - block_id * block_size
				assert data_size <= block_size
			block_data = randbytes(data_size)
			rfile.write(block_data)
			# lfile.write(block_data)
			# continue
			if block_id % 4 == 0:
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
	# for inst in instructions:
	# 	print(inst.source, inst.source_offset, inst.size, "=>", inst.target_offset)

	def fetch_function(offset: int, size: int) -> bytes:
		with open(remote_file, "rb") as rfile:
			rfile.seek(offset)
			return rfile.read(size)

	sha1 = patch_file(local_file, instructions, fetch_function)

	# print(local_file.read_bytes().hex())
	# print(remote_file.read_bytes().hex())

	assert sha1 == zsync_info.sha1

	local_bytes = sum([i.size for i in instructions if i.source == Source.Local])
	speedup = local_bytes * 100 / zsync_info.length
	print(f"Speedup: {speedup}%")
	assert round(speedup) == 75

	shutil.rmtree(tmp_path)
