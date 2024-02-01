# Copyright (c) 2023 uib GmbH <info@uib.de>
# This code is owned by the uib GmbH, Mainz, Germany (uib.de). All rights reserved.
# License: AGPL-3.0

import argparse
import logging
import random
import sys
from base64 import b64encode
from http.client import HTTPConnection, HTTPSConnection
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import BinaryIO, Optional
from urllib.parse import urlparse

from pyzsync import (
	SOURCE_REMOTE,
	FilePatcher,
	HTTPPatcher,
	Patcher,
	PatchInstruction,
	ProgressListener,
	create_zsync_file,
	create_zsync_info,
	get_patch_instructions,
	patch_file,
	read_zsync_file,
)


def zsyncmake(file_path: Path) -> None:
	create_zsync_file(file=file_path, zsync_file=file_path.with_name(f"{file_path.name}.zsync"))


def zsync(url: str, *, files: Optional[list[Path]] = None, username: Optional[str] = None, password: Optional[str] = None) -> None:
	if not url.endswith(".zsync"):
		print("Warning: URL does not end with .zsync")
	if not (url.startswith("http://") or url.startswith("https://") or url.startswith("file://")):
		url = f"file://{Path(url).absolute()}"
	url_obj = urlparse(url)

	if url_obj.scheme != "file":
		username = username or ""
		password = password or ""

		conn_class = HTTPConnection if url_obj.scheme == "http" else HTTPSConnection
		connection = conn_class(url_obj.netloc, timeout=600, blocksize=65536)

		headers = {}
		if username or password:
			auth = b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
			headers["Authorization"] = f"Basic {auth}"

		print(f"Fetching zsync file {url_obj.geturl()}...")

		connection.request("GET", url_obj.path, headers=headers)
		response = connection.getresponse()
		if response.status != 200:
			raise RuntimeError(
				f"Failed to fetch {url_obj.geturl()}: {response.status} - {response.read().decode('utf-8', 'ignore').strip()}"
			)
		total_size = int(response.headers["Content-Length"])
		position = 0
		with TemporaryDirectory() as temp_dir:
			zsync_file = Path(temp_dir) / url_obj.path.split("/")[-1]
			with open(zsync_file, "wb") as file:
				last_completed = 0.0
				while position < total_size:
					data = response.read(65536)
					if not data:
						raise RuntimeError(f"Failed to fetch {url_obj.geturl()}: read only {position/total_size} bytes")
					file.write(data)
					position += len(data)
					completed = round(position * 100 / total_size, 1)
					if completed != last_completed:
						print(f"\r{completed:0.1f} %", end="")
						last_completed = completed
				print("")
			zsync_info = read_zsync_file(zsync_file)
	else:
		zsync_info = read_zsync_file(Path(url_obj.path))

	local_files = [Path(Path(zsync_info.filename).name).absolute()]
	local_files.extend(local_files[0].parent.glob(f"{local_files[0].name}.zsync-tmp-*"))
	if files:
		local_files.extend(files)

	print(f"Analyzing {len(local_files)} local file{'s' if len(local_files) > 1 else ''}...")
	instructions = get_patch_instructions(zsync_info, local_files)
	remote_bytes = sum(i.size for i in instructions if i.source == SOURCE_REMOTE)
	ratio = remote_bytes * 100 / zsync_info.length
	print(f"Need to fetch {remote_bytes} bytes ({ratio:.2f} %)")

	path = url_obj.path.split("/")
	path[-1] = zsync_info.url.split("/")[-1]
	rurl = f"{url_obj.scheme}://{url_obj.netloc}{'/'.join(path)}"
	if ratio != 0:
		print(f"Fetching {remote_bytes} bytes from {rurl}...")

	class PrintingProgressListener(ProgressListener):
		def __init__(self) -> None:
			self.last_completed = 0.0

		def progress_changed(self, patcher: Patcher, position: int, total: int, per_second: int) -> None:
			completed = round(position * 100 / total, 1)
			if completed == self.last_completed:
				return
			print(
				f"\r::: {completed:0.1f} % ::: {position/1_000_000:.2f}/{total/1_000_000:.2f} MB ::: {per_second/1_000:.0f} kB/s :::",
				end="",
			)
			self.last_completed = completed

	def patcher_factory(instructions: list[PatchInstruction], target_file: BinaryIO) -> Patcher:
		patcher: Patcher
		if url_obj.scheme == "file":
			patcher = FilePatcher(instructions, target_file, Path(rurl.split("://", 1)[1]))
		else:
			patcher = HTTPPatcher(instructions, target_file, rurl, headers=headers)
		patcher.register_progress_listener(PrintingProgressListener())
		return patcher

	sha1 = patch_file(local_files, instructions, patcher_factory=patcher_factory, return_hash="sha1")
	if ratio != 0:
		print("")
	if sha1 != zsync_info.sha1:
		raise RuntimeError(f"SHA1 mismatch: {sha1.hex()} != {zsync_info.sha1.hex()}")

	print(f"Successfully created {local_files[0]} (reduction: {100-ratio:.2f} %)")


def compare(file1: Path, file2: Path) -> None:
	zsync_info = create_zsync_info(file1)
	instructions = get_patch_instructions(zsync_info, file2)
	file2_bytes = sum(i.size for i in instructions if i.source != SOURCE_REMOTE)
	ratio = file2_bytes * 100 / zsync_info.length
	print(f"{file2} contains {ratio:.2f}% of data to create {file1}")


def destroy(file: Path) -> None:
	tmp_file = file.with_name(f"{file.name}.zsync-destroy")
	chunk_size = 100_000
	file_size = file.stat().st_size
	if file_size < chunk_size:
		chunk_size = int(file_size / 8)
	with open(file, "rb") as src, open(tmp_file, "wb") as dst:
		while data := src.read(chunk_size):
			rand = random.randint(1, 7)
			if rand == 1:
				dst.write(random.randbytes(len(data)))
			elif rand == 2:
				dst.write(random.randbytes(int(len(data) / 3)))
			elif rand in (3, 4):
				continue
			else:
				dst.write(data)

	file.unlink()
	tmp_file.rename(file)
	print(f"File {file} destroyed")


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("--log-level", choices=["debug", "info", "warning", "error", "critical"], default="warning")

	subparsers = parser.add_subparsers(dest="command")

	p_zsyncmake = subparsers.add_parser("zsyncmake", help="Create zsync file from FILE")
	p_zsyncmake.add_argument("file", help="Path to the file")

	p_zsync = subparsers.add_parser("zsync", help="Fetch file from ZSYNC_URL")
	p_zsync.add_argument("zsync_url", help="URL to the zsync file")
	p_zsync.add_argument("--username", help="HTTP basic auth username")
	p_zsync.add_argument("--password", help="HTTP basic auth password")
	p_zsync.add_argument("--files", nargs="+", metavar="FILE", default=[], help="Additional local files to scan for usable blocks")

	p_compare = subparsers.add_parser("compare", help="Compare two files")
	p_compare.add_argument("file", help="Path to the file", nargs=2)

	p_destroy = subparsers.add_parser("destroy", help="Destroy a file (overwrite and remove parts of the file)")
	p_destroy.add_argument("file", help="Path to the file")

	args = parser.parse_args()

	logging.basicConfig(format="[%(levelno)d] [%(asctime)s.%(msecs)03d] %(message)s   (%(filename)s:%(lineno)d)")
	logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))

	if args.command == "zsyncmake":
		return zsyncmake(Path(args.file))

	if args.command == "zsync":
		return zsync(args.zsync_url, files=[Path(f) for f in args.files], username=args.username, password=args.password)

	if args.command == "compare":
		return compare(Path(args.file[0]), Path(args.file[1]))

	if args.command == "destroy":
		return destroy(Path(args.file))

	parser.print_help()


if __name__ == "__main__":
	try:
		main()
	except SystemExit as err:
		sys.exit(err.code)
	except BaseException as err:
		print(err, file=sys.stderr)
		sys.exit(1)
	sys.exit(0)
