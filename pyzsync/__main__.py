# Copyright (c) 2023 uib GmbH <info@uib.de>
# This code is owned by the uib GmbH, Mainz, Germany (uib.de). All rights reserved.
# License: AGPL-3.0

import argparse
import logging
import sys
from http.client import HTTPConnection, HTTPSConnection
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlparse

from pyzsync import (
	SOURCE_REMOTE,
	HTTPRangeReader,
	ProgressListener,
	Range,
	create_zsync_file,
	create_zsync_info,
	get_patch_instructions,
	patch_file,
	read_zsync_file,
)


def zsyncmake(file_path: Path) -> None:
	create_zsync_file(file=file_path, zsync_file=file_path.with_name(f"{file_path.name}.zsync"))


def zsync(url: str) -> None:
	url_obj = urlparse(url)

	conn_class = HTTPConnection if url_obj.scheme == "http" else HTTPSConnection
	connection = conn_class(url_obj.netloc, timeout=600, blocksize=65536)

	print(f"Fetching zsync file {url_obj.geturl()}...")
	connection.request("GET", url_obj.path)
	response = connection.getresponse()
	if response.status != 200:
		raise RuntimeError(f"Failed to fetch {url_obj.geturl()}: {response.status} - {response.read().decode('utf-8', 'ignore').strip()}")
	total_size = int(response.headers["Content-Length"])
	position = 0
	with TemporaryDirectory() as temp_dir:
		zsync_file = Path(temp_dir) / url_obj.path.split("/")[-1]
		with open(zsync_file, "wb") as file:
			last_completed = 0
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

	local_files = [Path(Path(zsync_info.filename).name).absolute()]
	local_files.extend(local_files[0].parent.glob(f"{local_files[0].name}.zsync-tmp-*"))

	print(f"Analyzing {len(local_files)} local file{'s' if len(local_files) > 1 else ''}...")
	instructions = get_patch_instructions(zsync_info, local_files)
	remote_bytes = sum(i.size for i in instructions if i.source == SOURCE_REMOTE)
	ratio = remote_bytes * 100 / zsync_info.length
	print(f"Need to fetch {remote_bytes} bytes ({ratio:.2f}%)")

	path = url_obj.path.split("/")
	path[-1] = zsync_info.url.split("/")[-1]
	rurl = f"{url_obj.scheme}://{url_obj.netloc}{'/'.join(path)}"
	if ratio != 0:
		print(f"Fetching {remote_bytes} bytes from {rurl}...")

	class PrintingProgressListener(ProgressListener):
		def __init__(self) -> None:
			self.last_completed = 0

		def progress_changed(self, reader: HTTPRangeReader, position: int, total: int, per_second: int) -> None:
			completed = round(position * 100 / total, 1)
			if completed == self.last_completed:
				return
			print(
				f"\r::: {completed:0.1f} % ::: {position/1_000_000:.2f}/{total/1_000_000:.2f} MB ::: {per_second/1_000:.0f} kB/s :::",
				end="",
			)
			self.last_completed = completed

	def range_reader_factory(ranges: list[Range]) -> HTTPRangeReader:
		range_reader = HTTPRangeReader(rurl, ranges)
		range_reader.register_progress_listener(PrintingProgressListener())
		return range_reader

	sha1 = patch_file(local_files, instructions, range_reader_factory=range_reader_factory, return_hash="sha1")
	if ratio != 0:
		print("")
	if sha1 != zsync_info.sha1:
		raise RuntimeError(f"SHA1 mismatch: {sha1.hex()} != {zsync_info.sha1.hex()}")

	print(f"Successfully created {local_files[0]}")


def compare(file1: Path, file2: Path) -> None:
	zsync_info = create_zsync_info(file1)
	instructions = get_patch_instructions(zsync_info, file2)
	file2_bytes = sum(i.size for i in instructions if i.source != SOURCE_REMOTE)
	ratio = file2_bytes * 100 / zsync_info.length
	print(f"{file2} contains {ratio:.2f}% of data to create {file1}")


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("--log-level", choices=["debug", "info", "warning", "error", "critical"], default="warning")

	subparsers = parser.add_subparsers(dest="command")

	p_zsyncmake = subparsers.add_parser("zsyncmake", help="Create zsync file from FILE")
	p_zsyncmake.add_argument("file", help="Path to the file")

	p_zsync = subparsers.add_parser("zsync", help="Fetch file from ZSYNC_URL")
	p_zsync.add_argument("zsync_url", help="URL to the zsync file")

	p_compare = subparsers.add_parser("compare", help="Compare two files")
	p_compare.add_argument("file", help="Path to the file", nargs=2)

	args = parser.parse_args()

	logging.basicConfig(format="[%(levelno)d] [%(asctime)s.%(msecs)03d] %(message)s   (%(filename)s:%(lineno)d)")
	logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))

	if args.command == "zsyncmake":
		return zsyncmake(Path(args.file))

	if args.command == "zsync":
		return zsync(args.zsync_url)

	if args.command == "compare":
		return compare(Path(args.file[0]), Path(args.file[1]))

	parser.print_help()


if __name__ == "__main__":
	try:
		main()
	except BaseException as err:
		print(err, file=sys.stderr)
		sys.exit(1)
	sys.exit(0)
