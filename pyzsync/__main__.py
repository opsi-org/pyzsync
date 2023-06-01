# Copyright (c) 2023 uib GmbH <info@uib.de>
# This code is owned by the uib GmbH, Mainz, Germany (uib.de). All rights reserved.
# License: AGPL-3.0

import argparse
import logging
import sys
from http.client import HTTPConnection, HTTPSConnection
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import BinaryIO
from urllib.parse import urlparse

from pyzsync import (
	SOURCE_REMOTE,
	HTTPRangeReader,
	Range,
	create_zsync_file,
	create_zsync_info,
	get_patch_instructions,
	patch_file,
	read_zsync_file,
)


def main() -> None:
	parser = argparse.ArgumentParser()
	subparsers = parser.add_subparsers(dest="command")

	zsyncmake = subparsers.add_parser("zsyncmake", help="Create zsync file from FILE")
	zsyncmake.add_argument("file", help="Path to the file")

	zsync = subparsers.add_parser("zsync", help="Fetch file from ZSYNC_URL")
	zsync.add_argument("zsync_url", help="URL to the zsync file")

	compare = subparsers.add_parser("compare", help="Compare two files")
	compare.add_argument("file", help="Path to the file", nargs=2)

	args = parser.parse_args()

	logging.basicConfig(format="[%(levelno)d] [%(asctime)s.%(msecs)03d] %(message)s   (%(filename)s:%(lineno)d)")
	logging.getLogger().setLevel(logging.INFO)

	if args.command == "zsyncmake":
		file = Path(args.file)
		create_zsync_file(file=file, zsync_file=file.with_name(f"{file.name}.zsync"))

	elif args.command == "zsync":
		url = urlparse(args.zsync_url)
		conn_class = HTTPConnection if url.scheme == "http" else HTTPSConnection
		connection = conn_class(url.netloc, timeout=600, blocksize=65536)
		print(f"Fetching zsync file {url.geturl()}...")
		connection.request("GET", url.path)
		response = connection.getresponse()
		if response.status != 200:
			raise RuntimeError(f"Failed to fetch {url.geturl()}: {response.status} - {response.read().decode('utf-8', 'ignore').strip()}")
		total_size = int(response.headers["Content-Length"])
		position = 0
		with TemporaryDirectory() as temp_dir:
			zsync_file = Path(temp_dir) / url.path.split("/")[-1]
			with open(zsync_file, "wb") as file:
				while position < total_size:
					data = response.read(65536)
					if not data:
						raise RuntimeError(f"Failed to fetch {url.geturl()}: read only {position/total_size} bytes")
					file.write(data)
					position += len(data)

			zsync_info = read_zsync_file(zsync_file)

		local_file = Path(Path(zsync_info.filename).name).absolute()
		instructions = get_patch_instructions(zsync_info, local_file)
		remote_bytes = sum([i.size for i in instructions if i.source == SOURCE_REMOTE])
		ratio = (zsync_info.length - remote_bytes) * 100 / zsync_info.length
		print(f"Local file {local_file} contains {ratio:.2f}% of data")

		path = url.path.split("/")
		path[-1] = zsync_info.url.split("/")[-1]
		rurl = f"{url.scheme}://{url.netloc}{'/'.join(path)}"
		if ratio != 100:
			print(f"Fetching {remote_bytes} bytes from {rurl}...")

		def fetch_function(ranges: list[Range]) -> BinaryIO:
			return HTTPRangeReader(rurl, ranges)

		sha1 = patch_file(local_file, instructions, fetch_function, return_hash="sha1")
		if sha1 != zsync_info.sha1:
			raise RuntimeError(f"SHA1 mismatch: {sha1} != {zsync_info.sha1}")

		print(f"Successfully created {local_file}")

	elif args.command == "compare":
		file1 = Path(args.file[0])
		file2 = Path(args.file[1])
		zsync_info = create_zsync_info(file1)
		instructions = get_patch_instructions(zsync_info, file2)
		file2_bytes = sum([i.size for i in instructions if i.source != SOURCE_REMOTE])
		ratio = file2_bytes * 100 / zsync_info.length
		print(f"{file2} contains {ratio:.2f}% of data to create {file1}")

	else:
		parser.print_help()


if __name__ == "__main__":
	try:
		main()
	except Exception as err:
		print(err, file=sys.stderr)
		sys.exit(1)
	sys.exit(0)
