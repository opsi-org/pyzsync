#!/usr/bin/env python

import sys
import argparse
from pathlib import Path
from pyzsync import (
	create_zsync_file, create_zsync_info, get_patch_instructions, Source
)


def main():
	parser = argparse.ArgumentParser()
	subparsers = parser.add_subparsers(dest="command")

	zsyncmake = subparsers.add_parser("zsyncmake", help="Create zsync file from FILE")
	zsyncmake.add_argument("file", help="Path to the file")

	compare = subparsers.add_parser("compare", help="Compare two files")
	compare.add_argument("file", help="Path to the file", nargs=2)

	args = parser.parse_args()

	if args.command == "zsyncmake":
		file = Path(args.file)
		create_zsync_file(file=file, zsync_file=file.with_name(f"{file.name}.zsync"))

	elif args.command == "compare":
		file1 = Path(args.file[0])
		file2 = Path(args.file[1])
		zsync_info = create_zsync_info(file1)
		instructions = get_patch_instructions(zsync_info, file2)
		file2_bytes = sum([i.size for i in instructions if i.source == Source.Local])
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
