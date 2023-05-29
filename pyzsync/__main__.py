#!/usr/bin/env python

import sys
import argparse
from pathlib import Path
from pyzsync import (
	calc_block_size, calc_block_infos, create_zsync_file
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
		block_size = max(calc_block_size(file1.stat().st_size), calc_block_size(file2.stat().st_size))
		block_infos1 = calc_block_infos(file=args.file[0], block_size=block_size, rsum_bytes=0, checksum_bytes=16)
		block_infos2 = calc_block_infos(file=args.file[1], block_size=block_size, rsum_bytes=0, checksum_bytes=16)
		hashes1 = [b.checksum for b in block_infos1]
		hashes2 = [b.checksum for b in block_infos2]
		hashes_shared = [h for h in hashes1 if h in hashes2]
		print(f"{file1} shares {len(hashes_shared)}/{len(hashes1)} blocks with {file2} ({len(hashes_shared) / len(hashes1) * 100:.2f}%)")
	else:
		parser.print_help()


if __name__ == "__main__":
	try:
		main()
	except Exception as err:
		print(err, file=sys.stderr)
		sys.exit(1)
	sys.exit(0)
