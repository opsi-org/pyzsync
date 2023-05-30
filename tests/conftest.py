# Copyright (c) 2023 uib GmbH <info@uib.de>
# This code is owned by the uib GmbH, Mainz, Germany (uib.de). All rights reserved.
# License: AGPL-3.0

import platform
from subprocess import check_output

import pytest
from _pytest.config import Config
from _pytest.nodes import Item


@pytest.hookimpl()
def pytest_configure(config: Config) -> None:
	config.addinivalue_line("markers", "zsync_available: mark test to run only if zsync is available")
	config.addinivalue_line("markers", "zsyncmake_available: mark test to run only if zsyncmake is available")
	config.addinivalue_line("markers", "targz_available: mark test to run only if tar is available")
	config.addinivalue_line("markers", "windows: mark test to run only on windows")
	config.addinivalue_line("markers", "linux: mark test to run only on linux")
	config.addinivalue_line("markers", "darwin: mark test to run only on darwin")
	config.addinivalue_line("markers", "posix: mark test to run only on posix")


PLATFORM = platform.system().lower()
try:
	ZSYNC_VERSION = check_output(["zsync", "-V"]).decode().split("\n", 1)[0].split()[1]
except Exception:  # pylint: disable=broad-except
	ZSYNC_VERSION = ""
try:
	ZSYNCMAKE_VERSION = check_output(["zsyncmake", "-V"]).decode().split("\n", 1)[0].split()[1]
except Exception:  # pylint: disable=broad-except
	ZSYNCMAKE_VERSION = ""
try:
	TAR_VERSION = check_output(["tar", "--version"]).decode().split("\n", 1)[0].split()[-1]
except Exception:  # pylint: disable=broad-except
	TAR_VERSION = ""
try:
	GZIP_VERSION = check_output(["gzip", "-V"]).decode().split("\n", 1)[0].split()[-1]
except Exception:  # pylint: disable=broad-except
	GZIP_VERSION = ""


def pytest_runtest_setup(item: Item) -> None:
	for marker in item.iter_markers():
		if marker.name == "zsync_available" and not ZSYNC_VERSION:
			pytest.skip("zsync not available")

		if marker.name == "zsyncmake_available" and not ZSYNCMAKE_VERSION:
			pytest.skip("zsyncmake not available")

		if marker.name == "targz_available" and (not TAR_VERSION or not GZIP_VERSION):
			pytest.skip("tar/gz not available")

		if marker.name in ("windows", "linux", "darwin") and marker.name != PLATFORM:
			pytest.skip(f"Cannot run on {PLATFORM}")

		if marker.name in ("posix") and PLATFORM not in ("linux", "darwin"):
			pytest.skip(f"Cannot run on {PLATFORM}")
