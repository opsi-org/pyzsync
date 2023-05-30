# Copyright (c) 2023 uib GmbH <info@uib.de>
# This code is owned by the uib GmbH, Mainz, Germany (uib.de). All rights reserved.
# License: AGPL-3.0
"""
This file is part of opsi - https://www.opsi.org
"""

import platform
from subprocess import check_output
import pytest
from _pytest.config import Config
from _pytest.nodes import Item


@pytest.hookimpl()
def pytest_configure(config: Config) -> None:
	config.addinivalue_line("markers", "zsyncmake_available: mark test to run only if zsyncmake is available")
	config.addinivalue_line("markers", "targz_available: mark test to run only if tar is available")
	config.addinivalue_line("markers", "windows: mark test to run only on windows")
	config.addinivalue_line("markers", "linux: mark test to run only on linux")
	config.addinivalue_line("markers", "darwin: mark test to run only on darwin")


PLATFORM = platform.system().lower()
try:
	ZSYNCMAKE_VERSION = check_output(["zsyncmake", "-V"]).decode().split("\n", 1)[0].split()[1]
except Exception as err:  # pylint: disable=broad-except
	ZSYNCMAKE_VERSION = None
try:
	TAR_VERSION = check_output(["tar", "--version"]).decode().split("\n", 1)[0].split()[-1]
except Exception as err:  # pylint: disable=broad-except
	TAR_VERSION = None
try:
	GZIP_VERSION = check_output(["gzip", "-V"]).decode().split("\n", 1)[0].split()[-1]
except Exception as err:  # pylint: disable=broad-except
	GZIP_VERSION = None

def pytest_runtest_setup(item: Item) -> None:
	for marker in item.iter_markers():
		if marker.name == "zsyncmake_available" and not ZSYNCMAKE_VERSION:
			pytest.skip("zsyncmake not available")

		if marker.name == "targz_available" and (not TAR_VERSION or not GZIP_VERSION):
			pytest.skip("tar/gz not available")

		if marker.name in ("windows", "linux", "darwin") and marker.name != PLATFORM:
			pytest.skip(f"Cannot run on {PLATFORM}")
