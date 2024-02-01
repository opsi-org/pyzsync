# pyzsync

A Python module written in Rust that implements the [zsync algorithm](http://zsync.moria.org.uk).

## Usage

### Use the Python module as a script
```shell
# Show help.
python -m pyzsync --help

# Create a zsync file.
poetry run python -m pyzsync zsyncmake bigfile

# Compare two files and show how much data from the first file
# can be used to create the second file using the zsync algorithm.
python -m pyzsync compare bigfile1 bigfile2

# Download a file using zsync.
# This will automatically use blocks from the local files
# noble-desktop-amd64.iso and noble-desktop-amd64.iso.zsync-tmp-*
# if available.
python -m pyzsync zsync https://cdimage.ubuntu.com/daily-live/current/noble-desktop-amd64.iso.zsync
```

### Use the Python module in a script
```python
from pyzsync import create_zsync_file

create_zsync_file("bigfile", "bigfile.zsync")
```

See `tests/test_pyzsync.py` and `pyzsync/__main__.py` for more examples.


## Build / Development
Based on [PyO3](https://pyo3.rs)

```
# Install toolchain (linux)
rustup toolchain install beta-x86_64-unknown-linux-gnu

# Build package in debug mode and install it to virtualenv
poetry install
poetry run maturin develop --release

# Run cargo test
cargo test --no-default-features

# Run pytest
poetry run pytest -vv

# Build release package
poetry run maturin build --release
```
