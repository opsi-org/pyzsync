# pyzsync

## PyO3
- https://pyo3.rs/v0.18.3/
- https://github.com/mre/hyperjson

## Develop
```
poetry run maturin develop --release
poetry run pytest -vv
```

### rust-analyzer

- https://stackoverflow.com/questions/76171390/proc-macro-main-not-expanded-rust-analyzer-not-spawning-server
- install toolchain (linux)

```
rustup toolchain install sbeta-x86_64-unknown-linux-gnu

```
## Build release
```
poetry run maturin build --release
```
