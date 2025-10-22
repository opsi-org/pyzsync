# Changelog of pyzsync

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [1.4.20] - 2025-10-22
### Fixed
- Avoid using spaces in the Range header, as this can cause issues with certain versions of Apache

## [1.4.19] - 2025-08-01
### Added
- Add Python 3.14 support

### Changed
- Move from poetry to uv

## [1.3.2] - 2024-08-22
### Changed
- Bump version

## [1.3.1] - 2024-08-21
### Changed
- Update python packages
- Add Python 3.13 support

## [1.3.0] - 2024-06-14
### Fixed
- Fix runner tags

### Changed
- Callback with block 0

### Added
- Implement progress_callback for zsync file creation

## [1.2.6] - 2024-03-18
### Changed
- Use clippy as linter

## [1.2.5] - 2024-02-01
### Changed
- Bump version

## [1.2.4] - 2024-02-01
### Fixed
- Fix release workflow

## [1.2.3] - 2024-02-01
### Changed
- Use build-backend maturin

## [1.2.2] - 2024-02-01
### Changed
- Bump version

## [1.2.1] - 2024-02-01
### Changed
- Update dependencies

## [1.2.0] - 2023-07-14
### Changed
- Use ruff

### Added
- Add Python 3.12 support

### Fixed
- Fix pyi

### Added
- Optimize patch instructions

## [1.0.0] - 2023-07-05
### Added
- Add progress_callback to get_patch_instructions
- Add Patcher.abort()

## [0.9.0] - 2023-06-15
### Changed
- Do not raise

### Fixed
- Fix update_rsum

## [0.8.0] - 2023-06-14
### Fixed
- Fix file delete

## [0.7.0] - 2023-06-13
### Changed
- Refactor RangeReader

### Added
- Add __repr__ to PatchInstruction

## [0.6.2] - 2023-06-13
### Changed
- Improve logging

## [0.6.1] - 2023-06-12
### Fixed
- Fix tests

## [0.6.0] - 2023-06-02
### Changed
- Do not call request in init

## [0.5.0] - 2023-06-02
### Fixed
- Fix division by zero

## [0.4.0] - 2023-06-01
### Changed
- Change limit

## [0.3.0] - 2023-06-01
### Fixed
- Fix test for windows

## [0.2.0] - 2023-05-31
### Fixed
- Fix reader

## [0.1.0] - 2023-05-30
### Changed
- Initial release
