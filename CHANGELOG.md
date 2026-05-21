# Changelog

All notable changes to Axon will be documented in this file.

This project follows semantic versioning for released crates, packages, and
CLI artifacts. The `.axon` binary format version is tracked separately in the
file header and documented in `docs/format-versioning.md`.

## [Unreleased]

### Added

- CLI integration tests for pack, validate, list, inspect, extract, unpack,
  convert, runtime inspect, runtime tensor, and runtime slice.
- Format versioning and compatibility policy.
- Python `pyproject.toml` package metadata and local build instructions.
- Tag-driven release workflow for native artifacts and Python distributions.
- FFI `axon_last_error` support and C header lifecycle documentation.

### Changed

- CLI user-facing errors now return clean stderr messages and nonzero exits
  instead of panic-shaped output for normal bad inputs.
- README now positions Axon as a model-weight container/runtime, not a neural
  network training framework.

## [1.0.0]

- Initial Rust workspace with core format library, runtime, CLI, FFI, Python
  bindings, examples, docs, and benchmarks.
