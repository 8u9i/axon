# Contributing to Axon

Thanks for helping make Axon better. Axon is a model-weight container and
runtime loader, so changes should preserve the core goals: safe parsing,
low-memory loading, zero-copy access where possible, and clear interoperability.

## Development Setup

```bash
cargo build --workspace
cargo test --workspace
```

For Python bindings:

```bash
python -m pip install -e ./python
python -m build ./python
```

## Quality Gates

Run these before opening a pull request:

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

If you touch Python package metadata, also run:

```bash
python -m build ./python
python -c "import axon; print(axon.__version__)"
```

## Format Compatibility

The `.axon` binary format has its own compatibility policy. Read
`docs/format-versioning.md` before changing headers, descriptor layout, dtype
codes, flags, offsets, or payload interpretation.

## Pull Request Guidelines

- Keep changes scoped to one behavior or subsystem.
- Add tests for user-facing CLI behavior, parser validation, and runtime access
  when applicable.
- Do not introduce panics for malformed or untrusted `.axon` files.
- Document new public Rust, C FFI, CLI, or Python APIs.
- Update `CHANGELOG.md` for externally visible behavior changes.

## Reporting Security Issues

Do not open public issues for vulnerabilities. Follow `SECURITY.md`.
