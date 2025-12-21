# Contributing to ManifoldDB

This document outlines the conventions and practices used in this project.

## Development Setup

1. Ensure you have the latest stable Rust toolchain installed
2. Clone the repository
3. Run `cargo build` to build all crates
4. Run `cargo test` to verify everything works

## Dependency Management

- **Use the latest versions** of all dependencies
- **Use `cargo add`** to add new dependencies (e.g., `cargo add serde -F derive`)
- Keep dependencies up to date by periodically running `cargo update`
- All shared dependencies should be specified in the workspace `Cargo.toml` with versions

## Code Organization

### Module Structure

- **Use `mod.rs` files** for controlling visibility, organization, and documentation
- Each module directory should have a `mod.rs` that:
  - Contains module-level documentation (`//!` comments)
  - Declares submodules (`mod foo;`)
  - Re-exports public items (`pub use foo::Bar;`)

Example structure:
```
src/
├── lib.rs           # Crate root
└── engine/
    ├── mod.rs       # Module documentation and re-exports
    ├── traits.rs    # Core traits
    └── error.rs     # Error types
```

### Crate Structure

This is a Cargo workspace with multiple crates:
- `manifoldb` - Main database crate and public API
- `manifoldb-core` - Core types (EntityId, Value, etc.)
- `manifoldb-storage` - Storage engine traits and backends
- `manifoldb-graph` - Graph storage and traversal
- `manifoldb-vector` - Vector storage and similarity search
- `manifoldb-query` - Query parsing and execution

## Code Quality

### Clippy

We use strict clippy settings. Run before committing:
```bash
cargo clippy --workspace --all-targets
```

Key lint groups enabled:
- `clippy::pedantic` - Stricter lints for better code
- `clippy::nursery` - Experimental lints
- `clippy::unwrap_used` / `clippy::expect_used` - Warn on panic-prone code (allowed in tests)

### Formatting

Use rustfmt with the project settings:
```bash
cargo fmt --all
```

### Testing

- Write tests for all new functionality
- Place unit tests in the same file using `#[cfg(test)] mod tests { ... }`
- Place integration tests in the `tests/` directory
- Run all tests with `cargo test --workspace`

## Conventions

### Error Handling

- Use `thiserror` for error types
- Return `Result` types, avoid panicking
- Use `?` operator for error propagation

### Documentation

- Add doc comments (`///`) to all public items
- Add module-level documentation (`//!`) to `mod.rs` files
- Include examples in documentation where helpful

### Naming

- Use descriptive names
- Prefer full words over abbreviations
- Types: `PascalCase`
- Functions/methods: `snake_case`
- Constants: `SCREAMING_SNAKE_CASE`

### Safety

- Avoid `unsafe` code unless absolutely necessary
- No `unwrap()` or `expect()` in library code (use proper error handling)
- Be mindful of OWASP security considerations
