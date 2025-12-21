# ManifoldDB Project Instructions

## Dependency Management
- Always use the latest versions of dependencies
- Use `cargo add` to add new dependencies (e.g., `cargo add serde -F derive`)
- Keep dependencies up to date with `cargo update`
- Shared dependencies go in workspace `Cargo.toml`

## Module Structure
- Use `mod.rs` files for controlling visibility, organization, and documentation
- Each module directory should have a `mod.rs` that:
  - Contains module-level documentation (`//!` comments)
  - Declares submodules (`mod foo;`)
  - Re-exports public items (`pub use foo::Bar;`)

## Code Quality
- Run `cargo clippy --workspace --all-targets` before committing
- Run `cargo fmt --all` to format code
- Write tests for all new functionality
- No `unwrap()` or `expect()` in library code (allowed in tests)

## Workspace Structure
- `manifoldb` - Main database crate and public API
- `manifoldb-core` - Core types (EntityId, Value, Entity, Edge)
- `manifoldb-storage` - Storage engine traits and Redb backend
- `manifoldb-graph` - Graph storage and traversal
- `manifoldb-vector` - Vector storage and similarity search
- `manifoldb-query` - Query parsing and execution

## Testing
- Unit tests go in the same file: `#[cfg(test)] mod tests { ... }`
- Integration tests go in `tests/` directory
- Run all tests: `cargo test --workspace`
