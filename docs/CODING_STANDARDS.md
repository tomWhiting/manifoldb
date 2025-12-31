# ManifoldDB Coding Standards Checklist

This document provides a checklist for code review and quality verification. Use this to ensure all code meets project standards before merging.

## Quick Reference

```
Rust Standards: Strict clippy, no unwrap/expect/panic in library code
Module Structure: mod.rs for declarations/re-exports, implementation in separate files
Dependencies: cargo add only, workspace Cargo.toml for shared deps
Testing: Unit tests in same file, integration tests in tests/
```

---

## Code Quality Checklist

### Error Handling

- [ ] **No `unwrap()` calls** - Use `?` operator or proper error handling
- [ ] **No `expect()` calls** - Return errors with context instead
- [ ] **No `panic!()` macro** - Errors should be recoverable
- [ ] **Proper Result/Option handling** - Use combinators or match expressions
- [ ] **Meaningful error messages** - Include context about what failed

```rust
// BAD
let value = map.get("key").unwrap();
let file = File::open(path).expect("failed to open");

// GOOD
let value = map.get("key").ok_or_else(|| anyhow!("missing key: {}", key))?;
let file = File::open(path)
    .with_context(|| format!("failed to open config file: {}", path.display()))?;
```

### Memory & Performance

- [ ] **No unnecessary `.clone()`** - Only clone when ownership transfer is needed
- [ ] **Use references where possible** - Prefer `&str` over `String` in parameters
- [ ] **Avoid allocations in hot paths** - Reuse buffers where appropriate
- [ ] **Prefer iterators over collect()** - Stream when possible, don't materialize unnecessarily
- [ ] **Use `Cow<str>` for conditional ownership** - When sometimes owned, sometimes borrowed

```rust
// BAD - unnecessary clone
fn process(data: &Data) {
    let owned = data.name.clone(); // clone not needed
    println!("{}", owned);
}

// GOOD - use reference
fn process(data: &Data) {
    println!("{}", &data.name);
}
```

### Safety

- [ ] **No `unsafe` blocks** - Unless absolutely necessary with detailed justification
- [ ] **No raw pointers** - Use safe abstractions
- [ ] **Validate inputs at boundaries** - Don't trust external data
- [ ] **No panicking in library code** - Only in tests or with clear documentation

### Module Organization

- [ ] **`mod.rs` for declarations only** - Module docs, submodule declarations, re-exports
- [ ] **No implementation in `mod.rs`** - Implementation goes in named files
- [ ] **One responsibility per module** - Clear separation of concerns
- [ ] **Consistent naming** - snake_case for files, modules, functions

```rust
// mod.rs should look like:
//! Module-level documentation explaining purpose.

mod submodule_a;
mod submodule_b;

pub use submodule_a::PublicType;
pub use submodule_b::public_function;
```

### Workspace Structure

Follow the established crate boundaries:

| Crate | Purpose | Dependencies |
|-------|---------|--------------|
| `manifoldb` | Public API, Database struct | All other crates |
| `manifoldb-core` | Core types (Entity, Value, Edge) | None |
| `manifoldb-storage` | Storage traits, Redb backend | manifoldb-core |
| `manifoldb-graph` | Graph storage, traversal | manifoldb-core, manifoldb-storage |
| `manifoldb-vector` | Vector storage, ANN search | manifoldb-core, manifoldb-storage |
| `manifoldb-query` | Query parsing, execution | All except manifoldb |

- [ ] **Respect crate boundaries** - Don't add dependencies that violate layering
- [ ] **Core types in manifoldb-core** - EntityId, Value, Entity, Edge belong there
- [ ] **Public API in manifoldb** - User-facing types re-exported from main crate

### Documentation

- [ ] **Module-level docs (`//!`)** - Explain purpose and usage
- [ ] **Public item docs (`///`)** - Document all public APIs
- [ ] **Examples in docs** - Show how to use complex APIs
- [ ] **No TODO comments without context** - Include what needs doing and why

```rust
/// Searches for entities matching the query vector.
///
/// # Arguments
///
/// * `vector_name` - The name of the vector field to search
/// * `query` - The query vector for similarity comparison
///
/// # Example
///
/// ```ignore
/// let results = db.search("documents", "embedding")
///     .query(vec![0.1, 0.2, 0.3])
///     .limit(10)
///     .execute()?;
/// ```
pub fn search(&self, collection: &str, vector_name: &str) -> EntitySearchBuilder {
    // ...
}
```

### Type Design

- [ ] **Use builder pattern for complex construction** - Fluent APIs with method chaining
- [ ] **Prefer enums over stringly-typed code** - Type safety over flexibility
- [ ] **Use newtypes for domain concepts** - `EntityId` not `u64`
- [ ] **Implement standard traits** - Debug, Clone, PartialEq where appropriate
- [ ] **Use `#[must_use]` on builders** - Prevent accidental drops

```rust
// Builder pattern example
#[must_use]
pub fn limit(mut self, limit: usize) -> Self {
    self.limit = limit;
    self
}
```

### Testing

- [ ] **Unit tests in same file** - `#[cfg(test)] mod tests { ... }`
- [ ] **Integration tests in `tests/` directory** - Cross-module workflows
- [ ] **Test edge cases** - Empty inputs, errors, boundaries
- [ ] **Descriptive test names** - `test_search_with_empty_results`
- [ ] **No unwrap() in assertions** - Use `assert!`, `assert_eq!`, or explicit matching

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn search_returns_empty_for_no_matches() {
        let db = setup_test_db();
        let results = db.search("docs", "embedding")
            .query(vec![0.0; 768])
            .execute()
            .expect("search should succeed");
        assert!(results.is_empty());
    }
}
```

### Dependencies

- [ ] **Use `cargo add` only** - Never edit Cargo.toml manually for deps
- [ ] **Workspace dependencies** - Shared deps go in workspace Cargo.toml
- [ ] **Minimal feature flags** - Only enable what's needed
- [ ] **Check for existing deps** - Don't add duplicates

---

## Clippy Lints

The project should pass strict clippy. Run before committing:

```bash
cargo clippy --workspace --all-targets -- -D warnings
```

Key lints to watch for:
- `clippy::unwrap_used` - Should be denied
- `clippy::expect_used` - Should be denied
- `clippy::panic` - Should be denied
- `clippy::unnecessary_clone` - Warn
- `clippy::redundant_clone` - Warn

---

## Pre-Commit Checklist

Before committing any code:

```bash
# Format
cargo fmt --all

# Lint
cargo clippy --workspace --all-targets -- -D warnings

# Test
cargo test --workspace

# Build (catch any remaining issues)
cargo build --workspace
```

All four must pass with no warnings or errors.

---

## Query Engine Specific Standards

For code in `manifoldb-query`:

### AST Types (`ast/`)

- [ ] **Implement Display** - For debugging and EXPLAIN output
- [ ] **Derive standard traits** - Debug, Clone, PartialEq at minimum
- [ ] **Use builder methods** - For constructing complex AST nodes

### Logical Plan (`plan/logical/`)

- [ ] **Plan nodes are immutable** - Build once, don't mutate
- [ ] **Schema propagation** - Each node knows its output schema
- [ ] **Display for EXPLAIN** - `display_tree()` method for plan visualization

### Physical Plan (`plan/physical/`)

- [ ] **Match logical plan structure** - Clear mapping from logical to physical
- [ ] **Cost estimates** - Physical nodes should support cost estimation (future)

### Execution (`exec/`)

- [ ] **Streaming where possible** - Don't buffer entire results unless necessary
- [ ] **Proper resource cleanup** - Close iterators, release locks

---

## Common Patterns

### Error Handling Pattern

```rust
use anyhow::{Context, Result};

pub fn load_config(path: &Path) -> Result<Config> {
    let content = fs::read_to_string(path)
        .with_context(|| format!("failed to read config from {}", path.display()))?;

    let config: Config = toml::from_str(&content)
        .context("failed to parse config as TOML")?;

    config.validate()?;
    Ok(config)
}
```

### Builder Pattern

```rust
#[derive(Debug, Clone)]
pub struct SearchBuilder {
    query: Option<Vec<f32>>,
    limit: usize,
    filter: Option<Filter>,
}

impl SearchBuilder {
    pub fn new() -> Self {
        Self { query: None, limit: 10, filter: None }
    }

    #[must_use]
    pub fn query(mut self, q: Vec<f32>) -> Self {
        self.query = Some(q);
        self
    }

    #[must_use]
    pub fn limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }

    pub fn execute(self) -> Result<Vec<ScoredEntity>> {
        let query = self.query.ok_or_else(|| anyhow!("query vector required"))?;
        // ... execute search
    }
}
```

### Module File Layout

```
src/
├── lib.rs              # Crate root, re-exports
├── config.rs           # Single-file module
├── feature/            # Multi-file module
│   ├── mod.rs          # Declarations + re-exports only
│   ├── types.rs        # Type definitions
│   └── operations.rs   # Implementation
```
