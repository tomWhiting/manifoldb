//! Database configuration.

/// Configuration options for opening a database.
#[derive(Debug, Clone)]
pub struct Config {
    /// Path to the database file.
    pub path: std::path::PathBuf,
    /// Whether to create the database if it doesn't exist.
    pub create_if_missing: bool,
}

impl Config {
    /// Create a new configuration with the given path.
    #[must_use]
    pub fn new(path: impl Into<std::path::PathBuf>) -> Self {
        Self { path: path.into(), create_if_missing: true }
    }

    /// Set whether to create the database if it doesn't exist.
    #[must_use]
    pub const fn create_if_missing(mut self, create: bool) -> Self {
        self.create_if_missing = create;
        self
    }
}
