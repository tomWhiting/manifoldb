//! Error types for the CLI.

use std::path::PathBuf;

use thiserror::Error;

/// CLI-specific result type.
pub type Result<T> = std::result::Result<T, CliError>;

/// CLI error types.
#[derive(Error, Debug)]
#[allow(dead_code)]
pub enum CliError {
    /// No database specified.
    #[error("no database specified. Use --database or set MANIFOLD_DB environment variable")]
    NoDatabaseSpecified,

    /// Database file not found.
    #[error("database not found: {0}")]
    DatabaseNotFound(PathBuf),

    /// ManifoldDB error.
    #[error("database error: {0}")]
    Database(#[from] manifoldb::Error),

    /// Transaction error.
    #[error("transaction error: {0}")]
    Transaction(#[from] manifoldb::TransactionError),

    /// IO error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON serialization error.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// CSV error.
    #[error("CSV error: {0}")]
    Csv(#[from] csv::Error),

    /// REPL error.
    #[error("REPL error: {0}")]
    Readline(#[from] rustyline::error::ReadlineError),

    /// Invalid input.
    #[error("invalid input: {0}")]
    InvalidInput(String),

    /// File not found.
    #[error("file not found: {0}")]
    FileNotFound(PathBuf),

    /// Server already running.
    #[error("server is already running")]
    ServerAlreadyRunning,

    /// Server not running.
    #[error("server is not running")]
    ServerNotRunning,

    /// Daemonization error.
    #[error("failed to daemonize: {0}")]
    Daemon(String),

    /// Invalid PID file.
    #[error("invalid PID file")]
    InvalidPidFile,

    /// Invalid path encoding.
    #[error("invalid path encoding")]
    InvalidPath,

    /// No home directory.
    #[error("could not determine home directory")]
    NoHomeDir,

    /// Unsupported platform.
    #[error("unsupported platform: {0}")]
    UnsupportedPlatform(String),

    /// Server error.
    #[error("server error: {0}")]
    Server(#[from] anyhow::Error),

    /// Nix/signal error (Unix only).
    #[cfg(unix)]
    #[error("system error: {0}")]
    Nix(#[from] nix::Error),

    /// Parse error.
    #[error("parse error: {0}")]
    Parse(String),
}
