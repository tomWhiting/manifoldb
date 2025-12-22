//! ManifoldDB Command Line Interface
//!
//! A CLI tool for interacting with ManifoldDB databases.

mod commands;
mod error;
mod output;
mod repl;

use std::path::PathBuf;

use clap::{Parser, Subcommand};

use crate::commands::{
    collections, execute_command, graph, import_export, indexes, info, open, query,
};
use crate::error::Result;
use crate::repl::Repl;

/// ManifoldDB Command Line Interface
///
/// A multi-paradigm database combining graph, vector, and relational capabilities.
#[derive(Parser, Debug)]
#[command(name = "manifold")]
#[command(author, version, about, long_about = None)]
#[command(propagate_version = true)]
pub struct Cli {
    /// Path to the database file
    #[arg(short, long, env = "MANIFOLD_DB", global = true)]
    pub database: Option<PathBuf>,

    /// Output format
    #[arg(short, long, value_enum, default_value = "table", global = true)]
    pub format: OutputFormat,

    /// The subcommand to execute.
    #[command(subcommand)]
    pub command: Commands,
}

/// Output format for query results
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum OutputFormat {
    /// Pretty-printed table format
    Table,
    /// JSON format
    Json,
    /// CSV format
    Csv,
    /// Compact single-line format
    Compact,
}

/// Available CLI commands.
#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Open or create a database (validates the path)
    Open {
        /// Path to the database file
        path: PathBuf,
    },

    /// Show database information and statistics
    Info,

    /// Execute a SQL query (SELECT)
    Query {
        /// The SQL query to execute
        sql: String,
    },

    /// Execute a SQL statement (INSERT, UPDATE, DELETE, DDL)
    Execute {
        /// The SQL statement to execute
        sql: String,
    },

    /// Start an interactive SQL REPL
    Repl,

    /// Import data from a file
    Import {
        /// Path to the file to import
        file: PathBuf,

        /// Name of the collection/table to import into
        #[arg(short, long)]
        collection: String,

        /// Format of the input file
        #[arg(short = 'F', long, value_enum, default_value = "json")]
        format: ImportFormat,
    },

    /// Export data to a file
    Export {
        /// Name of the collection/table to export
        collection: String,

        /// Output file path
        #[arg(short, long)]
        output: PathBuf,

        /// Format of the output file
        #[arg(short = 'F', long, value_enum, default_value = "json")]
        format: ExportFormat,
    },

    /// Manage collections (tables)
    #[command(subcommand)]
    Collections(CollectionsCommands),

    /// Manage indexes
    #[command(subcommand)]
    Indexes(IndexesCommands),

    /// Graph operations
    #[command(subcommand)]
    Graph(GraphCommands),
}

/// Import file formats
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum ImportFormat {
    /// JSON format
    Json,
    /// CSV format
    Csv,
}

/// Export file formats
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum ExportFormat {
    /// JSON format
    Json,
    /// CSV format
    Csv,
}

/// Collection management commands
#[derive(Subcommand, Debug)]
pub enum CollectionsCommands {
    /// List all collections
    List,

    /// Create a new collection
    Create {
        /// Name of the collection
        name: String,

        /// Path to a JSON config file defining the schema
        #[arg(short, long)]
        config: Option<PathBuf>,
    },

    /// Drop a collection
    Drop {
        /// Name of the collection to drop
        name: String,
    },

    /// Show collection information
    Info {
        /// Name of the collection
        name: String,
    },
}

/// Index management commands
#[derive(Subcommand, Debug)]
pub enum IndexesCommands {
    /// List all indexes
    List {
        /// Filter by collection name
        #[arg(short, long)]
        collection: Option<String>,
    },

    /// Rebuild an index
    Rebuild {
        /// Name of the index to rebuild
        name: String,
    },
}

/// Graph operation commands
#[derive(Subcommand, Debug)]
pub enum GraphCommands {
    /// Show graph statistics
    Stats,

    /// Traverse the graph from a starting node
    Traverse {
        /// Starting entity ID
        #[arg(long)]
        from: u64,

        /// Maximum traversal depth
        #[arg(long, default_value = "3")]
        depth: u32,

        /// Edge type filter (optional)
        #[arg(long)]
        edge_type: Option<String>,

        /// Direction: 'outgoing', 'incoming', or 'both'
        #[arg(long, default_value = "outgoing")]
        direction: TraversalDirection,
    },
}

/// Traversal direction for graph traversal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum TraversalDirection {
    /// Follow outgoing edges only.
    Outgoing,
    /// Follow incoming edges only.
    Incoming,
    /// Follow both outgoing and incoming edges.
    Both,
}

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Open { path } => open::run(&path),
        Commands::Info => info::run(cli.database.as_deref(), cli.format),
        Commands::Query { sql } => query::run(cli.database.as_deref(), &sql, cli.format),
        Commands::Execute { sql } => execute_command::run(cli.database.as_deref(), &sql),
        Commands::Repl => {
            let mut repl = Repl::new(cli.database)?;
            repl.run()
        }
        Commands::Import { file, collection, format } => {
            import_export::import(cli.database.as_deref(), &file, &collection, format)
        }
        Commands::Export { collection, output, format } => {
            import_export::export(cli.database.as_deref(), &collection, &output, format)
        }
        Commands::Collections(cmd) => collections::run(cli.database.as_deref(), cmd, cli.format),
        Commands::Indexes(cmd) => indexes::run(cli.database.as_deref(), cmd, cli.format),
        Commands::Graph(cmd) => graph::run(cli.database.as_deref(), cmd, cli.format),
    }
}
