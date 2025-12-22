//! Interactive REPL implementation.

use std::borrow::Cow;
use std::path::PathBuf;
use std::time::Instant;

use manifoldb::Database;
use rustyline::completion::{Completer, Pair};
use rustyline::error::ReadlineError;
use rustyline::highlight::Highlighter;
use rustyline::hint::Hinter;
use rustyline::validate::Validator;
use rustyline::{Config, Editor, Helper};

use crate::error::{CliError, Result};
use crate::output::{format_info_table, format_query_result, InfoRow};
use crate::OutputFormat;

/// The ManifoldDB REPL.
pub struct Repl {
    db: Database,
    db_path: Option<PathBuf>,
    format: OutputFormat,
    editor: Editor<ReplHelper, rustyline::history::DefaultHistory>,
}

impl Repl {
    /// Create a new REPL.
    pub fn new(db_path: Option<PathBuf>) -> Result<Self> {
        let db = if let Some(ref path) = db_path {
            Database::open(path)?
        } else {
            // Use in-memory database if no path specified
            Database::in_memory()?
        };

        let config = Config::builder()
            .history_ignore_space(true)
            .history_ignore_dups(true)?
            .auto_add_history(true)
            .build();

        let mut editor = Editor::with_config(config)?;
        editor.set_helper(Some(ReplHelper::new()));

        // Load history
        let history_path = Self::history_path();
        if history_path.exists() {
            let _ = editor.load_history(&history_path);
        }

        Ok(Self { db, db_path, format: OutputFormat::Table, editor })
    }

    /// Get the history file path.
    fn history_path() -> PathBuf {
        dirs::data_dir().unwrap_or_else(|| PathBuf::from(".")).join("manifold").join("history.txt")
    }

    /// Run the REPL loop.
    pub fn run(&mut self) -> Result<()> {
        self.print_welcome();

        loop {
            let prompt = self.prompt();
            match self.editor.readline(&prompt) {
                Ok(line) => {
                    let line = line.trim();
                    if line.is_empty() {
                        continue;
                    }

                    if let Err(e) = self.process_line(line) {
                        eprintln!("Error: {e}");
                    }
                }
                Err(ReadlineError::Interrupted) => {
                    println!("Use .exit or Ctrl-D to exit");
                }
                Err(ReadlineError::Eof) => {
                    println!("Goodbye!");
                    break;
                }
                Err(e) => {
                    return Err(e.into());
                }
            }
        }

        // Save history
        let history_path = Self::history_path();
        if let Some(parent) = history_path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let _ = self.editor.save_history(&history_path);

        Ok(())
    }

    /// Print welcome message.
    fn print_welcome(&self) {
        println!("ManifoldDB REPL v{}", env!("CARGO_PKG_VERSION"));
        if let Some(ref path) = self.db_path {
            println!("Connected to: {}", path.display());
        } else {
            println!("Using in-memory database");
        }
        println!("Type .help for available commands");
        println!();
    }

    /// Get the prompt string.
    fn prompt(&self) -> String {
        "manifold> ".to_string()
    }

    /// Process a line of input.
    fn process_line(&mut self, line: &str) -> Result<()> {
        // Check for meta-commands
        if line.starts_with('.') {
            return self.process_meta_command(line);
        }

        // Execute as SQL
        self.execute_sql(line)
    }

    /// Process a meta-command (starting with .).
    fn process_meta_command(&mut self, line: &str) -> Result<()> {
        let parts: Vec<&str> = line.split_whitespace().collect();
        let cmd = parts.first().copied().unwrap_or("");

        match cmd {
            ".help" | ".h" => {
                self.print_help();
            }
            ".exit" | ".quit" | ".q" => {
                println!("Goodbye!");
                std::process::exit(0);
            }
            ".schema" => {
                self.show_schema(parts.get(1).copied())?;
            }
            ".tables" => {
                self.show_tables()?;
            }
            ".indexes" => {
                self.show_indexes(parts.get(1).copied())?;
            }
            ".stats" => {
                self.show_stats()?;
            }
            ".format" => {
                if let Some(fmt) = parts.get(1) {
                    self.set_format(fmt)?;
                } else {
                    println!("Current format: {:?}", self.format);
                    println!("Available: table, json, csv, compact");
                }
            }
            ".clear" => {
                // Clear screen (ANSI escape)
                print!("\x1B[2J\x1B[1;1H");
            }
            ".open" => {
                if let Some(path) = parts.get(1) {
                    self.open_database(path)?;
                } else {
                    println!("Usage: .open <path>");
                }
            }
            ".graph" => {
                self.show_graph_stats()?;
            }
            _ => {
                println!("Unknown command: {cmd}");
                println!("Type .help for available commands");
            }
        }

        Ok(())
    }

    /// Print help message.
    fn print_help(&self) {
        println!("Meta-commands:");
        println!("  .help, .h          Show this help message");
        println!("  .exit, .quit, .q   Exit the REPL");
        println!("  .schema [table]    Show schema (optionally for a specific table)");
        println!("  .tables            List all tables");
        println!("  .indexes [table]   Show indexes (optionally for a specific table)");
        println!("  .stats             Show database statistics");
        println!("  .graph             Show graph statistics");
        println!("  .format [fmt]      Get/set output format (table, json, csv, compact)");
        println!("  .open <path>       Open a different database");
        println!("  .clear             Clear the screen");
        println!();
        println!("SQL commands:");
        println!("  SELECT ...         Execute a query");
        println!("  INSERT ...         Insert data");
        println!("  UPDATE ...         Update data");
        println!("  DELETE ...         Delete data");
        println!("  CREATE TABLE ...   Create a table");
        println!("  DROP TABLE ...     Drop a table");
        println!("  CREATE INDEX ...   Create an index");
        println!();
    }

    /// Execute a SQL statement.
    fn execute_sql(&self, sql: &str) -> Result<()> {
        let start = Instant::now();

        // Determine if this is a query or statement
        let sql_upper = sql.trim().to_uppercase();
        let is_query = sql_upper.starts_with("SELECT")
            || sql_upper.starts_with("WITH")
            || sql_upper.starts_with("SHOW");

        if is_query {
            let result = self.db.query(sql)?;
            let elapsed = start.elapsed();

            let output = format_query_result(&result, self.format)?;
            println!("{output}");
            println!("Time: {:.3}ms", elapsed.as_secs_f64() * 1000.0);
        } else {
            let affected = self.db.execute(sql)?;
            let elapsed = start.elapsed();

            println!("{affected} row(s) affected");
            println!("Time: {:.3}ms", elapsed.as_secs_f64() * 1000.0);
        }

        Ok(())
    }

    /// Show schema information.
    fn show_schema(&self, table: Option<&str>) -> Result<()> {
        let sql = if let Some(table) = table {
            format!(
                "SELECT column_name, data_type, is_nullable \
                 FROM information_schema.columns \
                 WHERE table_name = '{table}'"
            )
        } else {
            "SELECT table_name, column_name, data_type \
             FROM information_schema.columns \
             ORDER BY table_name, ordinal_position"
                .to_string()
        };

        let result = self.db.query(&sql)?;
        let output = format_query_result(&result, self.format)?;
        println!("{output}");
        Ok(())
    }

    /// Show all tables.
    fn show_tables(&self) -> Result<()> {
        let result = self.db.query("SELECT table_name FROM information_schema.tables")?;
        let output = format_query_result(&result, self.format)?;
        println!("{output}");
        Ok(())
    }

    /// Show indexes.
    fn show_indexes(&self, table: Option<&str>) -> Result<()> {
        let sql = if let Some(table) = table {
            format!(
                "SELECT index_name, column_name, index_type \
                 FROM information_schema.indexes \
                 WHERE table_name = '{table}'"
            )
        } else {
            "SELECT table_name, index_name, column_name, index_type \
             FROM information_schema.indexes"
                .to_string()
        };

        let result = self.db.query(&sql)?;
        let output = format_query_result(&result, self.format)?;
        println!("{output}");
        Ok(())
    }

    /// Show database statistics.
    fn show_stats(&self) -> Result<()> {
        let metrics = self.db.metrics();

        let successful = metrics.queries.total_queries - metrics.queries.failed_queries;
        let total_transactions = metrics.transactions.commits + metrics.transactions.rollbacks;

        let rows = vec![
            InfoRow {
                key: "Total Queries".to_string(),
                value: metrics.queries.total_queries.to_string(),
            },
            InfoRow { key: "Successful Queries".to_string(), value: successful.to_string() },
            InfoRow {
                key: "Failed Queries".to_string(),
                value: metrics.queries.failed_queries.to_string(),
            },
            InfoRow {
                key: "Total Transactions".to_string(),
                value: total_transactions.to_string(),
            },
            InfoRow { key: "Commits".to_string(), value: metrics.transactions.commits.to_string() },
            InfoRow {
                key: "Rollbacks".to_string(),
                value: metrics.transactions.rollbacks.to_string(),
            },
        ];

        println!("{}", format_info_table(rows));
        Ok(())
    }

    /// Show graph statistics.
    fn show_graph_stats(&self) -> Result<()> {
        let tx = self.db.begin_read()?;
        let entity_count = tx.count_entities(None)?;

        let mut edge_count = 0u64;
        for entity in tx.iter_entities(None)? {
            edge_count += tx.get_outgoing_edges(entity.id)?.len() as u64;
        }

        let rows = vec![
            InfoRow { key: "Total Entities".to_string(), value: entity_count.to_string() },
            InfoRow { key: "Total Edges".to_string(), value: edge_count.to_string() },
        ];

        println!("{}", format_info_table(rows));
        Ok(())
    }

    /// Set the output format.
    fn set_format(&mut self, fmt: &str) -> Result<()> {
        self.format = match fmt.to_lowercase().as_str() {
            "table" => OutputFormat::Table,
            "json" => OutputFormat::Json,
            "csv" => OutputFormat::Csv,
            "compact" => OutputFormat::Compact,
            _ => {
                return Err(CliError::InvalidInput(format!(
                    "Unknown format: {fmt}. Available: table, json, csv, compact"
                )));
            }
        };
        println!("Output format set to: {:?}", self.format);
        Ok(())
    }

    /// Open a different database.
    fn open_database(&mut self, path: &str) -> Result<()> {
        let path = PathBuf::from(path);
        self.db = Database::open(&path)?;
        self.db_path = Some(path.clone());
        println!("Opened database: {}", path.display());
        Ok(())
    }
}

/// REPL helper for completion, hints, and highlighting.
struct ReplHelper {
    keywords: Vec<String>,
}

impl ReplHelper {
    fn new() -> Self {
        let keywords = vec![
            // SQL keywords
            "SELECT",
            "FROM",
            "WHERE",
            "INSERT",
            "INTO",
            "VALUES",
            "UPDATE",
            "SET",
            "DELETE",
            "CREATE",
            "TABLE",
            "INDEX",
            "DROP",
            "ALTER",
            "AND",
            "OR",
            "NOT",
            "NULL",
            "IS",
            "IN",
            "LIKE",
            "ORDER",
            "BY",
            "ASC",
            "DESC",
            "LIMIT",
            "OFFSET",
            "JOIN",
            "LEFT",
            "RIGHT",
            "INNER",
            "OUTER",
            "ON",
            "GROUP",
            "HAVING",
            "DISTINCT",
            "COUNT",
            "SUM",
            "AVG",
            "MIN",
            "MAX",
            "AS",
            "PRIMARY",
            "KEY",
            "FOREIGN",
            "REFERENCES",
            "UNIQUE",
            "DEFAULT",
            "CHECK",
            "CONSTRAINT",
            // Data types
            "INTEGER",
            "TEXT",
            "REAL",
            "BLOB",
            "BOOLEAN",
            "VECTOR",
            "VARCHAR",
            "BIGINT",
            // Meta-commands
            ".help",
            ".exit",
            ".quit",
            ".schema",
            ".tables",
            ".indexes",
            ".stats",
            ".format",
            ".clear",
            ".open",
            ".graph",
        ]
        .into_iter()
        .map(String::from)
        .collect();

        Self { keywords }
    }
}

impl Completer for ReplHelper {
    type Candidate = Pair;

    fn complete(
        &self,
        line: &str,
        pos: usize,
        _ctx: &rustyline::Context<'_>,
    ) -> std::result::Result<(usize, Vec<Pair>), ReadlineError> {
        // Get the word being typed
        let start = line[..pos]
            .rfind(|c: char| c.is_whitespace() || c == '(' || c == ',')
            .map(|i| i + 1)
            .unwrap_or(0);

        let word = &line[start..pos];
        let word_upper = word.to_uppercase();

        // Find matching completions
        let completions: Vec<Pair> = self
            .keywords
            .iter()
            .filter(|kw| kw.to_uppercase().starts_with(&word_upper))
            .map(|kw| {
                let display = if word.chars().next().is_some_and(|c| c.is_lowercase()) {
                    kw.to_lowercase()
                } else {
                    kw.clone()
                };
                Pair { display: display.clone(), replacement: display }
            })
            .collect();

        Ok((start, completions))
    }
}

impl Hinter for ReplHelper {
    type Hint = String;

    fn hint(&self, _line: &str, _pos: usize, _ctx: &rustyline::Context<'_>) -> Option<String> {
        None
    }
}

impl Highlighter for ReplHelper {
    fn highlight<'l>(&self, line: &'l str, _pos: usize) -> Cow<'l, str> {
        // Simple keyword highlighting could be added here
        Cow::Borrowed(line)
    }

    fn highlight_prompt<'b, 's: 'b, 'p: 'b>(
        &'s self,
        prompt: &'p str,
        _default: bool,
    ) -> Cow<'b, str> {
        Cow::Borrowed(prompt)
    }

    fn highlight_hint<'h>(&self, hint: &'h str) -> Cow<'h, str> {
        Cow::Borrowed(hint)
    }

    fn highlight_char(&self, _line: &str, _pos: usize, _forced: bool) -> bool {
        false
    }
}

impl Validator for ReplHelper {}

impl Helper for ReplHelper {}
