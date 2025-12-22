//! ManifoldDB command-line interface.
//!
//! This binary provides backup and restore utilities for ManifoldDB databases.
//!
//! # Usage
//!
//! ## Full Backup
//!
//! ```bash
//! manifoldb backup --input mydb.manifold --output backup.jsonl
//! ```
//!
//! ## Restore
//!
//! ```bash
//! manifoldb restore --input backup.jsonl --output restored.manifold
//! ```
//!
//! ## Verify Backup
//!
//! ```bash
//! manifoldb verify --input backup.jsonl
//! ```

use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::PathBuf;
use std::process::ExitCode;

use manifoldb::backup::{self, BackupStatistics};
use manifoldb::Database;

/// ManifoldDB CLI arguments.
struct Args {
    command: Command,
}

enum Command {
    Backup { input: PathBuf, output: PathBuf, incremental: bool },
    Restore { input: PathBuf, output: PathBuf, fast: bool },
    Verify { input: PathBuf },
    Help,
}

fn main() -> ExitCode {
    match parse_args() {
        Ok(args) => match run(args) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("Error: {e}");
                ExitCode::FAILURE
            }
        },
        Err(msg) => {
            eprintln!("{msg}");
            ExitCode::FAILURE
        }
    }
}

fn parse_args() -> Result<Args, String> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        return Ok(Args { command: Command::Help });
    }

    match args[1].as_str() {
        "backup" => {
            let mut input = None;
            let mut output = None;
            let mut incremental = false;

            let mut i = 2;
            while i < args.len() {
                match args[i].as_str() {
                    "--input" | "-i" => {
                        i += 1;
                        input = Some(PathBuf::from(args.get(i).ok_or("Missing --input value")?));
                    }
                    "--output" | "-o" => {
                        i += 1;
                        output = Some(PathBuf::from(args.get(i).ok_or("Missing --output value")?));
                    }
                    "--incremental" => {
                        incremental = true;
                    }
                    arg => return Err(format!("Unknown argument: {arg}")),
                }
                i += 1;
            }

            let input = input.ok_or("Missing required --input argument")?;
            let output = output.ok_or("Missing required --output argument")?;

            Ok(Args { command: Command::Backup { input, output, incremental } })
        }
        "restore" => {
            let mut input = None;
            let mut output = None;
            let mut fast = false;

            let mut i = 2;
            while i < args.len() {
                match args[i].as_str() {
                    "--input" | "-i" => {
                        i += 1;
                        input = Some(PathBuf::from(args.get(i).ok_or("Missing --input value")?));
                    }
                    "--output" | "-o" => {
                        i += 1;
                        output = Some(PathBuf::from(args.get(i).ok_or("Missing --output value")?));
                    }
                    "--fast" => {
                        fast = true;
                    }
                    arg => return Err(format!("Unknown argument: {arg}")),
                }
                i += 1;
            }

            let input = input.ok_or("Missing required --input argument")?;
            let output = output.ok_or("Missing required --output argument")?;

            Ok(Args { command: Command::Restore { input, output, fast } })
        }
        "verify" => {
            let mut input = None;

            let mut i = 2;
            while i < args.len() {
                match args[i].as_str() {
                    "--input" | "-i" => {
                        i += 1;
                        input = Some(PathBuf::from(args.get(i).ok_or("Missing --input value")?));
                    }
                    arg => return Err(format!("Unknown argument: {arg}")),
                }
                i += 1;
            }

            let input = input.ok_or("Missing required --input argument")?;

            Ok(Args { command: Command::Verify { input } })
        }
        "help" | "--help" | "-h" => Ok(Args { command: Command::Help }),
        cmd => Err(format!("Unknown command: {cmd}\n\nRun 'manifoldb help' for usage.")),
    }
}

fn run(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    match args.command {
        Command::Backup { input, output, incremental } => run_backup(&input, &output, incremental),
        Command::Restore { input, output, fast } => run_restore(&input, &output, fast),
        Command::Verify { input } => run_verify(&input),
        Command::Help => {
            print_help();
            Ok(())
        }
    }
}

fn run_backup(
    input: &PathBuf,
    output: &PathBuf,
    incremental: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Opening database: {}", input.display());
    let db = Database::open(input)?;

    println!("Creating backup: {}", output.display());
    let file = File::create(output)?;
    let writer = BufWriter::new(file);

    let stats = if incremental {
        // For incremental, we'd need to track the previous sequence number
        // For now, just do a full backup with incremental metadata
        println!("Note: True incremental backup requires sequence tracking.");
        println!("      Performing full backup with incremental metadata.");
        backup::export_incremental(&db, writer, 0)?
    } else {
        backup::export_full(&db, writer)?
    };

    print_stats("Backup", &stats);
    Ok(())
}

fn run_restore(
    input: &PathBuf,
    output: &PathBuf,
    fast: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Opening backup: {}", input.display());
    let file = File::open(input)?;
    let reader = BufReader::new(file);

    println!("Creating database: {}", output.display());
    let db = Database::open(output)?;

    let importer = if fast {
        println!("Using fast import (no verification)");
        backup::Importer::with_options(reader, backup::ImportOptions::fast())
    } else {
        backup::Importer::new(reader)
    };

    let stats = importer.import_all(&db)?;

    print_stats("Restore", &stats);
    Ok(())
}

fn run_verify(input: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    println!("Verifying backup: {}", input.display());
    let file = File::open(input)?;
    let reader = BufReader::new(file);

    let stats = backup::verify(reader)?;

    print_stats("Verification", &stats);
    println!("Backup is valid.");
    Ok(())
}

fn print_stats(operation: &str, stats: &BackupStatistics) {
    println!("\n{operation} complete:");
    println!("  Entities: {}", stats.entity_count);
    println!("  Edges: {}", stats.edge_count);
    println!("  Metadata entries: {}", stats.metadata_count);
    println!("  Total records: {}", stats.total_records);
    if stats.uncompressed_size > 0 {
        println!("  Uncompressed size: {} bytes", stats.uncompressed_size);
    }
}

fn print_help() {
    println!(
        r"ManifoldDB - Backup and Restore Utility

USAGE:
    manifoldb <COMMAND> [OPTIONS]

COMMANDS:
    backup      Create a backup of a database
    restore     Restore a database from a backup
    verify      Verify a backup file without restoring
    help        Print this help message

BACKUP OPTIONS:
    --input, -i <PATH>      Path to the database to backup (required)
    --output, -o <PATH>     Path for the backup file (required)
    --incremental           Create an incremental backup (records sequence number)

RESTORE OPTIONS:
    --input, -i <PATH>      Path to the backup file (required)
    --output, -o <PATH>     Path for the restored database (required)
    --fast                  Skip verification for faster restore

VERIFY OPTIONS:
    --input, -i <PATH>      Path to the backup file to verify (required)

EXAMPLES:
    # Full backup
    manifoldb backup --input mydb.manifold --output backup.jsonl

    # Restore from backup
    manifoldb restore --input backup.jsonl --output restored.manifold

    # Verify backup integrity
    manifoldb verify --input backup.jsonl
"
    );
}
