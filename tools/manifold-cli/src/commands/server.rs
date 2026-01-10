//! Server command implementations.
//!
//! Provides commands to start, stop, and manage the ManifoldDB GraphQL server.

use std::fs::{self, File};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::error::{CliError, Result};
use crate::ServerCommands;

/// Default port for the server.
const DEFAULT_PORT: u16 = 6010;

/// Maximum number of ports to try when auto-selecting.
const PORT_SCAN_RANGE: u16 = 100;

/// Server state persisted to disk.
#[derive(Debug, Serialize, Deserialize)]
struct ServerState {
    /// Process ID of the running server.
    pid: i32,
    /// Port the server is listening on.
    port: u16,
    /// Host the server is bound to.
    host: String,
    /// Path to the database file.
    database: String,
    /// Timestamp when the server was started.
    started_at: String,
}

/// Default data directory path (~/.local/share/manifoldb/).
///
/// Uses ~/.local/share/manifoldb/ on all platforms for consistency.
fn data_dir() -> PathBuf {
    dirs::home_dir().unwrap_or_else(|| PathBuf::from(".")).join(".local/share/manifoldb")
}

/// PID file path (legacy, kept for compatibility).
fn pid_file() -> PathBuf {
    data_dir().join("server.pid")
}

/// State file path.
fn state_file() -> PathBuf {
    data_dir().join("server.state")
}

/// Default database path.
fn default_database() -> PathBuf {
    data_dir().join("default.db")
}

/// Log file path.
fn log_file() -> PathBuf {
    data_dir().join("server.log")
}

/// Ensure data directory exists.
fn ensure_data_dir() -> Result<PathBuf> {
    let dir = data_dir();
    fs::create_dir_all(&dir)?;
    Ok(dir)
}

/// Write server state to disk.
fn write_state(state: &ServerState) -> Result<()> {
    let content = serde_json::to_string_pretty(state)?;
    fs::write(state_file(), content)?;
    Ok(())
}

/// Read server state from disk.
fn read_state() -> Result<ServerState> {
    let content = fs::read_to_string(state_file())?;
    let state: ServerState = serde_json::from_str(&content)?;
    Ok(state)
}

/// Delete server state file.
fn delete_state() {
    let _ = fs::remove_file(state_file());
    let _ = fs::remove_file(pid_file()); // Clean up legacy PID file too
}

/// Check if a port is available on the given host.
fn is_port_available(host: &str, port: u16) -> bool {
    TcpListener::bind((host, port)).is_ok()
}

/// Find a free port starting from the requested port.
///
/// Returns the first available port in the range [port, port + PORT_SCAN_RANGE).
/// If none are available, returns None.
fn find_free_port(host: &str, starting_port: u16) -> Option<u16> {
    for offset in 0..PORT_SCAN_RANGE {
        let port = starting_port.saturating_add(offset);
        if is_port_available(host, port) {
            return Some(port);
        }
    }
    None
}

/// Run a server subcommand.
pub fn run(database: Option<&Path>, cmd: ServerCommands) -> Result<()> {
    match cmd {
        ServerCommands::Start { host, port, background, quiet } => {
            start(database, &host, port, background, quiet)
        }
        ServerCommands::Stop { quiet } => stop(quiet),
        ServerCommands::Status { json } => status(json),
        ServerCommands::Restart { host, port, quiet } => restart(database, &host, port, quiet),
        ServerCommands::Install { host, port } => install(database, &host, port),
        ServerCommands::Uninstall => uninstall(),
    }
}

/// Start the server.
fn start(
    database: Option<&Path>,
    host: &str,
    port: u16,
    background: bool,
    quiet: bool,
) -> Result<()> {
    ensure_data_dir()?;

    // Check if already running
    if is_running()? {
        if !quiet {
            eprintln!("Server is already running");
        }
        return Err(CliError::ServerAlreadyRunning);
    }

    // Find a free port (auto-select if requested port is in use)
    let actual_port = if is_port_available(host, port) {
        port
    } else {
        if !quiet {
            eprintln!("Port {} is in use, searching for available port...", port);
        }
        find_free_port(host, port).ok_or_else(|| {
            CliError::InvalidInput(format!(
                "No available port found in range {}-{}",
                port,
                port.saturating_add(PORT_SCAN_RANGE)
            ))
        })?
    };

    if actual_port != port && !quiet {
        println!("Using port {} (requested {} was in use)", actual_port, port);
    }

    let db_path = database.map(PathBuf::from).unwrap_or_else(default_database);

    if background {
        start_background(&db_path, host, actual_port, quiet)
    } else {
        start_foreground(&db_path, host, actual_port, quiet)
    }
}

/// Start server in foreground.
fn start_foreground(database: &Path, host: &str, port: u16, quiet: bool) -> Result<()> {
    if !quiet {
        println!("Starting ManifoldDB server...");
        println!("Database: {}", database.display());
        println!("Listening on http://{}:{}", host, port);
        println!("GraphQL Playground: http://{}:{}/graphql", host, port);
        println!("Press Ctrl+C to stop");
    }

    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("manifold_server=info".parse().expect("valid directive")),
        )
        .init();

    // Create runtime and run server
    let rt = tokio::runtime::Runtime::new()?;
    let db_str = database.to_str().ok_or(CliError::InvalidPath)?.to_string();
    let host = host.to_string();

    rt.block_on(async move { manifold_server::server::run(&db_str, &host, port).await })?;

    Ok(())
}

#[cfg(unix)]
/// Start server in background (daemonize).
fn start_background(database: &Path, host: &str, port: u16, quiet: bool) -> Result<()> {
    use daemonize::Daemonize;

    let pid_path = pid_file();
    let log = log_file();

    let stdout = File::create(&log)?;
    let stderr = stdout.try_clone()?;

    let db_str = database.to_str().ok_or(CliError::InvalidPath)?.to_string();
    let host_str = host.to_string();

    if !quiet {
        println!("Starting ManifoldDB server in background...");
        println!("Database: {}", database.display());
        println!("Listening on http://{}:{}", host, port);
        println!("State file: {}", state_file().display());
        println!("Log file: {}", log.display());
    }

    let daemonize = Daemonize::new()
        .pid_file(&pid_path)
        .chown_pid_file(true)
        .working_directory(data_dir())
        .stdout(stdout)
        .stderr(stderr);

    match daemonize.start() {
        Ok(()) => {
            // Write state file with server configuration
            let pid = std::process::id() as i32;
            let state = ServerState {
                pid,
                port,
                host: host_str.clone(),
                database: db_str.clone(),
                started_at: chrono::Utc::now().to_rfc3339(),
            };
            // Best effort - don't fail if we can't write state
            let _ = write_state(&state);

            // Initialize logging in daemon
            tracing_subscriber::fmt()
                .with_env_filter(
                    tracing_subscriber::EnvFilter::try_new("manifold_server=info")
                        .expect("valid directive"),
                )
                .init();

            let rt = tokio::runtime::Runtime::new()?;
            rt.block_on(
                async move { manifold_server::server::run(&db_str, &host_str, port).await },
            )?;
            Ok(())
        }
        Err(e) => Err(CliError::Daemon(e.to_string())),
    }
}

#[cfg(not(unix))]
fn start_background(_database: &Path, _host: &str, _port: u16, _quiet: bool) -> Result<()> {
    Err(CliError::UnsupportedPlatform("Background mode is only supported on Unix".into()))
}

/// Stop the running server.
fn stop(quiet: bool) -> Result<()> {
    if !is_running()? {
        if !quiet {
            println!("Server is not running");
        }
        return Ok(());
    }

    let pid = read_pid()?;

    if !quiet {
        println!("Stopping ManifoldDB server (PID: {})...", pid);
    }

    send_signal(pid, SignalType::Term)?;

    // Wait for graceful shutdown
    for _ in 0..50 {
        if !is_process_running(pid) {
            delete_state();
            if !quiet {
                println!("Server stopped.");
            }
            return Ok(());
        }
        std::thread::sleep(Duration::from_millis(100));
    }

    // Force kill
    send_signal(pid, SignalType::Kill)?;
    delete_state();

    if !quiet {
        println!("Server forcefully terminated.");
    }

    Ok(())
}

/// Check server status.
fn status(json_output: bool) -> Result<()> {
    let running = is_running()?;
    let state = read_state().ok();

    if json_output {
        let status = if let Some(ref s) = state {
            serde_json::json!({
                "running": running,
                "pid": s.pid,
                "host": s.host,
                "port": s.port,
                "database": s.database,
                "started_at": s.started_at,
                "data_dir": data_dir().to_string_lossy(),
                "log_file": log_file().to_string_lossy(),
            })
        } else {
            serde_json::json!({
                "running": running,
                "pid": null,
                "host": null,
                "port": null,
                "database": null,
                "started_at": null,
                "data_dir": data_dir().to_string_lossy(),
                "default_database": default_database().to_string_lossy(),
                "log_file": log_file().to_string_lossy(),
            })
        };
        println!("{}", serde_json::to_string_pretty(&status)?);
    } else if running {
        if let Some(ref s) = state {
            println!("Server is running (PID: {})", s.pid);
            println!("Listening on: http://{}:{}", s.host, s.port);
            println!("Database: {}", s.database);
            println!("Started at: {}", s.started_at);
        } else {
            println!("Server is running");
        }
        println!("Data directory: {}", data_dir().display());
        println!("Log file: {}", log_file().display());
    } else {
        println!("Server is not running");
        println!("Data directory: {}", data_dir().display());
    }

    Ok(())
}

/// Restart the server.
///
/// If the server is running, uses saved settings (port, host, database) unless
/// explicitly overridden via CLI arguments. This allows `manifold server restart`
/// to maintain the same configuration.
fn restart(database: Option<&Path>, host: &str, port: u16, quiet: bool) -> Result<()> {
    // Read saved state before stopping (if server is running)
    let saved_state = if is_running()? { read_state().ok() } else { None };

    // Stop the server if running
    if saved_state.is_some() {
        stop(quiet)?;
        std::thread::sleep(Duration::from_millis(500));
    }

    // Use saved settings as defaults, but allow CLI overrides
    // CLI defaults are "127.0.0.1" and DEFAULT_PORT, so we check against those
    let (final_host, final_port, final_db) = if let Some(ref state) = saved_state {
        // Use saved values unless CLI provided non-default values
        let use_host = if host == "127.0.0.1" { &state.host } else { host };
        let use_port = if port == DEFAULT_PORT { state.port } else { port };
        let use_db = database.map(PathBuf::from).unwrap_or_else(|| PathBuf::from(&state.database));
        (use_host.to_string(), use_port, Some(use_db))
    } else {
        (host.to_string(), port, database.map(PathBuf::from))
    };

    start(final_db.as_deref(), &final_host, final_port, true, quiet)
}

/// Install as system service.
#[cfg(target_os = "macos")]
fn install(database: Option<&Path>, host: &str, port: u16) -> Result<()> {
    ensure_data_dir()?;

    let db_path = database.map(PathBuf::from).unwrap_or_else(default_database);

    let manifest_path = std::env::current_exe()?;

    let plist = format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.manifoldb.server</string>
    <key>ProgramArguments</key>
    <array>
        <string>{}</string>
        <string>server</string>
        <string>start</string>
        <string>--database</string>
        <string>{}</string>
        <string>--host</string>
        <string>{}</string>
        <string>--port</string>
        <string>{}</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>{}</string>
    <key>StandardErrorPath</key>
    <string>{}</string>
    <key>WorkingDirectory</key>
    <string>{}</string>
</dict>
</plist>"#,
        manifest_path.display(),
        db_path.display(),
        host,
        port,
        log_file().display(),
        log_file().display(),
        data_dir().display(),
    );

    let plist_path = dirs::home_dir()
        .ok_or(CliError::NoHomeDir)?
        .join("Library/LaunchAgents/com.manifoldb.server.plist");

    // Create LaunchAgents directory if needed
    if let Some(parent) = plist_path.parent() {
        fs::create_dir_all(parent)?;
    }

    fs::write(&plist_path, plist)?;

    println!("Installed launchd service: {}", plist_path.display());
    println!();
    println!("To load now:");
    println!("  launchctl load {}", plist_path.display());
    println!();
    println!("To unload:");
    println!("  launchctl unload {}", plist_path.display());

    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn install(_database: Option<&Path>, _host: &str, _port: u16) -> Result<()> {
    Err(CliError::UnsupportedPlatform(
        "Service installation is currently only supported on macOS".into(),
    ))
}

/// Uninstall system service.
#[cfg(target_os = "macos")]
fn uninstall() -> Result<()> {
    let plist_path = dirs::home_dir()
        .ok_or(CliError::NoHomeDir)?
        .join("Library/LaunchAgents/com.manifoldb.server.plist");

    if plist_path.exists() {
        // Try to unload first
        let _ = std::process::Command::new("launchctl")
            .args(["unload", &plist_path.to_string_lossy()])
            .output();

        fs::remove_file(&plist_path)?;
        println!("Uninstalled launchd service");
    } else {
        println!("No service installed");
    }

    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn uninstall() -> Result<()> {
    Err(CliError::UnsupportedPlatform(
        "Service uninstallation is currently only supported on macOS".into(),
    ))
}

// Helper functions

fn is_running() -> Result<bool> {
    let pid_path = pid_file();
    if !pid_path.exists() {
        return Ok(false);
    }

    let pid = read_pid()?;
    Ok(is_process_running(pid))
}

fn read_pid() -> Result<i32> {
    let content = fs::read_to_string(pid_file())?;
    content.trim().parse().map_err(|_| CliError::InvalidPidFile)
}

/// Signal type for process control.
enum SignalType {
    Term,
    Kill,
}

#[cfg(unix)]
fn is_process_running(pid: i32) -> bool {
    use nix::sys::signal::kill;
    use nix::unistd::Pid;

    // Sending signal 0 checks if process exists
    kill(Pid::from_raw(pid), None).is_ok()
}

#[cfg(not(unix))]
fn is_process_running(_pid: i32) -> bool {
    false
}

#[cfg(unix)]
fn send_signal(pid: i32, signal: SignalType) -> Result<()> {
    use nix::sys::signal::{kill, Signal};
    use nix::unistd::Pid;

    let sig = match signal {
        SignalType::Term => Signal::SIGTERM,
        SignalType::Kill => Signal::SIGKILL,
    };

    kill(Pid::from_raw(pid), sig)?;
    Ok(())
}

#[cfg(not(unix))]
fn send_signal(_pid: i32, _signal: SignalType) -> Result<()> {
    Err(CliError::UnsupportedPlatform("Signal sending is only supported on Unix".into()))
}
