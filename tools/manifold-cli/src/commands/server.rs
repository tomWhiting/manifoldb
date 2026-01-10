//! Server command implementations.
//!
//! Provides commands to start, stop, and manage the ManifoldDB GraphQL server.

use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::error::{CliError, Result};
use crate::ServerCommands;

/// Default data directory path (~/.local/share/manifoldb/).
///
/// Uses ~/.local/share/manifoldb/ on all platforms for consistency.
fn data_dir() -> PathBuf {
    dirs::home_dir().unwrap_or_else(|| PathBuf::from(".")).join(".local/share/manifoldb")
}

/// PID file path.
fn pid_file() -> PathBuf {
    data_dir().join("server.pid")
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

    let db_path = database.map(PathBuf::from).unwrap_or_else(default_database);

    if background {
        start_background(&db_path, host, port, quiet)
    } else {
        start_foreground(&db_path, host, port, quiet)
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

    let pid = pid_file();
    let log = log_file();

    let stdout = File::create(&log)?;
    let stderr = stdout.try_clone()?;

    let db_str = database.to_str().ok_or(CliError::InvalidPath)?.to_string();
    let host = host.to_string();

    if !quiet {
        println!("Starting ManifoldDB server in background...");
        println!("Database: {}", database.display());
        println!("Listening on http://{}:{}", host, port);
        println!("PID file: {}", pid.display());
        println!("Log file: {}", log.display());
    }

    let daemonize = Daemonize::new()
        .pid_file(&pid)
        .chown_pid_file(true)
        .working_directory(data_dir())
        .stdout(stdout)
        .stderr(stderr);

    match daemonize.start() {
        Ok(()) => {
            // Initialize logging in daemon
            tracing_subscriber::fmt()
                .with_env_filter(
                    tracing_subscriber::EnvFilter::try_new("manifold_server=info")
                        .expect("valid directive"),
                )
                .init();

            let rt = tokio::runtime::Runtime::new()?;
            rt.block_on(async move { manifold_server::server::run(&db_str, &host, port).await })?;
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
            let _ = fs::remove_file(pid_file());
            if !quiet {
                println!("Server stopped.");
            }
            return Ok(());
        }
        std::thread::sleep(Duration::from_millis(100));
    }

    // Force kill
    send_signal(pid, SignalType::Kill)?;
    let _ = fs::remove_file(pid_file());

    if !quiet {
        println!("Server forcefully terminated.");
    }

    Ok(())
}

/// Check server status.
fn status(json_output: bool) -> Result<()> {
    let running = is_running()?;
    let pid = if running { read_pid().ok() } else { None };

    if json_output {
        let status = serde_json::json!({
            "running": running,
            "pid": pid,
            "pid_file": pid_file().to_string_lossy(),
            "data_dir": data_dir().to_string_lossy(),
            "default_database": default_database().to_string_lossy(),
            "log_file": log_file().to_string_lossy(),
        });
        println!("{}", serde_json::to_string_pretty(&status)?);
    } else if running {
        println!("Server is running (PID: {})", pid.unwrap_or(0));
        println!("Data directory: {}", data_dir().display());
        println!("PID file: {}", pid_file().display());
        println!("Log file: {}", log_file().display());
    } else {
        println!("Server is not running");
        println!("Data directory: {}", data_dir().display());
    }

    Ok(())
}

/// Restart the server.
fn restart(database: Option<&Path>, host: &str, port: u16, quiet: bool) -> Result<()> {
    if is_running()? {
        stop(quiet)?;
        std::thread::sleep(Duration::from_millis(500));
    }
    start(database, host, port, true, quiet)
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
