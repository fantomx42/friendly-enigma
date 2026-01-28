//! Ralph process runner
//!
//! Spawns ralph_loop.sh as a subprocess and captures output.

use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio, Child};
use std::sync::{Arc, Mutex};
use std::thread;
use crossbeam_channel::Sender;

use super::events::LogEntry;

/// Manages the Ralph subprocess
pub struct RalphRunner {
    objective: String,
    sender: Sender<LogEntry>,
    child: Arc<Mutex<Option<Child>>>,
    running: Arc<Mutex<bool>>,
}

impl RalphRunner {
    pub fn new(objective: String, sender: Sender<LogEntry>) -> Self {
        Self {
            objective,
            sender,
            child: Arc::new(Mutex::new(None)),
            running: Arc::new(Mutex::new(false)),
        }
    }

    /// Start the Ralph process
    pub fn start(&self) -> Result<(), String> {
        // Get the project directory (parent of ralph_gui)
        let project_dir = std::env::current_dir()
            .map_err(|e| e.to_string())?
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| std::env::current_dir().unwrap());

        let script_path = project_dir.join("ralph_loop.sh");

        if !script_path.exists() {
            // Try alternate location
            let alt_path = std::path::Path::new("/home/tristan/Documents/Ralph Ai/ai_tech_stack/ralph_loop.sh");
            if !alt_path.exists() {
                return Err(format!("ralph_loop.sh not found at {:?}", script_path));
            }
        }

        let mut child = Command::new("bash")
            .arg(&script_path)
            .arg("--v2")  // Force V2 mode for the GUI
            .arg(&self.objective)
            .current_dir(&project_dir)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("Failed to spawn: {}", e))?;

        // Take stdout and stderr
        let stdout = child.stdout.take().ok_or("Failed to capture stdout")?;
        let stderr = child.stderr.take().ok_or("Failed to capture stderr")?;

        // Store child handle
        *self.child.lock().unwrap() = Some(child);
        *self.running.lock().unwrap() = true;

        // Spawn thread to read stdout
        let sender_stdout = self.sender.clone();
        let running_stdout = self.running.clone();
        thread::spawn(move || {
            let reader = BufReader::new(stdout);
            for line in reader.lines().flatten() {
                if !line.trim().is_empty() {
                    let entry = LogEntry::parse(&line);
                    let _ = sender_stdout.send(entry);
                }
            }
            *running_stdout.lock().unwrap() = false;
        });

        // Spawn thread to read stderr
        let sender_stderr = self.sender.clone();
        thread::spawn(move || {
            let reader = BufReader::new(stderr);
            for line in reader.lines().flatten() {
                if !line.trim().is_empty() {
                    let entry = LogEntry::error(line);
                    let _ = sender_stderr.send(entry);
                }
            }
        });

        Ok(())
    }

    /// Check if Ralph is still running
    pub fn is_running(&self) -> bool {
        *self.running.lock().unwrap()
    }

    /// Kill the Ralph process
    pub fn kill(&self) {
        if let Some(ref mut child) = *self.child.lock().unwrap() {
            let _ = child.kill();
        }
        *self.running.lock().unwrap() = false;
    }
}

impl Drop for RalphRunner {
    fn drop(&mut self) {
        self.kill();
    }
}
