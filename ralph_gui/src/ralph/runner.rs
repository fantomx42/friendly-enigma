//! Ralph process runner
//!
//! Spawns ralph_loop.sh as a subprocess and captures output.

use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio, Child};
use std::sync::{Arc, Mutex};
use std::thread;
use crossbeam_channel::Sender;

use super::events::LogEntry;
use super::messages::Message;

/// Manages the Ralph subprocess
pub struct RalphRunner {
    objective: String,
    log_sender: Sender<LogEntry>,
    msg_sender: Sender<Message>,
    child: Arc<Mutex<Option<Child>>>,
    running: Arc<Mutex<bool>>,
    stdin: Arc<Mutex<Option<std::process::ChildStdin>>>,
}

impl RalphRunner {
    pub fn new(objective: String, log_sender: Sender<LogEntry>, msg_sender: Sender<Message>) -> Self {
        Self {
            objective,
            log_sender,
            msg_sender,
            child: Arc::new(Mutex::new(None)),
            running: Arc::new(Mutex::new(false)),
            stdin: Arc::new(Mutex::new(None)),
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
            .stdin(Stdio::piped())
            .spawn()
            .map_err(|e| format!("Failed to spawn: {}", e))?;

        // Take stdout, stderr, and stdin
        let stdout = child.stdout.take().ok_or("Failed to capture stdout")?;
        let stderr = child.stderr.take().ok_or("Failed to capture stderr")?;
        let stdin = child.stdin.take().ok_or("Failed to capture stdin")?;

        // Store child handle and stdin
        *self.child.lock().unwrap() = Some(child);
        *self.stdin.lock().unwrap() = Some(stdin);
        *self.running.lock().unwrap() = true;

        // Spawn thread to read stdout
        let log_sender = self.log_sender.clone();
        let msg_sender = self.msg_sender.clone();
        let running_stdout = self.running.clone();
        thread::spawn(move || {
            let reader = BufReader::new(stdout);
            for line in reader.lines().flatten() {
                if !line.trim().is_empty() {
                    if line.starts_with("[MESSAGE]") {
                        let json_part = line.trim_start_matches("[MESSAGE]").trim();
                        if let Ok(msg) = serde_json::from_str::<Message>(json_part) {
                            let _ = msg_sender.send(msg);
                        }
                    }
                    
                    let entry = LogEntry::parse(&line);
                    let _ = log_sender.send(entry);
                }
            }
            *running_stdout.lock().unwrap() = false;
        });

        // Spawn thread to read stderr
        let log_sender_err = self.log_sender.clone();
        thread::spawn(move || {
            let reader = BufReader::new(stderr);
            for line in reader.lines().flatten() {
                if !line.trim().is_empty() {
                    let entry = LogEntry::error(line);
                    let _ = log_sender_err.send(entry);
                }
            }
        });

        Ok(())
    }

    /// Check if Ralph is still running
    pub fn is_running(&self) -> bool {
        *self.running.lock().unwrap()
    }

    /// Send a message to the Ralph process
    pub fn send_message(&self, msg: &Message) -> Result<(), String> {
        if let Some(ref mut stdin) = *self.stdin.lock().unwrap() {
            let json = serde_json::to_string(msg).map_err(|e| e.to_string())?;
            writeln!(stdin, "[MESSAGE] {}", json).map_err(|e| e.to_string())?;
            stdin.flush().map_err(|e| e.to_string())?;
            Ok(())
        } else {
            Err("Stdin not available".to_string())
        }
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
