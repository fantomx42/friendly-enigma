//! Event types and log parsing

use chrono::Local;
use serde_json::Value;

/// Log entry level
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogLevel {
    Info,
    System,
    Agent,
    Error,
    Success,
    Thought,
}

/// A single log entry
#[derive(Debug, Clone)]
pub struct LogEntry {
    pub timestamp: String,
    pub level: LogLevel,
    pub message: String,
}

impl LogEntry {
    pub fn new(level: LogLevel, message: String) -> Self {
        Self {
            timestamp: Local::now().format("%H:%M:%S").to_string(),
            level,
            message,
        }
    }

    #[allow(dead_code)]
    pub fn info(message: String) -> Self {
        Self::new(LogLevel::Info, message)
    }

    pub fn system(message: String) -> Self {
        Self::new(LogLevel::System, message)
    }

    #[allow(dead_code)]
    pub fn agent(message: String) -> Self {
        Self::new(LogLevel::Agent, message)
    }

    pub fn error(message: String) -> Self {
        Self::new(LogLevel::Error, message)
    }

    #[allow(dead_code)]
    pub fn success(message: String) -> Self {
        Self::new(LogLevel::Success, message)
    }

    pub fn thought(message: String) -> Self {
        Self::new(LogLevel::Thought, message)
    }

    /// Parse a log line and create appropriate entry
    pub fn parse(line: &str) -> Self {
        let line = line.trim();

        // Determine level based on content
        let level = if line.contains("<think>") || line.contains("</think>") {
            LogLevel::Thought
        } else if line.contains("[AGENT:") || line.contains("[Swarm]") || line.contains("[V2]") {
            LogLevel::Agent
        } else if line.contains("Error") || line.contains("ERROR") || line.contains("FAIL") {
            LogLevel::Error
        } else if line.contains("<promise>COMPLETE</promise>") || line.contains("SUCCESS") || line.contains("✅") {
            LogLevel::Success
        } else if line.starts_with("[Runner]") || line.starts_with("[Planner]") || line.starts_with("[Git]") || line.starts_with("[METRICS]") {
            LogLevel::System
        } else {
            LogLevel::Info
        };

        Self::new(level, line.to_string())
    }
}

/// Agent state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AgentState {
    #[default]
    Idle,
    Active,
}

/// Metrics from Ralph execution
#[derive(Debug, Clone, Default)]
pub struct Metrics {
    pub total_tokens: u64,
    pub last_duration_ms: u64,
    pub active_model: String,
    pub iterations: u32,
}

impl Metrics {
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self {
            total_tokens: 0,
            last_duration_ms: 0,
            active_model: String::from("qwen2.5-coder:14b"),
            iterations: 0,
        }
    }

    pub fn update_from_json(&mut self, json_str: &str) {
        if let Ok(v) = serde_json::from_str::<Value>(json_str) {
            if let Some(type_str) = v["type"].as_str() {
                match type_str {
                    "llm_call" => {
                        if let Some(prompt) = v["prompt_tokens"].as_u64() {
                            self.total_tokens += prompt;
                        }
                        if let Some(completion) = v["completion_tokens"].as_u64() {
                            self.total_tokens += completion;
                        }
                        if let Some(duration) = v["duration_ms"].as_u64() {
                            self.last_duration_ms = duration;
                        }
                        if let Some(model) = v["model"].as_str() {
                            self.active_model = model.to_string();
                        }
                    }
                    "iteration" => {
                        if let Some(iter) = v["iteration"].as_u64() {
                            self.iterations = iter as u32;
                        }
                    }
                    _ => {}
                }
            }
        }
    }
}
