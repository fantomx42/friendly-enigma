//! Main application state and update loop

use eframe::egui;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use crossbeam_channel::{Receiver, Sender, unbounded};
use serde_json::Value;

use crate::ralph::{RalphRunner, AgentState, Metrics, LogEntry};
use crate::ui;
use crate::theme;

/// Maximum number of log entries to keep
const MAX_LOG_ENTRIES: usize = 500;

/// Agent names in the swarm
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Agent {
    Translator,
    Orchestrator,
    Engineer,
    Designer,
    Asic,
}

impl Agent {
    pub fn all() -> &'static [Agent] {
        &[
            Agent::Translator,
            Agent::Orchestrator,
            Agent::Engineer,
            Agent::Designer,
            Agent::Asic,
        ]
    }

    pub fn name(&self) -> &'static str {
        match self {
            Agent::Translator => "Translator",
            Agent::Orchestrator => "Orchestrator",
            Agent::Engineer => "Engineer",
            Agent::Designer => "Designer",
            Agent::Asic => "ASICs",
        }
    }
}

/// Task in the plan
#[derive(Debug, Clone)]
pub struct Task {
    pub id: usize,
    pub description: String,
    pub status: TaskStatus,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TaskStatus {
    Pending,
    InProgress,
    Complete,
}

/// Main application state
pub struct RalphApp {
    // Input
    pub objective_input: String,

    // Agent states
    pub agent_states: std::collections::HashMap<Agent, AgentState>,
    pub active_connection: Option<(Agent, Agent)>,

    // Logs
    pub logs: VecDeque<LogEntry>,
    pub show_system_logs: bool,

    // Thinking
    pub current_thought: String,
    pub is_thinking: bool,

    // Metrics
    pub metrics: Metrics,

    // Tasks
    pub tasks: Vec<Task>,

    // Runner
    runner: Option<RalphRunner>,
    log_receiver: Option<Receiver<LogEntry>>,

    // Animation state
    pub animation_time: f32,
}

impl Default for RalphApp {
    fn default() -> Self {
        let mut agent_states = std::collections::HashMap::new();
        for agent in Agent::all() {
            agent_states.insert(*agent, AgentState::Idle);
        }

        Self {
            objective_input: String::new(),
            agent_states,
            active_connection: None,
            logs: VecDeque::new(),
            show_system_logs: false,
            current_thought: String::new(),
            is_thinking: false,
            metrics: Metrics::default(),
            tasks: Vec::new(),
            runner: None,
            log_receiver: None,
            animation_time: 0.0,
        }
    }
}

impl RalphApp {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        Self::default()
    }

    /// Start a new Ralph run with the given objective
    pub fn start_run(&mut self, objective: String) {
        // Clear previous state
        self.logs.clear();
        self.tasks.clear();
        self.current_thought.clear();
        self.is_thinking = false;
        for state in self.agent_states.values_mut() {
            *state = AgentState::Idle;
        }
        self.active_connection = None;

        // Create runner
        let (sender, receiver) = unbounded();
        self.log_receiver = Some(receiver);

        let runner = RalphRunner::new(objective, sender);
        if let Err(e) = runner.start() {
            self.add_log(LogEntry::error(format!("Failed to start: {}", e)));
        } else {
            self.runner = Some(runner);
            self.add_log(LogEntry::system("Starting Ralph...".to_string()));
        }
    }

    /// Add a log entry
    pub fn add_log(&mut self, entry: LogEntry) {
        self.logs.push_back(entry);
        if self.logs.len() > MAX_LOG_ENTRIES {
            self.logs.pop_front();
        }
    }

    /// Process incoming log messages and update state
    fn process_messages(&mut self) {
        let receiver = if let Some(ref r) = self.log_receiver {
            r.clone()
        } else {
            return;
        };

        // Process all pending messages
        while let Ok(entry) = receiver.try_recv() {
            let message = entry.message.clone();

            // Handle thinking blocks
            if message.contains("<think>") {
                self.is_thinking = true;
                self.current_thought.clear();
                // If the line contains more than just the tag, capture it
                let parts: Vec<&str> = message.split("<think>").collect();
                if parts.len() > 1 {
                    self.current_thought.push_str(parts[1]);
                }
                continue;
            } else if message.contains("</think>") {
                self.is_thinking = false;
                // If the line contains more than just the tag, capture it
                let parts: Vec<&str> = message.split("</think>").collect();
                if !parts[0].is_empty() {
                    self.current_thought.push_str(parts[0]);
                }
                // Add the complete thought to logs
                self.add_log(LogEntry::thought(self.current_thought.clone()));
                continue;
            }

            if self.is_thinking {
                self.current_thought.push_str(&message);
                self.current_thought.push('\n');
                continue;
            }

            // Parse agent events from log
            self.parse_agent_event(&entry);

            // Parse metrics
            if entry.message.starts_with("[METRICS]") {
                let json_part = entry.message.trim_start_matches("[METRICS]").trim();
                self.metrics.update_from_json(json_part);
            }

            // Parse plan
            if entry.message.starts_with("[PLAN]") {
                let json_part = entry.message.trim_start_matches("[PLAN]").trim();
                if let Ok(v) = serde_json::from_str::<Value>(json_part) {
                    if let Some(tasks_val) = v["tasks"].as_array() {
                        self.tasks = tasks_val.iter().map(|t| {
                            let id = t["id"].as_u64().unwrap_or(0) as usize;
                            let description = t["description"].as_str().unwrap_or("").to_string();
                            let status_str = t["status"].as_str().unwrap_or("pending");
                            let status = match status_str {
                                "complete" => TaskStatus::Complete,
                                "in_progress" => TaskStatus::InProgress,
                                _ => TaskStatus::Pending,
                            };
                            Task { id, description, status }
                        }).collect();
                    }
                }
            }

            self.add_log(entry);
        }
    }

    /// Parse agent state changes from log entries
    fn parse_agent_event(&mut self, entry: &LogEntry) {
        let text = &entry.message;

        // Check for agent markers [AGENT:NAME:START/END]
        if text.contains("[AGENT:") {
            if text.contains(":START]") {
                if text.contains("ORCHESTRATOR") {
                    self.set_agent_active(Agent::Orchestrator);
                    self.active_connection = Some((Agent::Translator, Agent::Orchestrator));
                } else if text.contains("ENGINEER") {
                    self.set_agent_active(Agent::Engineer);
                    self.active_connection = Some((Agent::Orchestrator, Agent::Engineer));
                } else if text.contains("DESIGNER") {
                    self.set_agent_active(Agent::Designer);
                    self.active_connection = Some((Agent::Engineer, Agent::Designer));
                } else if text.contains("TRANSLATOR") {
                    self.set_agent_active(Agent::Translator);
                    self.active_connection = None;
                }
            } else if text.contains(":END]") {
                // Reset agent to idle (the next START will activate another)
                if text.contains("ORCHESTRATOR") {
                    self.agent_states.insert(Agent::Orchestrator, AgentState::Idle);
                } else if text.contains("ENGINEER") {
                    self.agent_states.insert(Agent::Engineer, AgentState::Idle);
                } else if text.contains("DESIGNER") {
                    self.agent_states.insert(Agent::Designer, AgentState::Idle);
                } else if text.contains("TRANSLATOR") {
                    self.agent_states.insert(Agent::Translator, AgentState::Idle);
                }
            }
        }

        // Also check for legacy patterns
        if text.contains("[Swarm] Orchestrator is thinking") {
            self.set_agent_active(Agent::Orchestrator);
            self.active_connection = Some((Agent::Translator, Agent::Orchestrator));
        } else if text.contains("[Swarm] Engineer is coding") {
            self.set_agent_active(Agent::Engineer);
            self.active_connection = Some((Agent::Orchestrator, Agent::Engineer));
        } else if text.contains("[Swarm] Designer is reviewing") {
            self.set_agent_active(Agent::Designer);
            self.active_connection = Some((Agent::Engineer, Agent::Designer));
        } else if text.contains("[V2] Translator processing") {
            self.set_agent_active(Agent::Translator);
        } else if text.contains("[V2] Spawning ASIC") || text.contains("ASIC:") {
            self.set_agent_active(Agent::Asic);
            self.active_connection = Some((Agent::Designer, Agent::Asic));
        } else if text.contains("<promise>COMPLETE</promise>") {
            // All done - reset all agents
            for state in self.agent_states.values_mut() {
                *state = AgentState::Idle;
            }
            self.active_connection = None;
        }
    }

    fn set_agent_active(&mut self, agent: Agent) {
        // Set all to idle first
        for state in self.agent_states.values_mut() {
            *state = AgentState::Idle;
        }
        // Activate the specified agent
        self.agent_states.insert(agent, AgentState::Active);
    }

    /// Check if Ralph is currently running
    pub fn is_running(&self) -> bool {
        self.runner.as_ref().map(|r| r.is_running()).unwrap_or(false)
    }
}

impl eframe::App for RalphApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Update animation time
        self.animation_time += ctx.input(|i| i.predicted_dt);

        // Process incoming messages
        self.process_messages();

        // Request repaint for animations
        ctx.request_repaint();

        // Top panel with title
        egui::TopBottomPanel::top("header").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading(egui::RichText::new("RALPH AI").color(theme::AGENT_ACTIVE).strong());
                ui.separator();
                ui.label(egui::RichText::new("Agent Dashboard").color(theme::TEXT_SECONDARY));

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if self.is_running() {
                        ui.label(egui::RichText::new("● RUNNING").color(theme::SUCCESS));
                    } else {
                        ui.label(egui::RichText::new("○ IDLE").color(theme::TEXT_MUTED));
                    }
                });
            });
        });

        // Right sidebar with metrics and tasks
        egui::SidePanel::right("sidebar")
            .min_width(200.0)
            .max_width(280.0)
            .show(ctx, |ui| {
                ui::metrics::show(ui, &self.metrics);
                ui.add_space(16.0);
                ui::tasks::show(ui, &self.tasks);
            });

        // Bottom panel with input
        egui::TopBottomPanel::bottom("input_panel")
            .min_height(60.0)
            .show(ctx, |ui| {
                ui::input::show(ui, self);
            });

        // Central panel with agent flow and logs
        egui::CentralPanel::default().show(ctx, |ui| {
            // Agent flow visualization (top half)
            let available_height = ui.available_height();
            egui::Frame::default()
                .fill(theme::BG_CARD)
                .corner_radius(egui::CornerRadius::same(12))
                .inner_margin(16.0)
                .show(ui, |ui| {
                    ui.set_min_height(available_height * 0.4);
                    ui::agent_flow::show(ui, self);
                });

            // Central panel with agent flow, thinking, and logs
            ui.add_space(12.0);

            // Thinking (if active)
            if self.is_thinking || !self.current_thought.is_empty() {
                egui::Frame::default()
                    .fill(theme::BG_INPUT)
                    .corner_radius(egui::CornerRadius::same(8))
                    .inner_margin(12.0)
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new("Thinking...").color(theme::TEXT_MUTED).italics());
                            if self.is_thinking {
                                ui.spinner();
                            }
                        });
                        ui.add_space(4.0);
                        egui::ScrollArea::vertical()
                            .max_height(150.0)
                            .show(ui, |ui| {
                                ui.label(egui::RichText::new(&self.current_thought).color(theme::TEXT_SECONDARY).small().italics());
                            });
                    });
                ui.add_space(12.0);
            }

            // Logs (bottom half)
            egui::Frame::default()
                .fill(theme::BG_CARD)
                .corner_radius(egui::CornerRadius::same(12))
                .inner_margin(16.0)
                .show(ui, |ui| {
                    ui::logs::show(ui, self);
                });
        });
    }
}
