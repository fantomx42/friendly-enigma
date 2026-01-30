//! Task Control Center

use egui::RichText;
use crate::app::{RalphApp, Agent};
use crate::theme;

pub fn show(ui: &mut egui::Ui, app: &mut RalphApp) {
    ui.heading(RichText::new("Control Center").color(theme::TEXT_PRIMARY));
    ui.add_space(8.0);

    // Lifecycle Management
    ui.label(RichText::new("LIFECYCLE").color(theme::TEXT_MUTED).small().strong());
    ui.horizontal(|ui| {
        let is_running = app.is_running();
        
        // Start button (only if not running)
        if ui.add_enabled(!is_running, egui::Button::new("Start")).clicked() {
            let objective = app.objective_input.clone();
            if !objective.trim().is_empty() {
                app.start_run(objective);
            }
        }

        // Pause button (toggle)
        let pause_label = if app.is_paused { "Resume" } else { "Pause" };
        if ui.add_enabled(is_running, egui::Button::new(pause_label)).clicked() {
            app.is_paused = !app.is_paused;
            // TODO: Send pause message to bus
        }

        // Stop button
        if ui.add_enabled(is_running, egui::Button::new("Stop")).clicked() {
            if let Some(ref runner) = app.runner() {
                runner.kill();
            }
        }

        // Flush button
        if ui.button("Flush").on_hover_text("Emergency Flush: Clear diagnostic bus").clicked() {
            // TODO: Implementation for emergency flush
            app.add_log(crate::ralph::LogEntry::system("Emergency bus flush requested".to_string()));
        }
    });

    ui.add_space(16.0);

    // Sandbox Toggle
    ui.horizontal(|ui| {
        ui.label("Sandbox Mode:");
        if ui.checkbox(&mut app.sandbox_enabled, "").on_hover_text("Run agents inside a Docker container").changed() {
             app.add_log(crate::ralph::LogEntry::system(format!("Sandbox mode toggled: {}", app.sandbox_enabled)));
        }
    });

    ui.add_space(16.0);

    // Agent Configuration
    ui.label(RichText::new("AGENTS").color(theme::TEXT_MUTED).small().strong());
    for agent in Agent::all() {
        ui.collapsing(agent.name(), |ui| {
            ui.horizontal(|ui| {
                let mut enabled = app.enabled_agents.get(agent).copied().unwrap_or(true);
                if ui.checkbox(&mut enabled, "Enabled").changed() {
                    app.enabled_agents.insert(*agent, enabled);
                }
            });

            if let Some(params) = app.agent_params.get_mut(agent) {
                ui.add(egui::Slider::new(&mut params.temperature, 0.0..=2.0).text("Temp"));
                ui.add(egui::Slider::new(&mut params.top_p, 0.0..=1.0).text("Top P"));
            }
        });
    }
}
