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
        if ui.button("Start").clicked() {
            // Start logic
        }
        if ui.button("Pause").clicked() {
            app.is_paused = !app.is_paused;
        }
        if ui.button("Stop").clicked() {
            // Stop logic
        }
        if ui.button("Flush").clicked() {
            // Flush logic
        }
    });

    ui.add_space(16.0);

    // Sandbox Toggle
    ui.horizontal(|ui| {
        ui.label("Sandbox Mode:");
        ui.checkbox(&mut app.sandbox_enabled, "");
    });

    ui.add_space(16.0);

    // Agent Configuration
    ui.label(RichText::new("AGENTS").color(theme::TEXT_MUTED).small().strong());
    for agent in Agent::all() {
        ui.horizontal(|ui| {
            let mut enabled = app.enabled_agents.get(agent).copied().unwrap_or(true);
            if ui.checkbox(&mut enabled, agent.name()).changed() {
                app.enabled_agents.insert(*agent, enabled);
            }
        });
    }
}
