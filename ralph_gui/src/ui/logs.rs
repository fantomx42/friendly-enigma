//! Log viewer panel

use egui::{RichText, ScrollArea};
use crate::app::RalphApp;
use crate::ralph::{LogEntry, LogLevel};
use crate::theme;

pub fn show(ui: &mut egui::Ui, app: &mut RalphApp) {
    ui.horizontal(|ui| {
        ui.heading(RichText::new("Logs").color(theme::TEXT_PRIMARY));

        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui.button("Clear").clicked() {
                app.logs.clear();
            }

            ui.checkbox(&mut app.show_system_logs, "System");
        });
    });

    ui.add_space(8.0);

    ScrollArea::vertical()
        .auto_shrink([false, false])
        .stick_to_bottom(true)
        .show(ui, |ui| {
            for entry in app.logs.iter() {
                // Filter system logs if not showing them
                if !app.show_system_logs && entry.level == LogLevel::System {
                    continue;
                }

                log_entry_row(ui, entry);
            }
        });
}

fn log_entry_row(ui: &mut egui::Ui, entry: &LogEntry) {
    ui.horizontal(|ui| {
        // Timestamp
        ui.label(
            RichText::new(&entry.timestamp)
                .color(theme::TEXT_MUTED)
                .small()
                .monospace(),
        );

        // Level indicator
        let (level_text, level_color) = match entry.level {
            LogLevel::Info => ("INFO", theme::TEXT_SECONDARY),
            LogLevel::System => ("SYS", theme::TEXT_MUTED),
            LogLevel::Agent => ("AGENT", theme::ACCENT),
            LogLevel::Error => ("ERR", theme::ERROR),
            LogLevel::Success => ("OK", theme::SUCCESS),
            LogLevel::Thought => ("THINK", theme::TEXT_MUTED),
        };

        ui.label(RichText::new(level_text).color(level_color).small().monospace());

        // Message
        let msg_color = match entry.level {
            LogLevel::Error => theme::ERROR,
            LogLevel::Success => theme::SUCCESS,
            LogLevel::Agent => theme::AGENT_ACTIVE,
            LogLevel::Thought => theme::TEXT_SECONDARY,
            _ => theme::TEXT_PRIMARY,
        };

        let mut text = RichText::new(&entry.message).color(msg_color);
        if entry.level == LogLevel::Thought {
            text = text.italics();
        }

        ui.label(text);
    });
}
