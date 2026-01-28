//! Task list panel

use egui::RichText;
use crate::app::{Task, TaskStatus};
use crate::theme;

pub fn show(ui: &mut egui::Ui, tasks: &[Task]) {
    ui.heading(RichText::new("Tasks").color(theme::TEXT_PRIMARY));
    ui.add_space(8.0);

    if tasks.is_empty() {
        egui::Frame::default()
            .fill(theme::BG_INPUT)
            .corner_radius(egui::CornerRadius::same(8))
            .inner_margin(12.0)
            .show(ui, |ui| {
                ui.label(RichText::new("No tasks yet").color(theme::TEXT_MUTED).italics());
            });
        return;
    }

    egui::Frame::default()
        .fill(theme::BG_INPUT)
        .corner_radius(egui::CornerRadius::same(8))
        .inner_margin(12.0)
        .show(ui, |ui| {
            for task in tasks {
                task_row(ui, task);
                ui.add_space(4.0);
            }
        });
}

fn task_row(ui: &mut egui::Ui, task: &Task) {
    ui.horizontal(|ui| {
        let (icon, color) = match task.status {
            TaskStatus::Complete => ("✓", theme::SUCCESS),
            TaskStatus::InProgress => ("●", theme::AGENT_ACTIVE),
            TaskStatus::Pending => ("○", theme::TEXT_MUTED),
        };

        ui.label(RichText::new(icon).color(color));

        let text_color = match task.status {
            TaskStatus::Complete => theme::TEXT_MUTED,
            TaskStatus::InProgress => theme::TEXT_PRIMARY,
            TaskStatus::Pending => theme::TEXT_SECONDARY,
        };

        // Truncate long descriptions
        let desc = if task.description.len() > 30 {
            format!("{}...", &task.description[..27])
        } else {
            task.description.clone()
        };

        ui.label(RichText::new(desc).color(text_color));
    });
}
