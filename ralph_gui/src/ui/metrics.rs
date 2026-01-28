//! Metrics sidebar panel

use egui::RichText;
use crate::ralph::Metrics;
use crate::theme;

pub fn show(ui: &mut egui::Ui, metrics: &Metrics) {
    ui.heading(RichText::new("Metrics").color(theme::TEXT_PRIMARY));
    ui.add_space(8.0);

    egui::Frame::default()
        .fill(theme::BG_INPUT)
        .corner_radius(egui::CornerRadius::same(8))
        .inner_margin(12.0)
        .show(ui, |ui| {
            ui.vertical(|ui| {
                metric_row(ui, "Tokens", &format_number(metrics.total_tokens));
                ui.add_space(4.0);
                metric_row(ui, "Duration", &format_duration(metrics.last_duration_ms));
                ui.add_space(4.0);
                metric_row(ui, "Model", &metrics.active_model);
                ui.add_space(4.0);
                metric_row(ui, "Iterations", &metrics.iterations.to_string());
            });
        });
}

fn metric_row(ui: &mut egui::Ui, label: &str, value: &str) {
    ui.horizontal(|ui| {
        ui.label(RichText::new(label).color(theme::TEXT_MUTED).small());
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.label(RichText::new(value).color(theme::TEXT_PRIMARY).strong());
        });
    });
}

fn format_number(n: u64) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}k", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}

fn format_duration(ms: u64) -> String {
    if ms >= 60_000 {
        format!("{:.1}m", ms as f64 / 60_000.0)
    } else if ms >= 1_000 {
        format!("{:.1}s", ms as f64 / 1_000.0)
    } else {
        format!("{}ms", ms)
    }
}
