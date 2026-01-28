//! Objective input panel

use egui::RichText;
use crate::app::RalphApp;
use crate::theme;

pub fn show(ui: &mut egui::Ui, app: &mut RalphApp) {
    ui.horizontal(|ui| {
        // Text input
        let response = ui.add_sized(
            [ui.available_width() - 80.0, 36.0],
            egui::TextEdit::singleline(&mut app.objective_input)
                .hint_text("Enter your objective...")
                .font(egui::TextStyle::Body),
        );

        // Handle Enter key
        let enter_pressed = response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));

        // Send button
        let can_send = !app.objective_input.trim().is_empty() && !app.is_running();

        let button = egui::Button::new(
            RichText::new("Send")
                .color(if can_send { theme::TEXT_PRIMARY } else { theme::TEXT_MUTED }),
        );

        let send_clicked = ui
            .add_enabled(can_send, button)
            .clicked();

        if (enter_pressed || send_clicked) && can_send {
            let objective = app.objective_input.trim().to_string();
            app.objective_input.clear();
            app.start_run(objective);
        }
    });
}
