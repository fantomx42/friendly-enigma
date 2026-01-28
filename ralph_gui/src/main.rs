//! Ralph AI - Native GUI Dashboard
//!
//! A modern dashboard for visualizing Ralph AI agent activity in real-time.

mod app;
mod theme;
mod ui;
mod ralph;

use app::RalphApp;

fn main() -> eframe::Result<()> {
    // Set up native window options
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 800.0])
            .with_min_inner_size([800.0, 600.0])
            .with_title("Ralph AI"),
        ..Default::default()
    };

    // Run the app
    eframe::run_native(
        "Ralph AI",
        options,
        Box::new(|cc| {
            // Set up custom fonts and style
            let mut style = (*cc.egui_ctx.style()).clone();
            style.visuals = theme::dark_visuals();
            cc.egui_ctx.set_style(style);

            Ok(Box::new(RalphApp::new(cc)))
        }),
    )
}
