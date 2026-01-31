//! Visual theme for Ralph AI dashboard
//!
//! Dark cyberpunk-inspired color scheme with coral accents.

use egui::{Color32, Visuals, CornerRadius, Stroke};

// Color palette
pub const BG_DARK: Color32 = Color32::from_rgb(26, 26, 46);       // #1a1a2e
pub const BG_CARD: Color32 = Color32::from_rgb(22, 33, 62);       // #16213e
pub const BG_INPUT: Color32 = Color32::from_rgb(15, 25, 45);      // #0f192d

pub const AGENT_ACTIVE: Color32 = Color32::from_rgb(233, 69, 96);  // #e94560 coral
pub const AGENT_IDLE: Color32 = Color32::from_rgb(15, 52, 96);     // #0f3460 muted blue

pub const TEXT_PRIMARY: Color32 = Color32::from_rgb(234, 234, 234); // #eaeaea
pub const TEXT_SECONDARY: Color32 = Color32::from_rgb(150, 150, 170);
pub const TEXT_MUTED: Color32 = Color32::from_rgb(100, 100, 120);

pub const SUCCESS: Color32 = Color32::from_rgb(78, 204, 163);      // #4ecca3 teal
pub const _WARNING: Color32 = Color32::from_rgb(255, 193, 7);       // amber
pub const ERROR: Color32 = Color32::from_rgb(244, 67, 54);         // red
pub const ACCENT: Color32 = Color32::from_rgb(123, 44, 191);       // #7b2cbf purple

pub const CONNECTION_ACTIVE: Color32 = Color32::from_rgb(233, 69, 96);
pub const CONNECTION_IDLE: Color32 = Color32::from_rgb(60, 60, 80);

/// Create dark visuals for the app
pub fn dark_visuals() -> Visuals {
    let mut visuals = Visuals::dark();

    // Window and panel backgrounds
    visuals.window_fill = BG_DARK;
    visuals.panel_fill = BG_DARK;
    visuals.faint_bg_color = BG_CARD;
    visuals.extreme_bg_color = BG_INPUT;

    // Widget colors
    visuals.widgets.noninteractive.bg_fill = BG_CARD;
    visuals.widgets.noninteractive.fg_stroke = Stroke::new(1.0, TEXT_SECONDARY);
    visuals.widgets.noninteractive.corner_radius = CornerRadius::same(8);

    visuals.widgets.inactive.bg_fill = BG_CARD;
    visuals.widgets.inactive.fg_stroke = Stroke::new(1.0, TEXT_PRIMARY);
    visuals.widgets.inactive.corner_radius = CornerRadius::same(8);

    visuals.widgets.hovered.bg_fill = Color32::from_rgb(35, 50, 80);
    visuals.widgets.hovered.fg_stroke = Stroke::new(1.0, AGENT_ACTIVE);
    visuals.widgets.hovered.corner_radius = CornerRadius::same(8);

    visuals.widgets.active.bg_fill = AGENT_ACTIVE;
    visuals.widgets.active.fg_stroke = Stroke::new(1.0, TEXT_PRIMARY);
    visuals.widgets.active.corner_radius = CornerRadius::same(8);

    // Selection
    visuals.selection.bg_fill = Color32::from_rgba_unmultiplied(233, 69, 96, 80);
    visuals.selection.stroke = Stroke::new(1.0, AGENT_ACTIVE);

    // Hyperlinks
    visuals.hyperlink_color = ACCENT;

    // Window
    visuals.window_corner_radius = CornerRadius::same(12);
    visuals.window_stroke = Stroke::new(1.0, Color32::from_rgb(40, 40, 60));

    visuals
}
