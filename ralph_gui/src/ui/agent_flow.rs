//! Agent flow visualization
//!
//! Draws a visual representation of the agent swarm with animated connections.

use egui::{Pos2, Vec2, Color32, Stroke, Rect, Painter};
use crate::app::{RalphApp, Agent};
use crate::ralph::AgentState;
use crate::theme;

/// Node radius
const NODE_RADIUS: f32 = 28.0;
const GLOW_RADIUS: f32 = 36.0;

/// Show the agent flow visualization
pub fn show(ui: &mut egui::Ui, app: &RalphApp) {
    ui.horizontal(|ui| {
        ui.heading(egui::RichText::new("Agent Flow").color(theme::TEXT_PRIMARY));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.label(egui::RichText::new("○ idle  ● active").color(theme::TEXT_MUTED).small());
        });
    });

    ui.add_space(8.0);

    // Get the available rect for drawing
    let (response, painter) = ui.allocate_painter(
        Vec2::new(ui.available_width(), ui.available_height().max(200.0)),
        egui::Sense::hover(),
    );

    let rect = response.rect;
    let center = rect.center();

    // Calculate node positions
    // Layout:
    //   Translator ──► Orchestrator
    //                      │
    //                      ▼
    //   Designer  ◄──► Engineer
    //       │
    //       ▼
    //     ASICs

    let positions = calculate_positions(center, rect.width(), rect.height());

    // Draw connections first (so they're behind nodes)
    draw_connections(&painter, &positions, app);

    // Draw nodes
    for (agent, pos) in &positions {
        draw_agent_node(&painter, *pos, *agent, app);
    }
}

fn calculate_positions(center: Pos2, width: f32, height: f32) -> Vec<(Agent, Pos2)> {
    let h_spacing = (width * 0.35).min(180.0);
    let v_spacing = (height * 0.35).min(100.0);

    vec![
        (Agent::Translator, Pos2::new(center.x - h_spacing, center.y - v_spacing * 0.8)),
        (Agent::Orchestrator, Pos2::new(center.x + h_spacing * 0.3, center.y - v_spacing * 0.8)),
        (Agent::Engineer, Pos2::new(center.x + h_spacing * 0.3, center.y + v_spacing * 0.4)),
        (Agent::Designer, Pos2::new(center.x - h_spacing, center.y + v_spacing * 0.4)),
        (Agent::Asic, Pos2::new(center.x - h_spacing, center.y + v_spacing * 1.4)),
    ]
}

fn draw_connections(painter: &Painter, positions: &[(Agent, Pos2)], app: &RalphApp) {
    let get_pos = |agent: Agent| -> Pos2 {
        positions.iter().find(|(a, _)| *a == agent).map(|(_, p)| *p).unwrap_or_default()
    };

    // Define connections: (from, to)
    let connections = [
        (Agent::Translator, Agent::Orchestrator),
        (Agent::Orchestrator, Agent::Engineer),
        (Agent::Engineer, Agent::Designer),
        (Agent::Designer, Agent::Asic),
    ];

    for (from, to) in connections {
        let from_pos = get_pos(from);
        let to_pos = get_pos(to);

        let is_active = app.active_connection == Some((from, to));

        let color = if is_active {
            theme::CONNECTION_ACTIVE
        } else {
            theme::CONNECTION_IDLE
        };

        let stroke_width = if is_active { 3.0 } else { 1.5 };

        // Draw line
        painter.line_segment(
            [from_pos, to_pos],
            Stroke::new(stroke_width, color),
        );

        // Draw arrow head
        if is_active {
            draw_arrow_head(painter, from_pos, to_pos, color);
        }
    }

    // Draw bidirectional connection between Engineer and Designer
    let eng_pos = get_pos(Agent::Engineer);
    let des_pos = get_pos(Agent::Designer);

    // Check if either direction is active
    let is_bidir_active = app.active_connection == Some((Agent::Engineer, Agent::Designer))
        || app.active_connection == Some((Agent::Designer, Agent::Engineer));

    let color = if is_bidir_active {
        theme::CONNECTION_ACTIVE
    } else {
        theme::CONNECTION_IDLE
    };

    // Already drawn above, just add reverse arrow if active
    if is_bidir_active && app.active_connection == Some((Agent::Designer, Agent::Engineer)) {
        draw_arrow_head(painter, des_pos, eng_pos, color);
    }
}

fn draw_arrow_head(painter: &Painter, from: Pos2, to: Pos2, color: Color32) {
    let dir = (to - from).normalized();
    let arrow_size = 10.0;

    // Calculate arrow tip position (offset from node center)
    let tip = to - dir * NODE_RADIUS;

    // Perpendicular direction
    let perp = Vec2::new(-dir.y, dir.x);

    let p1 = tip - dir * arrow_size + perp * (arrow_size * 0.5);
    let p2 = tip - dir * arrow_size - perp * (arrow_size * 0.5);

    painter.add(egui::Shape::convex_polygon(
        vec![tip, p1, p2],
        color,
        Stroke::NONE,
    ));
}

fn draw_agent_node(painter: &Painter, pos: Pos2, agent: Agent, app: &RalphApp) {
    let state = app.agent_states.get(&agent).copied().unwrap_or(AgentState::Idle);
    let is_active = state == AgentState::Active;

    // Draw glow for active agents
    if is_active {
        let glow_alpha = (((app.animation_time * 3.0).sin() * 0.3 + 0.7) * 255.0) as u8;

        for i in 0..3 {
            let radius = GLOW_RADIUS + (i as f32 * 4.0);
            let alpha = glow_alpha / (i + 1) as u8;
            painter.circle_filled(
                pos,
                radius,
                Color32::from_rgba_unmultiplied(233, 69, 96, alpha / 3),
            );
        }
    }

    // Draw main circle
    let fill_color = if is_active {
        theme::AGENT_ACTIVE
    } else {
        theme::AGENT_IDLE
    };

    let stroke_color = if is_active {
        theme::AGENT_ACTIVE
    } else {
        theme::CONNECTION_IDLE
    };

    painter.circle(
        pos,
        NODE_RADIUS,
        fill_color,
        Stroke::new(2.0, stroke_color),
    );

    // Draw label below
    let label = agent.name();
    let text_pos = Pos2::new(pos.x, pos.y + NODE_RADIUS + 12.0);

    painter.text(
        text_pos,
        egui::Align2::CENTER_TOP,
        label,
        egui::FontId::proportional(13.0),
        if is_active { theme::TEXT_PRIMARY } else { theme::TEXT_SECONDARY },
    );

    // Draw status indicator inside circle
    let status_char = if is_active { "●" } else { "○" };
    painter.text(
        pos,
        egui::Align2::CENTER_CENTER,
        status_char,
        egui::FontId::proportional(16.0),
        theme::TEXT_PRIMARY,
    );
}
