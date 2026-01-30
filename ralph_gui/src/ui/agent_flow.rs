//! Agent flow visualization
//!
//! Draws a visual representation of the agent swarm with animated connections.

use egui::{Pos2, Vec2, Color32, Stroke, Painter};
use crate::app::{RalphApp, Agent};
use crate::ralph::AgentState;
use crate::theme;

/// Node radius
const NODE_RADIUS: f32 = 28.0;
const GLOW_RADIUS: f32 = 36.0;

/// Show the agent flow visualization
pub fn show(ui: &mut egui::Ui, app: &mut RalphApp) {
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

    // Update graph simulation
    app.graph.center = center;
    let dt = ui.input(|i| i.stable_dt).min(0.1);
    app.graph.update(dt);

    // Draw connections first (so they're behind nodes)
    draw_connections(&painter, app);

    // Draw nodes
    for agent in Agent::all() {
        let node = &app.graph.nodes[agent];
        draw_agent_node(&painter, node.pos, *agent, app);
    }
}

fn draw_connections(painter: &Painter, app: &RalphApp) {
    // Define connections: (from, to)
    let connections = [
        (Agent::Translator, Agent::Orchestrator),
        (Agent::Orchestrator, Agent::Engineer),
        (Agent::Engineer, Agent::Designer),
        (Agent::Designer, Agent::Asic),
    ];

    for (from, to) in connections {
        let from_pos = app.graph.nodes[&from].pos;
        let to_pos = app.graph.nodes[&to].pos;

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
    let eng_pos = app.graph.nodes[&Agent::Engineer].pos;
    let des_pos = app.graph.nodes[&Agent::Designer].pos;

    // Check if either direction is active
    let is_bidir_active = app.active_connection == Some((Agent::Engineer, Agent::Designer))
        || app.active_connection == Some((Agent::Designer, Agent::Engineer));

    let color = if is_bidir_active {
        theme::CONNECTION_ACTIVE
    } else {
        theme::CONNECTION_IDLE
    };

    // Draw bidirectional line
    painter.line_segment(
        [eng_pos, des_pos],
        Stroke::new(if is_bidir_active { 3.0 } else { 1.5 }, color),
    );

    if is_bidir_active {
        if app.active_connection == Some((Agent::Engineer, Agent::Designer)) {
            draw_arrow_head(painter, eng_pos, des_pos, color);
        } else {
            draw_arrow_head(painter, des_pos, eng_pos, color);
        }
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