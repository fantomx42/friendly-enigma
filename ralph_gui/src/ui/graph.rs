//! Force-directed graph logic

use egui::{Pos2, Vec2};
use crate::app::Agent;
use std::collections::HashMap;

/// A node in the graph representing an agent
#[derive(Debug, Clone)]
pub struct Node {
    pub agent: Agent,
    pub pos: Pos2,
    pub vel: Vec2,
    pub mass: f32,
}

/// An edge representing communication between agents
#[derive(Debug, Clone)]
pub struct Edge {
    pub from: Agent,
    pub to: Agent,
    pub strength: f32,
    pub last_pulse: f32, // timestamp of last communication
}

/// Force-directed graph simulation
pub struct ForceGraph {
    pub nodes: HashMap<Agent, Node>,
    pub edges: Vec<Edge>,
    pub center: Pos2,
    pub config: GraphConfig,
}

pub struct GraphConfig {
    pub repulsion: f32,
    pub attraction: f32,
    pub damping: f32,
    pub ideal_length: f32,
}

impl Default for GraphConfig {
    fn default() -> Self {
        Self {
            repulsion: 1500.0,
            attraction: 0.05,
            damping: 0.95,
            ideal_length: 120.0,
        }
    }
}

impl ForceGraph {
    pub fn new(center: Pos2) -> Self {
        let mut nodes = HashMap::new();
        for agent in crate::app::Agent::all() {
            nodes.insert(*agent, Node {
                agent: *agent,
                pos: center + Vec2::new(fastrand::f32() * 10.0, fastrand::f32() * 10.0),
                vel: Vec2::ZERO,
                mass: 1.0,
            });
        }

        Self {
            nodes,
            edges: Vec::new(),
            center,
            config: GraphConfig::default(),
        }
    }

    pub fn update(&mut self, dt: f32) {
        let agents = crate::app::Agent::all();
        let mut forces: HashMap<Agent, Vec2> = agents.iter().map(|a| (*a, Vec2::ZERO)).collect();

        // 1. Repulsion between all nodes
        for i in 0..agents.len() {
            for j in i + 1..agents.len() {
                let a = agents[i];
                let b = agents[j];
                
                let pos_a = self.nodes[&a].pos;
                let pos_b = self.nodes[&b].pos;
                
                let diff = pos_a - pos_b;
                let dist_sq = diff.length_sq().max(100.0);
                let force = diff.normalized() * (self.config.repulsion / dist_sq);
                
                *forces.get_mut(&a).unwrap() += force;
                *forces.get_mut(&b).unwrap() -= force;
            }
        }

        // 2. Attraction towards center
        for agent in agents {
            let pos = self.nodes[agent].pos;
            let diff = self.center - pos;
            *forces.get_mut(agent).unwrap() += diff * 0.01;
        }

        // 3. Attraction along edges (springs)
        for edge in &self.edges {
            let pos_a = self.nodes[&edge.from].pos;
            let pos_b = self.nodes[&edge.to].pos;
            
            let diff = pos_b - pos_a;
            let dist = diff.length();
            let force = diff.normalized() * (dist - self.config.ideal_length) * self.config.attraction;
            
            *forces.get_mut(&edge.from).unwrap() += force;
            *forces.get_mut(&edge.to).unwrap() -= force;
        }

        // 4. Integrate
        for agent in agents {
            let node = self.nodes.get_mut(agent).unwrap();
            let accel = forces[agent] / node.mass;
            node.vel = (node.vel + accel * dt) * self.config.damping;
            node.pos += node.vel * dt;
        }
    }
}
