#[cfg(test)]
mod tests {
    use crate::app::{RalphApp, Agent};

    #[test]
    fn test_app_initial_state() {
        let app = RalphApp::default();
        
        // Existing state
        assert_eq!(app.objective_input, "");
        assert!(app.logs.is_empty());
        
        // New control state placeholders
        assert!(!app.is_paused);
        assert!(!app.sandbox_enabled);
        
        for agent in Agent::all() {
            assert!(app.enabled_agents.get(agent).copied().unwrap_or(false));
        }
    }

    #[test]
    fn test_graph_update() {
        use crate::ui::graph::ForceGraph;
        use egui::Pos2;
        
        let mut graph = ForceGraph::new(Pos2::new(0.0, 0.0));
        let initial_pos = graph.nodes[&Agent::Orchestrator].pos;
        
        graph.update(0.1);
        let new_pos = graph.nodes[&Agent::Orchestrator].pos;
        
        // Positions should have moved due to repulsion
        assert_ne!(initial_pos, new_pos);
    }

    #[test]
    fn test_message_parsing() {
        use crate::ralph::messages::{Message, MessageType};
        
        let json = r#"{
            "id": "abc12345",
            "type": "work_request",
            "sender": "orchestrator",
            "receiver": "engineer",
            "payload": {"plan": "Do something"},
            "timestamp": "2026-01-29T12:00:00Z"
        }"#;
        
        let msg: Message = serde_json::from_str(json).expect("Failed to parse message");
        assert_eq!(msg.id, "abc12345");
        assert_eq!(msg.msg_type, MessageType::WorkRequest);
        assert_eq!(msg.sender, "orchestrator");
        assert_eq!(msg.payload["plan"], "Do something");
    }

    #[test]
    fn test_agent_params_state() {
        let mut app = RalphApp::default();
        let orchestrator = Agent::Orchestrator;
        
        // Initial state
        if let Some(params) = app.agent_params.get(&orchestrator) {
            assert_eq!(params.temperature, 0.7);
        } else {
            panic!("Orchestrator params not found");
        }
        
        // Update state
        if let Some(params) = app.agent_params.get_mut(&orchestrator) {
            params.temperature = 1.5;
        }
        
        assert_eq!(app.agent_params.get(&orchestrator).unwrap().temperature, 1.5);
    }
}