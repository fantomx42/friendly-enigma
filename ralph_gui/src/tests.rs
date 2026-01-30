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
}