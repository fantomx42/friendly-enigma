#[cfg(test)]
mod tests {
    use crate::app::{RalphApp, Agent};

    #[test]
    fn test_app_initial_state() {
        let app = RalphApp::default();
        
        // Existing state
        assert_eq!(app.objective_input, "");
        assert!(app.logs.is_empty());
        
        // New control state placeholders (should fail to compile initially)
        assert!(!app.is_paused);
        assert!(!app.sandbox_enabled);
        
        for agent in Agent::all() {
            assert!(app.enabled_agents.get(agent).copied().unwrap_or(false));
        }
    }
}
