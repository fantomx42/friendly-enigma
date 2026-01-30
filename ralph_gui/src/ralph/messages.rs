//! Message protocol for inter-agent communication

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MessageType {
    WorkRequest,
    CodeOutput,
    RevisionReq,
    AsicRequest,
    AsicResponse,
    Options,
    Evaluation,
    Complete,
    Error,
    Status,
    Diagnostic,
    Abort,
    ForkliftRequest,
    ForkliftResponse,
    ToolRequest,
    ToolResponse,
    ToolConfirm,
    RemSleepStart,
    RemSleepComplete,
    ConsolidationRequest,
    ConsolidationResponse,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub id: String,
    #[serde(rename = "type")]
    pub msg_type: MessageType,
    pub sender: String,
    pub receiver: String,
    pub payload: Value,
    pub timestamp: String,
    pub correlation_id: Option<String>,
    #[serde(default)]
    pub metadata: Value,
}

impl Message {
    pub fn abort() -> Self {
        Self::new(MessageType::Abort, "orchestrator", serde_json::json!({}))
    }

    pub fn status(status: &str) -> Self {
        Self::new(MessageType::Status, "system", serde_json::json!({"status": status}))
    }

    fn new(msg_type: MessageType, receiver: &str, payload: Value) -> Self {
        Self {
            id: fastrand::u32(..).to_string(),
            msg_type,
            sender: "gui".to_string(),
            receiver: receiver.to_string(),
            payload,
            timestamp: chrono::Local::now().to_rfc3339(),
            correlation_id: None,
            metadata: serde_json::json!({}),
        }
    }
}
