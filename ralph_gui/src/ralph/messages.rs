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
