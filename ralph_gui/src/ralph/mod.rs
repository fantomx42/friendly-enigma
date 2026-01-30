//! Ralph AI integration
//!
//! Handles spawning Ralph, capturing output, and parsing events.

mod runner;
mod events;
pub mod messages;

pub use runner::RalphRunner;
pub use events::{LogEntry, LogLevel, AgentState, Metrics};
