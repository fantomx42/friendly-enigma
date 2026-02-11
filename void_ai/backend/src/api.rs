use axum::{routing::{get, post}, Json, Router, extract::State};
use std::path::{Path, PathBuf};
use serde::{Deserialize, Serialize};
use anyhow::Result;
use crate::fs;
use std::sync::Arc;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FileNode {
    pub name: String,
    pub is_dir: bool,
    pub children: Option<Vec<FileNode>>,
}

#[derive(Deserialize)]
pub struct SetRootRequest {
    pub path: String,
}

pub fn router() -> Router<Arc<crate::AppState>> {
    Router::new()
        .route("/map", get(get_project_map))
        .route("/set_root", post(set_project_root))
}

async fn get_project_map(State(state): State<Arc<crate::AppState>>) -> Json<FileNode> {
    let root_path = state.project_root.read().await.clone();
    let root = scan_path(&root_path).await.unwrap_or(FileNode {
        name: root_path.to_string_lossy().to_string(),
        is_dir: true,
        children: Some(vec![]),
    });
    Json(root)
}

async fn set_project_root(
    State(state): State<Arc<crate::AppState>>,
    Json(payload): Json<SetRootRequest>,
) -> Json<serde_json::Value> {
    let mut root = state.project_root.write().await;
    *root = PathBuf::from(payload.path);
    Json(serde_json::json!({ "status": "ok" }))
}

async fn scan_path(path: impl AsRef<Path>) -> Result<FileNode> {
    let name = path.as_ref().file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| path.as_ref().to_string_lossy().to_string());
    
    let is_dir = path.as_ref().is_dir();
    let mut children = None;

    if is_dir {
        let mut nodes = Vec::new();
        // Limit depth to avoid massive scans
        if let Ok(entries) = fs::list_dir(&path).await {
            for entry in entries {
                if entry == "target" || entry == ".git" || entry == "node_modules" {
                    continue;
                }
                let child_path = path.as_ref().join(entry);
                // Simple depth check could be added here
                if let Ok(node) = Box::pin(scan_path(child_path)).await {
                    nodes.push(node);
                }
            }
        }
        children = Some(nodes);
    }

    Ok(FileNode { name, is_dir, children })
}