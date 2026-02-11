use axum::{routing::get, Json, Router};
use std::path::Path;
use serde::{Deserialize, Serialize};
use anyhow::Result;
use crate::fs;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FileNode {
    pub name: String,
    pub is_dir: bool,
    pub children: Option<Vec<FileNode>>,
}

pub fn router() -> Router<std::sync::Arc<crate::AppState>> {
    Router::new().route("/map", get(get_project_map))
}

async fn get_project_map() -> Json<FileNode> {
    let root = scan_path(".").await.unwrap_or(FileNode {
        name: "root".to_string(),
        is_dir: true,
        children: Some(vec![]),
    });
    Json(root)
}

async fn scan_path(path: impl AsRef<Path>) -> Result<FileNode> {
    let name = path.as_ref().file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| ".".to_string());
    
    let is_dir = path.as_ref().is_dir();
    let mut children = None;

    if is_dir {
        let mut nodes = Vec::new();
        let entries = fs::list_dir(&path).await?;
        for entry in entries {
            // Ignore common noise folders to keep the map clean for you
            if entry == "target" || entry == ".git" || entry == "node_modules" {
                continue;
            }
            let child_path = path.as_ref().join(entry);
            if let Ok(node) = Box::pin(scan_path(child_path)).await {
                nodes.push(node);
            }
        }
        children = Some(nodes);
    }

    Ok(FileNode { name, is_dir, children })
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::Body,
        http::{Request, StatusCode},
    };
    use tower::util::ServiceExt;
    use std::sync::Arc;
    use crate::AppState;
    use crate::ollama::OllamaClient;
    use crate::db::Db;

    #[tokio::test]
    async fn test_get_map() {
        let db = Db::new().await.unwrap();
        let state = Arc::new(AppState {
            ollama: OllamaClient::new("http://localhost:11434".to_string()),
            db,
        });
        let app = crate::app(state);

        let response = app
            .oneshot(Request::builder().uri("/api/map").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }
}
