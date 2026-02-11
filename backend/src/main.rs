use axum::{
    extract::State,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;
use tower_http::cors::{Any, CorsLayer};

mod api;
mod db;
mod fs;
mod ollama;
use db::Db;
use ollama::OllamaClient;

pub struct AppState {
    pub ollama: OllamaClient,
    pub db: Db,
    pub project_root: RwLock<PathBuf>,
}

#[tokio::main]
async fn main() {
    let db = Db::new().await.unwrap();
    let project_root = RwLock::new(std::env::current_dir().unwrap());
    
    let state = Arc::new(AppState {
        ollama: OllamaClient::new("http://localhost:11434".to_string()),
        db,
        project_root,
    });

    let app = app(state);

    let addr = SocketAddr::from(([127, 0, 0, 1], 3000));
    println!("listening on {}", addr);
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

pub fn app(state: Arc<AppState>) -> Router {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    Router::new()
        .route("/health", get(health_check))
        .route("/telemetry", get(telemetry))
        .route("/chat", post(chat))
        .nest("/api", api::router())
        .layer(cors)
        .with_state(state)
}

async fn health_check() -> &'static str {
    "OK"
}

#[derive(Serialize, Deserialize)]
struct Telemetry {
    cpu_usage: f32,
    memory_usage: f32,
    gpu_usage: Option<f32>,
}

async fn telemetry() -> Json<Telemetry> {
    Json(Telemetry {
        cpu_usage: 10.5,
        memory_usage: 45.0,
        gpu_usage: Some(15.0),
    })
}

#[derive(Deserialize)]
struct ChatRequest {
    model: String,
    prompt: String,
}

#[derive(Serialize)]
struct ChatResponse {
    response: String,
}

async fn chat(
    State(_state): State<Arc<AppState>>,
    Json(payload): Json<ChatRequest>,
) -> Json<ChatResponse> {
    println!("Received chat request: {}", payload.prompt);
    
    // Simple echo for verification
    Json(ChatResponse {
        response: format!("Echo: {}", payload.prompt),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::Body,
        http::{Request, StatusCode},
    };
    use tower::util::ServiceExt;

    async fn test_app() -> Router {
        let db = Db::new().await.unwrap();
        let state = Arc::new(AppState {
            ollama: OllamaClient::new("http://localhost:11434".to_string()),
            db,
            project_root: RwLock::new(PathBuf::from(".")),
        });
        app(state)
    }

    #[tokio::test]
    async fn health_check() {
        let app = test_app().await;

        let response = app
            .oneshot(Request::builder().uri("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn telemetry() {
        let app = test_app().await;

        let response = app
            .oneshot(Request::builder().uri("/telemetry").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }
}