use axum::{
    extract::State,
    response::sse::{Event, Sse},
    routing::{get, post},
    Json, Router,
};
use futures_util::stream::Stream;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::Arc;

mod ollama;
use ollama::OllamaClient;

struct AppState {
    ollama: OllamaClient,
}

#[tokio::main]
async fn main() {
    let state = Arc::new(AppState {
        ollama: OllamaClient::new("http://localhost:11434".to_string()),
    });

    let app = app(state);

    let addr = SocketAddr::from(([127, 0, 0, 1], 3000));
    println!("listening on {}", addr);
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

fn app(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(health_check))
        .route("/telemetry", get(telemetry))
        .route("/chat", post(chat))
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

async fn chat(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<ChatRequest>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let stream = state
        .ollama
        .generate_stream(payload.model, payload.prompt)
        .await
        .unwrap();

    let sse_stream = stream.map(|result| {
        match result {
            Ok(bytes) => {
                let text = String::from_utf8_lossy(&bytes).to_string();
                Ok(Event::default().data(text))
            }
            Err(e) => {
                Ok(Event::default().event("error").data(e.to_string()))
            }
        }
    });

    Sse::new(sse_stream)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::Body,
        http::{Request, StatusCode},
    };
    use tower::util::ServiceExt; // for `oneshot`

    fn test_app() -> Router {
        let state = Arc::new(AppState {
            ollama: OllamaClient::new("http://localhost:11434".to_string()),
        });
        app(state)
    }

    #[tokio::test]
    async fn health_check() {
        let app = test_app();

        let response = app
            .oneshot(Request::builder().uri("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn telemetry() {
        let app = test_app();

        let response = app
            .oneshot(Request::builder().uri("/telemetry").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }
}
