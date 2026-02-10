use axum::{routing::get, Json, Router};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;

mod ollama;

#[tokio::main]
async fn main() {
    let app = app();

    let addr = SocketAddr::from(([127, 0, 0, 1], 3000));
    println!("listening on {}", addr);
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

fn app() -> Router {
    Router::new()
        .route("/health", get(health_check))
        .route("/telemetry", get(telemetry))
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

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::Body,
        http::{Request, StatusCode},
    };
    use tower::util::ServiceExt; // for `oneshot`

    #[tokio::test]
    async fn health_check() {
        let app = app();

        let response = app
            .oneshot(Request::builder().uri("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn telemetry() {
        let app = app();

        let response = app
            .oneshot(Request::builder().uri("/telemetry").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }
}
