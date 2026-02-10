use serde::{Deserialize, Serialize};
use anyhow::Result;
use futures_util::stream::BoxStream;
use futures_util::StreamExt;
use bytes::Bytes;

pub struct OllamaClient {
    base_url: String,
    client: reqwest::Client,
}

#[derive(Serialize)]
struct GenerateRequest {
    model: String,
    prompt: String,
    stream: bool,
}

#[derive(Deserialize, Debug, Serialize, Clone)]
pub struct GenerateResponse {
    pub response: String,
    pub done: bool,
}

impl OllamaClient {
    pub fn new(base_url: String) -> Self {
        Self {
            base_url,
            client: reqwest::Client::new(),
        }
    }

    pub async fn generate(&self, model: &str, prompt: &str) -> Result<GenerateResponse> {
        let url = format!("{}/api/generate", self.base_url);
        let request = GenerateRequest {
            model: model.to_string(),
            prompt: prompt.to_string(),
            stream: false,
        };

        let response = self.client.post(url)
            .json(&request)
            .send()
            .await?
            .json::<GenerateResponse>()
            .await?;

        Ok(response)
    }

    pub async fn generate_stream(&self, model: String, prompt: String) -> Result<BoxStream<'static, Result<Bytes, reqwest::Error>>> {
        let url = format!("{}/api/generate", self.base_url);
        let request = GenerateRequest {
            model,
            prompt,
            stream: true,
        };

        let response = self.client.post(url)
            .json(&request)
            .send()
            .await?;

        Ok(response.bytes_stream().boxed())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_new_client() {
        let client = OllamaClient::new("http://localhost:11434".to_string());
        assert_eq!(client.base_url, "http://localhost:11434");
    }

    #[tokio::test]
    async fn test_generate_failure() {
        let client = OllamaClient::new("http://localhost:11111".to_string());
        let result = client.generate("llama3", "hello").await;
        assert!(result.is_err());
    }
}
