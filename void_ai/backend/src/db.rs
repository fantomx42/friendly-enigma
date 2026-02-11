use surrealdb::engine::any::{self, Any};
use surrealdb::Surreal;
use anyhow::Result;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
pub struct ProjectFile {
    pub path: String,
    pub name: String,
    pub is_dir: bool,
}

pub struct Db {
    pub client: Surreal<Any>,
}

impl Db {
    pub async fn new() -> Result<Self> {
        let client = any::connect("mem://").await?;
        client.use_ns("twai").use_db("twai").await?;
        Ok(Self { client })
    }

    pub async fn index_file(&self, file: ProjectFile) -> Result<()> {
        let _: Option<ProjectFile> = self.client
            .create(("file", &file.path))
            .content(file)
            .await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_db_init() {
        let db = Db::new().await.unwrap();
        let file = ProjectFile {
            path: "src/main.rs".to_string(),
            name: "main.rs".to_string(),
            is_dir: false,
        };
        db.index_file(file).await.unwrap();
    }
}
