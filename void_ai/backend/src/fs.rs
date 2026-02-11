use std::path::Path;
use tokio::fs;
use anyhow::Result;

pub async fn read_file<P: AsRef<Path>>(path: P) -> Result<String> {
    Ok(fs::read_to_string(path).await?)
}

pub async fn write_file<P: AsRef<Path>>(path: P, content: &str) -> Result<()> {
    if let Some(parent) = path.as_ref().parent() {
        fs::create_dir_all(parent).await?;
    }
    fs::write(path, content).await?;
    Ok(())
}

pub async fn list_dir<P: AsRef<Path>>(path: P) -> Result<Vec<String>> {
    let mut entries = fs::read_dir(path).await?;
    let mut files = Vec::new();
    while let Some(entry) = entries.next_entry().await? {
        files.push(entry.file_name().to_string_lossy().to_string());
    }
    Ok(files)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_fs_ops() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.txt");
        
        write_file::<&std::path::Path>(&file_path, "hello").await.unwrap();
        let content = read_file::<&std::path::Path>(&file_path).await.unwrap();
        assert_eq!(content, "hello");

        let files = list_dir::<&std::path::Path>(dir.path()).await.unwrap();
        assert!(files.contains(&"test.txt".to_string()));
    }
}
