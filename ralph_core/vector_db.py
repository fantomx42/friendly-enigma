import os
import json
import hashlib
import numpy as np
import requests
from typing import List, Dict, Any, Optional

LOCAL_DB_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "memory", "vectors.json")
GLOBAL_DB_FILE = os.path.expanduser("~/.ralph/global_memory/vectors.json")
OLLAMA_EMBED_API = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"

class VectorDB:
    def __init__(self):
        self.local_db_file = LOCAL_DB_FILE
        self.global_db_file = GLOBAL_DB_FILE

        # Embedding cache to avoid re-computing same embeddings
        self.embedding_cache: Dict[str, List[float]] = {}

        # In-memory storage for search (combined)
        self.vectors: List[List[float]] = []
        self.documents: List[Dict[str, Any]] = []

        # Keep track of indices for saving?
        # Easier strategy: Load separately, keep generic "add" interface, but "save" updates specific files.
        # Actually, let's keep separate lists in memory to make saving easier.
        self.local_vectors = []
        self.local_docs = []
        self.global_vectors = []
        self.global_docs = []

        self._load()

    def _load(self):
        # Load Local
        if os.path.exists(self.local_db_file):
            try:
                with open(self.local_db_file, 'r') as f:
                    data = json.load(f)
                    self.local_vectors = data.get("vectors", [])
                    self.local_docs = data.get("documents", [])
            except Exception as e:
                print(f"[VectorDB] Local load error: {e}")
        
        # Load Global
        if os.path.exists(self.global_db_file):
            try:
                with open(self.global_db_file, 'r') as f:
                    data = json.load(f)
                    self.global_vectors = data.get("vectors", [])
                    self.global_docs = data.get("documents", [])
            except Exception as e:
                print(f"[VectorDB] Global load error: {e}")
                
        # Combine for search
        self.vectors = self.local_vectors + self.global_vectors
        self.documents = self.local_docs + self.global_docs

    def _save(self, scope: str):
        try:
            if scope == "local":
                target_file = self.local_db_file
                data = {"vectors": self.local_vectors, "documents": self.local_docs}
            else:
                target_file = self.global_db_file
                data = {"vectors": self.global_vectors, "documents": self.global_docs}
                
            os.makedirs(os.path.dirname(target_file), exist_ok=True)
            with open(target_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            print(f"[VectorDB] Save error ({scope}): {e}")

    def _get_embedding(self, text: str) -> Optional[List[float]]:
        # Check cache first (avoids redundant API calls)
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in self.embedding_cache:
            return self.embedding_cache[text_hash]

        try:
            response = requests.post(
                OLLAMA_EMBED_API,
                json={
                    "model": EMBED_MODEL,
                    "prompt": text,
                    "keep_alive": "10m"  # Keep embedding model loaded
                },
                timeout=10
            )
            response.raise_for_status()
            embedding = response.json().get("embedding")

            # Cache the result
            if embedding:
                self.embedding_cache[text_hash] = embedding

            return embedding
        except Exception as e:
            print(f"[VectorDB] Embedding error: {e}")
            return None

    def add(self, text: str, metadata: Dict[str, Any], scope: str = "local"):
        """
        Adds a document to the vector store.
        scope: 'local' (project specific) or 'global' (cross-project).
        """
        embedding = self._get_embedding(text)
        if embedding:
            doc = {
                "text": text,
                "metadata": metadata
            }
            
            if scope == "global":
                self.global_vectors.append(embedding)
                self.global_docs.append(doc)
                self._save("global")
            else:
                self.local_vectors.append(embedding)
                self.local_docs.append(doc)
                self._save("local")
                
            # Update combined view
            self.vectors.append(embedding)
            self.documents.append(doc)
            
            print(f"[VectorDB] Added ({scope}): {text[:30]}...")

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Searches for similar documents (across both local and global)."""
        query_embedding = self._get_embedding(query)
        if not query_embedding or not self.vectors:
            return []

        # Convert to numpy for fast cosine similarity
        vecs = np.array(self.vectors)
        q_vec = np.array(query_embedding)

        # Cosine Similarity: (A . B) / (||A|| * ||B||)
        norm_vecs = np.linalg.norm(vecs, axis=1)
        norm_q = np.linalg.norm(q_vec)
        
        if norm_q == 0:
            return []

        similarities = np.dot(vecs, q_vec) / (norm_vecs * norm_q)
        
        # Get top indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                "document": self.documents[idx],
                "score": float(similarities[idx])
            })
            
        return results

# Global instance
vector_memory = VectorDB()
