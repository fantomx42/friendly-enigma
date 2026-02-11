import aiosqlite
import os
from typing import List, Optional, Dict, Any
from datetime import datetime

class MetadataStore:
    def __init__(self, db_path: str):
        self.db_path = db_path

    async def initialize(self) -> None:
        """Create the database and tables if they don't exist."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    uuid TEXT NOT NULL UNIQUE,
                    key TEXT NOT NULL,
                    blob_path TEXT NOT NULL,
                    hit_count INTEGER DEFAULT 0,
                    stability REAL DEFAULT 0.0,
                    confidence REAL DEFAULT 0.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await db.commit()

    def connection(self):
        """Returns an async connection context manager."""
        return aiosqlite.connect(self.db_path)

    async def create_memory(
        self, 
        key: str, 
        uuid: str,
        blob_path: str, 
        stability: float = 0.0, 
        confidence: float = 0.0
    ) -> int:
        async with self.connection() as db:
            cursor = await db.execute(
                """
                INSERT INTO memories (key, uuid, blob_path, stability, confidence)
                VALUES (?, ?, ?, ?, ?)
                """,
                (key, uuid, blob_path, stability, confidence)
            )
            await db.commit()
            return cursor.lastrowid

    async def get_memory(self, memory_id: int) -> Optional[Dict[str, Any]]:
        async with self.connection() as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM memories WHERE id = ?", (memory_id,)
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    return dict(row)
                return None

    async def get_memory_by_uuid(self, memory_uuid: str) -> Optional[Dict[str, Any]]:
        async with self.connection() as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM memories WHERE uuid = ?", (memory_uuid,)
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    return dict(row)
                return None

    async def increment_hit_count(self, memory_id: int) -> None:
        async with self.connection() as db:
            await db.execute(
                """
                UPDATE memories 
                SET hit_count = hit_count + 1, 
                    last_accessed = CURRENT_TIMESTAMP 
                WHERE id = ?
                """,
                (memory_id,)
            )
            await db.commit()

    async def increment_confidence(self, memory_id: int, amount: float = 0.01) -> None:
        async with self.connection() as db:
            await db.execute(
                """
                UPDATE memories 
                SET confidence = MIN(1.0, confidence + ?)
                WHERE id = ?
                """,
                (amount, memory_id)
            )
            await db.commit()

    async def update_scores(
        self, 
        memory_id: int, 
        stability: Optional[float] = None, 
        confidence: Optional[float] = None
    ) -> None:
        async with self.connection() as db:
            if stability is not None and confidence is not None:
                await db.execute(
                    "UPDATE memories SET stability = ?, confidence = ? WHERE id = ?",
                    (stability, confidence, memory_id)
                )
            elif stability is not None:
                await db.execute(
                    "UPDATE memories SET stability = ? WHERE id = ?",
                    (stability, memory_id)
                )
            elif confidence is not None:
                await db.execute(
                    "UPDATE memories SET confidence = ? WHERE id = ?",
                    (confidence, memory_id)
                )
            await db.commit()

    async def list_memories(self) -> List[Dict[str, Any]]:
        async with self.connection() as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT * FROM memories") as cursor:
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]

    async def delete_memory(self, memory_id: int) -> None:
        async with self.connection() as db:
            await db.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            await db.commit()

class BlobStore:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def _get_path(self, memory_id: str) -> str:
        return os.path.join(self.base_dir, f"{memory_id}.npy")

    def save(self, memory_id: str, frame: Any) -> str:
        """Saves a frame (torch.Tensor or np.ndarray) to disk."""
        import numpy as np
        import torch
        
        if isinstance(frame, torch.Tensor):
            frame_np = frame.detach().cpu().numpy()
        else:
            frame_np = frame
            
        path = self._get_path(memory_id)
        np.save(path, frame_np)
        return path

    def load(self, memory_id: str) -> Any:
        """Loads a frame from disk as a torch.Tensor."""
        import numpy as np
        import torch
        
        path = self._get_path(memory_id)
        if not os.path.exists(path):
            raise FileNotFoundError(f"No blob found for memory {memory_id}")
            
        frame_np = np.load(path)
        return torch.from_numpy(frame_np)

    def delete(self, memory_id: str) -> None:
        path = self._get_path(memory_id)
        if os.path.exists(path):
            os.remove(path)

class StorageController:
    def __init__(self, metadata: MetadataStore, blobs: BlobStore):
        self.metadata = metadata
        self.blobs = blobs

    async def initialize(self) -> None:
        await self.metadata.initialize()

    async def save_memory(
        self, 
        key: str, 
        uuid: str, 
        frame: Any, 
        stability: float = 0.0, 
        confidence: float = 0.0
    ) -> int:
        """Saves both metadata and blob."""
        blob_path = self.blobs.save(uuid, frame)
        memory_id = await self.metadata.create_memory(
            key=key,
            uuid=uuid,
            blob_path=blob_path,
            stability=stability,
            confidence=confidence
        )
        return memory_id

    async def load_memory(self, memory_id: int) -> Optional[Dict[str, Any]]:
        """Loads metadata and frame."""
        meta = await self.metadata.get_memory(memory_id)
        if not meta:
            return None
            
        # Extract UUID from blob_path (or we should have stored UUID in DB)
        # For now, let's assume the blob_path is enough to load
        # Actually, let's just use the filename from blob_path minus .npy
        blob_filename = os.path.basename(meta["blob_path"])
        blob_id = os.path.splitext(blob_filename)[0]
        
        frame = self.blobs.load(blob_id)
        meta["frame"] = frame
        return meta

    async def load_by_uuid(self, memory_uuid: str) -> Optional[Dict[str, Any]]:
        """Loads metadata and frame by UUID."""
        meta = await self.metadata.get_memory_by_uuid(memory_uuid)
        if not meta:
            return None
            
        frame = self.blobs.load(memory_uuid)
        meta["frame"] = frame
        return meta

    async def delete_memory(self, memory_id: int) -> None:
        """Deletes both metadata and blob."""
        meta = await self.metadata.get_memory(memory_id)
        if meta:
            blob_filename = os.path.basename(meta["blob_path"])
            blob_id = os.path.splitext(blob_filename)[0]
            self.blobs.delete(blob_id)
            await self.metadata.delete_memory(memory_id)
