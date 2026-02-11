import os
import uuid
import torch
import math
from typing import List, Dict, Any, Optional
from wheeler.core.codec import TextCodec
from wheeler.core.engine import DynamicsEngine
from wheeler.core.reasoning import ReasoningEngine
from wheeler.core.storage import MetadataStore, BlobStore, StorageController

class WheelerMemory:
    def __init__(self, storage_dir: str, device: str = "cpu"):
        self.storage_dir = storage_dir
        self.device = device
        
        self.codec = TextCodec()
        self.engine = DynamicsEngine(device=device)
        self.reasoning = ReasoningEngine(device=device)
        
        db_path = os.path.join(storage_dir, "wheeler.db")
        blob_dir = os.path.join(storage_dir, "blobs")
        
        self.storage = StorageController(
            MetadataStore(db_path),
            BlobStore(blob_dir)
        )

    async def initialize(self) -> None:
        await self.storage.initialize()

    async def store(self, text: str) -> str:
        """Encodes text, runs dynamics, and stores the attractor."""
        # 1. Encode
        initial_grid = self.codec.encode(text)
        
        # 2. Run dynamics to find attractor
        attractor, stability = self.engine.run_with_stats(initial_grid, steps=10)
        
        # 3. Store
        memory_uuid = str(uuid.uuid4())
        # Initial confidence 0.5 to allow for consolidation growth
        memory_id = await self.storage.save_memory(
            key=text,
            uuid=memory_uuid,
            frame=attractor,
            stability=stability,
            confidence=0.5
        )
        return memory_uuid

    async def recall(self, text: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Encodes query, runs dynamics, and finds similar memories."""
        # 1. Encode query
        query_initial = self.codec.encode(text)
        
        # 2. Run dynamics for query
        query_attractor, _ = self.engine.run_with_stats(query_initial, steps=10)
        
        # 3. Search storage
        # Fetch a limited set of recent/stable memories to prevent DoS
        all_meta = await self.storage.metadata.list_memories(limit=500)
        results = []
        
        for meta in all_meta:
            # Load the frame
            blob_filename = os.path.basename(meta["blob_path"])
            blob_id = os.path.splitext(blob_filename)[0]
            stored_frame = self.storage.blobs.load(blob_id)
            
            # Calculate similarity (correlation)
            similarity = self._calculate_similarity(query_attractor, stored_frame)
            
            # Use cached stability
            stability = meta.get("stability", 0.0)
            
            meta["similarity"] = similarity
            meta["stability"] = stability
            # Combined score: similarity weighted by stability
            meta["score"] = similarity * (0.5 + 0.5 * stability)
            
            results.append(meta)
            
        # Sort by combined score descending
        results.sort(key=lambda x: x["score"], reverse=True)
        
        return results[:limit]

    async def recall_by_frame(self, frame: torch.Tensor, limit: int = 5) -> List[Dict[str, Any]]:
        """Finds memories similar to a given attractor frame."""
        # Fetch a limited set of recent/stable memories to prevent DoS
        all_meta = await self.storage.metadata.list_memories(limit=500)
        results = []
        
        for meta in all_meta:
            blob_filename = os.path.basename(meta["blob_path"])
            blob_id = os.path.splitext(blob_filename)[0]
            stored_frame = self.storage.blobs.load(blob_id)
            
            similarity = self._calculate_similarity(frame, stored_frame)
            stability = meta.get("stability", 0.0)
            
            meta["similarity"] = similarity
            meta["stability"] = stability
            meta["score"] = similarity * (0.5 + 0.5 * stability)
            results.append(meta)
            
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    async def infer(self, text_a: str, text_b: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Associative reasoning: blends two concepts and finds the resulting attractor's 
        nearest neighbors in memory.
        """
        # 1. Get attractors for both
        fa = self.engine.run(self.codec.encode(text_a), steps=10)
        fb = self.engine.run(self.codec.encode(text_b), steps=10)
        
        # 2. Blend
        blended = self.reasoning.blend([fa, fb])
        
        # 3. Project to new attractor
        attractor = self.engine.run(blended, steps=10)
        
        # 4. Search memory using this new "inferred" attractor
        # Fetch a limited set of recent/stable memories to prevent DoS
        all_meta = await self.storage.metadata.list_memories(limit=500)
        results = []
        for meta in all_meta:
            blob_filename = os.path.basename(meta["blob_path"])
            blob_id = os.path.splitext(blob_filename)[0]
            stored_frame = self.storage.blobs.load(blob_id)
            
            similarity = self._calculate_similarity(attractor, stored_frame)
            stability = meta.get("stability", 0.0)
            
            meta["similarity"] = similarity
            meta["stability"] = stability
            meta["score"] = similarity * (0.5 + 0.5 * stability)
            results.append(meta)
            
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    async def load_by_uuid(self, memory_uuid: str) -> Optional[Dict[str, Any]]:
        """Loads a memory record and its frame by UUID."""
        return await self.storage.load_by_uuid(memory_uuid)

    def _calculate_stability(self, meta: Dict[str, Any], stored_frame: torch.Tensor) -> float:
        """
        Calculates stability score based on hit count, persistence, and survival.
        """
        # 1. Hit count score (40%)
        hits = meta.get("hit_count", 0)
        hit_score = 1.0 / (1.0 + math.exp(-0.1 * hits)) # Sigmoid
        hit_score = (hit_score - 0.5) * 2.0 # Normalize to 0-1 (approx)
        hit_score = max(0.0, min(1.0, hit_score))

        # 2. Persistence score (30%)
        # Re-encode original key and check correlation
        re_encoded = self.codec.encode(meta["key"])
        persistence_corr = self._calculate_similarity(re_encoded, stored_frame)
        persistence_score = (persistence_corr + 1.0) / 2.0

        # 3. Survival score (30%)
        # Run 5 steps of dynamics and check how much survives
        compressed = self.engine.run(stored_frame, steps=5)
        survival_corr = self._calculate_similarity(stored_frame, compressed)
        survival_score = (survival_corr + 1.0) / 2.0

        return 0.4 * hit_score + 0.3 * persistence_score + 0.3 * survival_score

    def _calculate_similarity(self, f1: torch.Tensor, f2: torch.Tensor) -> float:
        """Pearson correlation between two frames."""
        f1_flat = f1.view(-1)
        f2_flat = f2.view(-1)
        
        f1_mean = f1_flat.mean()
        f2_mean = f2_flat.mean()
        
        f1_centered = f1_flat - f1_mean
        f2_centered = f2_flat - f2_mean
        
        norm1 = f1_centered.norm()
        norm2 = f2_centered.norm()
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        correlation = (f1_centered @ f2_centered) / (norm1 * norm2)
        return float(correlation.item())
