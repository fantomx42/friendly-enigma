import os
import asyncio
from pathlib import Path
from wheeler.core.memory import WheelerMemory

class WheelerBridge:
    def __init__(self, storage_dir: str = None):
        if storage_dir is None:
            storage_dir = str(Path.home() / ".wheeler_memory")
        self.storage_dir = storage_dir
        self.wm = WheelerMemory(storage_dir)
        self._initialized = False

    async def _ensure_initialized(self):
        if not self._initialized:
            os.makedirs(self.storage_dir, exist_ok=True)
            await self.wm.initialize()
            self._initialized = True

    async def store(self, text: str, memory_type: str = "iteration") -> str:
        """Stores text in Wheeler Memory."""
        await self._ensure_initialized()
        # Note: Current WheelerMemory.store doesn't support 'type' in DB yet
        # but we can add it to the key or metadata later.
        return await self.wm.store(text)

    async def recall(self, query: str, limit: int = 3) -> str:
        """Recalls memories and returns a formatted string for Ralph."""
        await self._ensure_initialized()
        results = await self.wm.recall(query, limit=limit)
        
        if not results:
            return ""
            
        output_lines = []
        for res in results:
            text = res["key"][:200]
            stability = res["stability"]
            score = res["score"]
            # Formatting to match what Ralph expects
            output_lines.append(
                f"[memory] (stability: {stability:.2f}, relevance: {score:.2f}) {text}"
            )
            
        return "\n".join(output_lines)

# Singleton-like access for scripts
_bridge = None

def get_bridge(storage_dir: str = None) -> WheelerBridge:
    global _bridge
    if _bridge is None:
        _bridge = WheelerBridge(storage_dir)
    return _bridge
