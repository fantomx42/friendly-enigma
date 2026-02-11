import asyncio
import random
import logging
from typing import Optional
from wheeler.core.memory import WheelerMemory

logger = logging.getLogger("wheeler.autonomic")

class AutonomicSystem:
    def __init__(self, memory: WheelerMemory):
        self.memory = memory
        self.running = False
        self._task: Optional[asyncio.Task] = None
        self.tick_rate = 30.0  # Seconds between ticks
        self.dream_chance = 0.5
        self.consolidation_chance = 0.5

    async def start(self):
        if self.running:
            return
        self.running = True
        self._task = asyncio.create_task(self._loop())
        logger.info("Autonomic system started.")

    async def stop(self):
        self.running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Autonomic system stopped.")

    async def _loop(self):
        while self.running:
            try:
                await asyncio.sleep(self.tick_rate)
                
                if random.random() < self.consolidation_chance:
                    await self._consolidate()
                    
                if random.random() < self.dream_chance:
                    await self._dream()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in autonomic loop: {e}")

    async def _consolidate(self):
        """Strengthen confidence of frequently accessed memories."""
        all_meta = await self.memory.storage.metadata.list_memories()
        # Find memories with hits > 3 or similar
        for meta in all_meta:
            if meta.get("hit_count", 0) > 3:
                # Strengthen confidence
                await self.memory.storage.metadata.increment_confidence(meta["id"], amount=0.01)

    async def _dream(self):
        """Randomly blend memories and find resonance."""
        all_meta = await self.memory.storage.metadata.list_memories()
        if len(all_meta) < 2:
            return None
            
        # 1. Pick two random memories
        m1, m2 = random.sample(all_meta, 2)
        
        # 2. Load frames
        f1 = self.memory.storage.blobs.load(m1["uuid"])
        f2 = self.memory.storage.blobs.load(m2["uuid"])
        
        # 3. Blend and run dynamics
        blended = self.memory.reasoning.blend([f1, f2])
        dream_attractor = self.memory.engine.run(blended, steps=10)
        
        # 4. Search memory for resonance
        # We need a way to recall by frame directly
        results = await self.memory.recall_by_frame(dream_attractor, limit=1)
        
        if results and results[0]["similarity"] > 0.8:
            # Dream resonated with an existing memory!
            # Strengthen that memory
            best = results[0]
            await self.memory.storage.metadata.increment_confidence(best["id"], amount=0.02)
            logger.info(f"Dream resonated with memory UUID: {best['uuid']}")
            return best
            
        return None
