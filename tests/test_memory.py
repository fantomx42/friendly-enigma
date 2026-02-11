import pytest
import asyncio
import torch
from wheeler.core.memory import WheelerMemory

@pytest.fixture
async def memory(tmp_path):
    storage_dir = tmp_path / "wheeler_storage"
    storage_dir.mkdir()
    wm = WheelerMemory(str(storage_dir))
    await wm.initialize()
    return wm

@pytest.mark.asyncio
async def test_store_and_recall_exact(memory):
    text = "Hello Wheeler"
    memory_id = await memory.store(text)
    assert memory_id is not None
    
    results = await memory.recall(text)
    assert len(results) > 0
    assert results[0]["key"] == text
    assert results[0]["similarity"] > 0.9

@pytest.mark.asyncio
async def test_recall_similarity(memory):
    await memory.store("The cat is on the mat")
    await memory.store("The dog is in the yard")
    
    # Query for something similar to the first one
    results = await memory.recall("A cat on a mat")
    assert len(results) > 0
    assert "cat" in results[0]["key"]
    
@pytest.mark.asyncio
async def test_recall_no_match(memory):
    await memory.store("Relevant information")
    results = await memory.recall("Completely different topic")
    
    # Similarity should be low
    if results:
        assert results[0]["similarity"] < 0.5

@pytest.mark.asyncio
async def test_recall_with_stability_weighting(memory):
    # Store two similar things
    # One will be made more "stable" manually for testing
    text1 = "Very stable memory"
    text2 = "Unstable memory"
    
    id1 = await memory.store(text1)
    id2 = await memory.store(text2)
    
    # Manually boost hit count for text1 via storage
    meta1 = (await memory.storage.metadata.list_memories())[0]
    await memory.storage.metadata.increment_hit_count(meta1["id"])
    await memory.storage.metadata.increment_hit_count(meta1["id"])
    await memory.storage.metadata.increment_hit_count(meta1["id"])
    
    # Recall with a query that might match both
    results = await memory.recall("memory", limit=2)
    
    # Even if similarities are close, the one with more hits should have higher stability
    assert "stability" in results[0]
    # In our implementation, we'll want results sorted by a combined score

@pytest.mark.asyncio
async def test_infer(memory):
    # Store some base concepts
    await memory.store("Fire")
    await memory.store("Water")
    await memory.store("Steam")
    
    # Infer between Fire and Water
    results = await memory.infer("Fire", "Water", limit=3)
    
    assert len(results) > 0
    # "Fire" or "Water" or "Steam" should be among the results
    keys = [r["key"] for r in results]
    assert any(k in keys for k in ["Fire", "Water", "Steam"])
