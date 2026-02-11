import pytest
import asyncio
import os
import sqlite3
from wheeler.core.storage import MetadataStore

@pytest.fixture
async def store(tmp_path):
    db_path = tmp_path / "test_metadata.db"
    store = MetadataStore(db_path)
    await store.initialize()
    return store

@pytest.mark.asyncio
async def test_store_initialization(tmp_path):
    db_path = tmp_path / "init_test.db"
    store = MetadataStore(db_path)
    await store.initialize()
    assert os.path.exists(db_path)
    
    # Verify table exists
    async with store.connection() as db:
        cursor = await db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='memories'")
        table = await cursor.fetchone()
        assert table is not None

@pytest.mark.asyncio
async def test_create_and_get_memory(store):
    memory_id = await store.create_memory(
        key="test_key",
        uuid="test_uuid",
        blob_path="/tmp/test.npy",
        stability=0.85,
        confidence=0.9
    )
    
    assert memory_id is not None
    
    memory = await store.get_memory(memory_id)
    assert memory["key"] == "test_key"
    assert memory["uuid"] == "test_uuid"
    assert memory["blob_path"] == "/tmp/test.npy"
    assert memory["stability"] == 0.85
    assert memory["confidence"] == 0.9
    assert memory["hit_count"] == 0

@pytest.mark.asyncio
async def test_update_hit_count(store):
    memory_id = await store.create_memory(key="hits", uuid="uuid_hits", blob_path="path")
    
    await store.increment_hit_count(memory_id)
    await store.increment_hit_count(memory_id)
    
    memory = await store.get_memory(memory_id)
    assert memory["hit_count"] == 2

@pytest.mark.asyncio
async def test_update_scores(store):
    memory_id = await store.create_memory(key="scores", uuid="uuid_scores", blob_path="path")
    
    await store.update_scores(memory_id, stability=0.99, confidence=0.7)
    
    memory = await store.get_memory(memory_id)
    assert memory["stability"] == 0.99
    assert memory["confidence"] == 0.7

@pytest.mark.asyncio
async def test_list_memories(store):
    await store.create_memory(key="m1", uuid="u1", blob_path="p1")
    await store.create_memory(key="m2", uuid="u2", blob_path="p2")
    
    memories = await store.list_memories()
    assert len(memories) == 2
    keys = [m["key"] for m in memories]
    assert "m1" in keys
    assert "m2" in keys

@pytest.mark.asyncio
async def test_delete_memory(store):
    memory_id = await store.create_memory(key="to_delete", uuid="uuid_delete", blob_path="path")
    await store.delete_memory(memory_id)
    
    memory = await store.get_memory(memory_id)
    assert memory is None

def test_blob_store_save_load(tmp_path):
    from wheeler.core.storage import BlobStore
    import numpy as np
    import torch
    
    blob_dir = tmp_path / "blobs"
    blob_dir.mkdir()
    store = BlobStore(str(blob_dir))
    
    # Create a dummy frame (128x128)
    frame = torch.randn(128, 128)
    memory_id = "test_uuid"
    
    path = store.save(memory_id, frame)
    assert os.path.exists(path)
    assert memory_id in path
    
    loaded_frame = store.load(memory_id)
    assert torch.allclose(frame, loaded_frame)

def test_blob_store_delete(tmp_path):
    from wheeler.core.storage import BlobStore
    import torch
    
    blob_dir = tmp_path / "blobs"
    blob_dir.mkdir()
    store = BlobStore(str(blob_dir))
    
    frame = torch.randn(128, 128)
    memory_id = "to_delete"
    path = store.save(memory_id, frame)
    
    store.delete(memory_id)
    assert not os.path.exists(path)

@pytest.mark.asyncio
async def test_storage_controller_save_load(tmp_path):
    from wheeler.core.storage import MetadataStore, BlobStore, StorageController
    import torch
    import uuid
    
    db_path = tmp_path / "hybrid.db"
    blob_dir = tmp_path / "blobs"
    blob_dir.mkdir()
    
    controller = StorageController(
        MetadataStore(str(db_path)),
        BlobStore(str(blob_dir))
    )
    await controller.initialize()
    
    frame = torch.randn(128, 128)
    memory_uuid = str(uuid.uuid4())
    
    memory_id = await controller.save_memory(
        key="hybrid_test",
        uuid=memory_uuid,
        frame=frame,
        stability=0.7
    )
    
    # Load back
    loaded_memory = await controller.load_memory(memory_id)
    assert loaded_memory["key"] == "hybrid_test"
    assert torch.allclose(loaded_memory["frame"], frame)
    assert loaded_memory["stability"] == 0.7

@pytest.mark.asyncio
async def test_storage_controller_delete(tmp_path):
    from wheeler.core.storage import MetadataStore, BlobStore, StorageController
    import torch
    import uuid
    
    controller = StorageController(
        MetadataStore(str(tmp_path / "db")),
        BlobStore(str(tmp_path / "blobs"))
    )
    await controller.initialize()
    
    memory_id = await controller.save_memory(
        key="delete_me",
        uuid=str(uuid.uuid4()),
        frame=torch.zeros(128, 128)
    )
    
    # Verify file exists
    memory = await controller.metadata.get_memory(memory_id)
    assert os.path.exists(memory["blob_path"])
    
    await controller.delete_memory(memory_id)
    
    # Verify record and file are gone
    assert await controller.metadata.get_memory(memory_id) is None
    assert not os.path.exists(memory["blob_path"])
