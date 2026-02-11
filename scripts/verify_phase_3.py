import asyncio
import os
import torch
import uuid
from wheeler.core.storage import MetadataStore, BlobStore, StorageController

async def verify():
    print("--- Phase 3 Manual Verification ---")
    
    # Setup temporary verification directory
    base_dir = "verification_storage"
    os.makedirs(base_dir, exist_ok=True)
    
    db_path = os.path.join(base_dir, "wheeler.db")
    blob_dir = os.path.join(base_dir, "blobs")
    
    print(f"1. Initializing storage in {base_dir}/...")
    metadata = MetadataStore(db_path)
    blobs = BlobStore(blob_dir)
    storage = StorageController(metadata, blobs)
    await storage.initialize()
    
    # Create dummy data
    print("2. Creating a test memory...")
    frame = torch.randn(128, 128)
    mem_uuid = str(uuid.uuid4())
    
    memory_id = await storage.save_memory(
        key="Phase 3 Verification",
        uuid=mem_uuid,
        frame=frame,
        stability=0.99
    )
    print(f"   - Saved memory ID: {memory_id}")
    print(f"   - Saved UUID: {mem_uuid}")
    
    # Verify files
    print("3. Verifying file persistence...")
    if os.path.exists(db_path):
        size = os.path.getsize(db_path)
        print(f"   - [OK] Database found: {db_path} ({size} bytes)")
    else:
        print(f"   - [FAIL] Database missing: {db_path}")

    # Check for blob file
    # We need to find the file in blobs/
    # The blob store saves as {uuid}.npy
    blob_path = os.path.join(blob_dir, f"{mem_uuid}.npy")
    if os.path.exists(blob_path):
        size = os.path.getsize(blob_path)
        print(f"   - [OK] Blob file found: {blob_path} ({size} bytes)")
    else:
        print(f"   - [FAIL] Blob file missing: {blob_path}")

    # Verify data content
    print("4. Verifying data integrity...")
    loaded = await storage.load_memory(memory_id)
    if loaded:
        print("   - [OK] Memory loaded back from disk.")
        if torch.allclose(loaded["frame"], frame):
             print("   - [OK] Tensor data matches exactly.")
        else:
             print("   - [FAIL] Tensor data mismatch.")
    else:
        print("   - [FAIL] Could not load memory.")

    print("--- Done ---")

if __name__ == "__main__":
    asyncio.run(verify())
