import asyncio
import os
import sys
from tqdm import tqdm

# Ensure we can import the wheeler package
sys.path.append(os.getcwd())

try:
    from datasets import load_dataset
except ImportError:
    print("Error: 'datasets' library not found. Run: pip install datasets tqdm")
    sys.exit(1)

from wheeler.core.memory import WheelerMemory

async def ingest(limit=100):
    storage_dir = './.wheeler'
    wm = WheelerMemory(storage_dir)
    await wm.initialize()
    
    print(f"Loading 'databricks-dolly-15k' general knowledge dataset...")
    # Dolly contains high-quality human-written facts and instructions
    dataset = load_dataset("databricks/databricks-dolly-15k", split="train", streaming=True)
    
    print(f"Ingesting first {limit} entries into Wheeler Memory...")
    count = 0
    # Streaming dataset is a standard iterable, not async iterable
    for entry in tqdm(dataset, total=limit):
        if count >= limit:
            break
        
        # Dolly structure: instruction, context, response
        # We'll take the response as the 'fact' to remember
        text = entry.get('response', '')
        if not text:
            continue
            
        # Clean up text (limit to first 300 chars to keep attractors focused)
        text = text[:300]
        
        await wm.store(text)
        count += 1
        
    print(f"\nSuccessfully ingested {count} memories.")
    print("You can now use 'python -m wheeler recall \"your query\"' to see the results!")

if __name__ == "__main__":
    # Default to 100 for a quick start
    limit = 100
    if len(sys.argv) > 1:
        limit = int(sys.argv[1])
        
    asyncio.run(ingest(limit))
