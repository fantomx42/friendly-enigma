# Python API Reference

```python
from wheeler_memory import (
    store_with_rotation_retry,
    recall_memory,
    list_memories,
    select_chunk,
    select_recall_chunks,
)

# Store
result = store_with_rotation_retry("fix the auth bug", chunk="code")
# result["state"] == "CONVERGED"

# Recall
matches = recall_memory("authentication error", top_k=5)
for m in matches:
    print(m["text"], m["similarity"], m["chunk"])

# Chunk routing
select_chunk("debug the python script")          # → "code"
select_chunk("buy groceries")                     # → "daily_tasks"
select_recall_chunks("quantum physics equation")  # → ["science", "general"]

# List all memories
for mem in list_memories():
    print(mem["chunk"], mem["text"])
```
