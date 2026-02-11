# Specification: Knowledge Ingestion & System Cleanup

## Goal
Populate the Wheeler Memory system with a foundational "library" of general knowledge to improve associative recall and reasoning, while removing temporary test artifacts to maintain system hygiene.

## Requirements
1. **Environment Preparation**: Ensure `datasets` and `tqdm` are installed in the project's virtual environment.
2. **Data Source**: Use the `databricks/databricks-dolly-15k` dataset from Hugging Face.
3. **Ingestion Logic**:
    - Load the dataset in streaming mode.
    - Extract the `response` field from each entry.
    - Truncate text to 300 characters to ensure stable attractors.
    - Store the first 100 entries into the main `.wheeler` storage.
4. **Cleanup**:
    - Remove the `.wheeler_test` directory if it exists.
    - Remove the `scripts/test_ingestion_speed.py` (already done, but verify).
5. **Verification**:
    - Run a `recall` command to verify that the new knowledge is searchable.
    - Run a `dream` cycle to ensure the system can process the new memories.

## Success Criteria
- 100 new memories successfully stored in `.wheeler/wheeler.db`.
- `.wheeler_test` directory is deleted.
- `recall` returns relevant results from the Dolly dataset.
