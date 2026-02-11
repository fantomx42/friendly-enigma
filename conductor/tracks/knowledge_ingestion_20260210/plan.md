# Implementation Plan - Knowledge Ingestion & System Cleanup

This track focuses on populating the memory system with general knowledge and cleaning up test artifacts.

## Phase 1: Environment & Script Finalization
- [ ] Task: Install dependencies (`datasets`, `tqdm`)
    - [ ] Run `.venv/bin/pip install datasets tqdm`
- [ ] Task: Verify ingestion script (`scripts/ingest_hf.py`)
    - [ ] Ensure the script correctly targets `databricks/databricks-dolly-15k`
    - [ ] Ensure the script truncates text to 300 chars
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Environment & Script Finalization' (Protocol in workflow.md)

## Phase 2: Knowledge Ingestion
- [ ] Task: Execute ingestion
    - [ ] Run `PYTHONPATH=. .venv/bin/python3 scripts/ingest_hf.py 100`
- [ ] Task: Verify memory count
    - [ ] Check `wheeler.db` or use `recall` to confirm data presence
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Knowledge Ingestion' (Protocol in workflow.md)

## Phase 3: System Cleanup
- [ ] Task: Remove test artifacts
    - [ ] Delete `./.wheeler_test` directory
    - [ ] Ensure any other temporary speed test files are gone
- [ ] Task: Conductor - User Manual Verification 'Phase 3: System Cleanup' (Protocol in workflow.md)

## Phase 4: Final Validation
- [ ] Task: Run recall test
    - [ ] Perform `recall` on a general topic (e.g., "science" or "history")
- [ ] Task: Run autonomic cycle
    - [ ] Execute `dream --ticks 5` to verify consolidation of new data
- [ ] Task: Conductor - User Manual Verification 'Phase 4: Final Validation' (Protocol in workflow.md)
