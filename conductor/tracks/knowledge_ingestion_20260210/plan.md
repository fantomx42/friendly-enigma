# Implementation Plan - Knowledge Ingestion & System Cleanup

This track focuses on populating the memory system with general knowledge and cleaning up test artifacts.

## Phase 1: Environment & Script Finalization
- [x] Task: Install dependencies (`datasets`, `tqdm`) [005eb98]
    - [x] Run `.venv/bin/pip install datasets tqdm`
- [x] Task: Verify ingestion script (`scripts/ingest_hf.py`) [94fdece]
    - [x] Ensure the script correctly targets `databricks/databricks-dolly-15k`
    - [x] Ensure the script truncates text to 300 chars
- [x] Task: Conductor - User Manual Verification 'Phase 1: Environment & Script Finalization' (Protocol in workflow.md) [44eeed1]

## Phase 2: Knowledge Ingestion
- [x] Task: Execute ingestion [420ba82]
    - [x] Run `PYTHONPATH=. .venv/bin/python3 scripts/ingest_hf.py 100`
- [x] Task: Verify memory count [420ba82]
    - [x] Check `wheeler.db` or use `recall` to confirm data presence
- [x] Task: Conductor - User Manual Verification 'Phase 2: Knowledge Ingestion' (Protocol in workflow.md) [420ba82]

## Phase 3: System Cleanup
- [x] Task: Remove test artifacts [9f9f4e6]
    - [x] Delete `./.wheeler_test` directory
    - [x] Ensure any other temporary speed test files are gone
- [x] Task: Conductor - User Manual Verification 'Phase 3: System Cleanup' (Protocol in workflow.md) [9f9f4e6]

## Phase 4: Final Validation
- [ ] Task: Run recall test
    - [ ] Perform `recall` on a general topic (e.g., "science" or "history")
- [ ] Task: Run autonomic cycle
    - [ ] Execute `dream --ticks 5` to verify consolidation of new data
- [ ] Task: Conductor - User Manual Verification 'Phase 4: Final Validation' (Protocol in workflow.md)
