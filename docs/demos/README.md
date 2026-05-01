# Wheeler Memory Demos

Static HTML demonstrations of the Wheeler Memory system.

## Files

- **demo.html** — Interactive educational walkthrough. Standalone, opens directly in a browser. Covers CA dynamics, attractors, temperature, and example evolutions.
- **dashboard.html** — Historical control-panel UI. Was previously served by a `wheeler-ui` Python entry point that has been retired (the entry point and its server were removed in v0.3.6 because they had drifted out of date with the core).
- **chat.html** — Historical chat-style demo, also previously server-backed.

## Status

Only `demo.html` is fully functional today and runs offline as a static page. `dashboard.html` and `chat.html` are kept for reference; they will not work without a server implementing the `/api/` endpoints they expect.

## Future

If you want a live dashboard back, the right path is a fresh implementation that talks to the current `wheeler_memory.recall_api` and `wheeler_memory.storage` surfaces — not a revival of the old `wheeler_ui.py`.
