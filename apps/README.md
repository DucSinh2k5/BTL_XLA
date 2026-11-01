# Applications

- `sun397_demo/`: the existing FastAPI backend, HTML/CSS/JavaScript frontend, assets and API tests.
- `segmentation/`: empty reserved slot, not an implemented demo.

From the project root run:

```powershell
python -m uvicorn --app-dir apps sun397_demo.app:app --host 127.0.0.1 --port 8000
```

The package-local README preserves its earlier package-root run instructions.
Current model-unavailable responses are intentional until a compatible checkpoint adapter is implemented.
