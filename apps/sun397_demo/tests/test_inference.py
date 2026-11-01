import asyncio
import json
from pathlib import Path
import sys
import os
import subprocess

import pytest

from sun397_demo.inference import InferenceManager, InferenceUnavailable, ModelWorker, Settings, ModelRoute


def settings(tmp_path):
    bundles = {name: tmp_path / name for name in ("tacs", "mtacs")}
    for root in bundles.values():
        root.mkdir()
        (root / "model.pt").write_bytes(b"test fixture, not model weights")
        (root / "manifest.json").write_text("{}")
    return Settings(bundles, sys.executable, None, tmp_path / "runtime", startup_timeout=10,
                    prediction_timeout=10)


def test_settings_default_to_managed_artifacts(monkeypatch):
    from sun397_demo.inference import PROJECT
    monkeypatch.setenv("SUN397_TACS_BUNDLE", str(PROJECT / "artifacts/deployment/sun397/tacs_visual"))
    monkeypatch.setenv("SUN397_MTACS_BUNDLE", str(PROJECT / "artifacts/deployment/sun397/mtacs_full_both"))
    result = Settings.from_environment()
    assert result.bundles["tacs"].name == "tacs_visual"
    assert result.bundles["mtacs"].name == "mtacs_full_both"


def test_missing_bundle_reports_error_without_loading_anything(tmp_path):
    opts = Settings({"tacs": tmp_path / "missing"}, sys.executable, None, tmp_path / "runtime")
    worker = ModelWorker("tacs", opts)
    worker.start()
    assert worker.status()["code"] == "error"
    assert worker.process is None
    with pytest.raises(InferenceUnavailable):
        worker.predict(b"image")


def test_persistent_process_protocol_and_shutdown(tmp_path, monkeypatch):
    import sun397_demo.inference as inference
    script = tmp_path / "fixture_worker.py"
    script.write_text('''import sys,json,hashlib
from pathlib import Path
def emit(event,**values):
 print("SUN397_EVENT "+json.dumps(dict(event=event,**values)),flush=True)
emit("progress",message="fixture loading")
emit("ready",details={"test_fixture":True})
for line in sys.stdin:
 req=json.loads(line)
 content=Path(req["image"]).read_bytes()
 Path(req["output"]).write_text(json.dumps({"model":"tacs","prediction":{"label":"fixture"},"sha256":hashlib.sha256(content).hexdigest()}))
 emit("result")
''', encoding="utf-8")
    original_popen = inference.subprocess.Popen
    def popen(command, **kwargs):
        assert command[3:5] == ["--model", "tacs"]
        assert kwargs["env"]["HF_HUB_OFFLINE"] == "1"
        return original_popen([sys.executable, "-u", str(script)], **kwargs)
    monkeypatch.setattr(inference.subprocess, "Popen", popen)
    worker = ModelWorker("tacs", settings(tmp_path))
    try:
        worker.start()
        assert worker.status()["available"]
        pid = worker.process.pid
        first, second = worker.predict(b"first"), worker.predict(b"second")
        assert first["sha256"] != second["sha256"]
        assert worker.process.pid == pid
        assert not list((worker.settings.runtime / "requests").iterdir())
    finally:
        worker.stop()
    assert worker.process.poll() is not None


def test_manager_automatically_starts_both_models_and_serializes_requests(tmp_path):
    async def run():
        manager = InferenceManager(settings(tmp_path))
        calls, active = [], []
        class Worker:
            def __init__(self, name): self.name = name
            def start(self): calls.append(self.name)
            def stop(self): calls.append("stop:" + self.name)
            def status(self): return {"available": True}
            def predict(self, content):
                import time
                active.append(self.name)
                assert len(active) == 1
                time.sleep(0.01)
                active.pop()
                return self.name
        manager.workers = {name: Worker(name) for name in ("tacs", "mtacs")}
        await manager.start()
        await manager._startup
        assert calls == ["mtacs", "tacs"]
        assert await asyncio.gather(manager.predict("tacs", b"a"), manager.predict("mtacs", b"b")) == ["tacs", "mtacs"]
        await manager.close()
        assert calls[-2:] == ["stop:tacs", "stop:mtacs"]
    asyncio.run(run())


@pytest.mark.skipif(os.name != "nt", reason="Windows parent handle watchdog")
def test_worker_exits_when_its_api_parent_disappears(tmp_path):
    import ctypes
    from ctypes import wintypes
    import sun397_demo.inference as inference
    apps = str(Path(inference.__file__).resolve().parents[1])
    ready = tmp_path / "child_ready"
    script = tmp_path / "parent.py"
    script.write_text('''import subprocess,sys,os,time
from pathlib import Path
apps,ready=sys.argv[1:]
code="import sys,time;from pathlib import Path;sys.path.insert(0,sys.argv[1]);from sun397_demo.worker import watch_parent;watch_parent(int(sys.argv[2]));Path(sys.argv[3]).write_text('ready');time.sleep(15)"
child=subprocess.Popen([sys.executable,"-c",code,apps,str(os.getpid()),ready],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
deadline=time.monotonic()+10
while not Path(ready).exists():
 if time.monotonic()>deadline: raise RuntimeError("child watchdog did not initialize")
 time.sleep(0.05)
print(child.pid,flush=True)
''', encoding="utf-8")
    parent = subprocess.run([sys.executable, str(script), apps, str(ready)], capture_output=True,
                            text=True, timeout=15, check=True)
    child_pid = int(parent.stdout.strip())
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE,wintypes.DWORD]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x00100000,False,child_pid)
    if handle:
        try:
            assert kernel.WaitForSingleObject(handle,5000) == 0
        finally:
            kernel.CloseHandle(handle)
    else:
        assert ctypes.get_last_error() == 87  # Child already exited and was reaped.


def test_shared_process_routes_models_without_mixing_their_status_or_predictions(tmp_path):
    async def run():
        options = settings(tmp_path)
        options.shared_process = True
        manager = InferenceManager(options)
        calls = []
        class Shared:
            def start(self): calls.append("start_once")
            def stop(self): calls.append("stop_once")
            def status(self):
                return {"available":True,"code":"ready","shared_process":True,
                        "models":{"tacs":{"caption_models_loaded":False},"mtacs":{"caption_models_loaded":True}}}
            def predict(self, content, model=None):
                calls.append((model,content))
                return {"model":model}
        manager.shared = Shared()
        manager.workers = {name:ModelRoute(name,manager.shared) for name in ("tacs","mtacs")}
        await manager.start()
        await manager._startup
        assert manager.status("tacs")["caption_models_loaded"] is False
        assert manager.status("mtacs")["caption_models_loaded"] is True
        assert (await manager.predict("tacs",b"query"))["model"] == "tacs"
        assert (await manager.predict("mtacs",b"query"))["model"] == "mtacs"
        await manager.close()
        assert calls == ["start_once",("tacs",b"query"),("mtacs",b"query"),"stop_once"]
    asyncio.run(run())
