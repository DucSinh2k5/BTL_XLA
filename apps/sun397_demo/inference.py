"""Persistent, isolated model workers; importing the web app never imports torch."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from contextlib import contextmanager
import json
import logging
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import uuid

LOG = logging.getLogger(__name__)
PROJECT = Path(__file__).resolve().parents[2]
EVENT_PREFIX = "SUN397_EVENT "


class InferenceUnavailable(RuntimeError):
    pass


@contextmanager
def request_directory(root, model):
    # Inherit workspace ACLs. Python 3.12 mkdtemp's owner-only ACL can exclude
    # the restricted Windows token used by the API or the separate worker.
    root = Path(root).resolve()
    folder = root / f"{model}_{uuid.uuid4().hex}"
    folder.mkdir()
    try:
        yield folder
    finally:
        if not folder.resolve().is_relative_to(root):
            raise ValueError("Request cleanup must stay inside runtime requests")
        for name in ("upload.image", "result.json"):
            (folder / name).unlink(missing_ok=True)
        folder.rmdir()


@dataclass
class Settings:
    bundles: dict[str, Path]
    python: str
    dependencies: Path | None
    runtime: Path
    device: str = "auto"
    autoload: bool = True
    startup_timeout: float = 1800
    prediction_timeout: float = 600
    threads: int = 4
    caption_weight_cache: Path | None = None
    shared_process: bool = False
    caption_override: Path | None = None

    @classmethod
    def from_environment(cls):
        local = PROJECT / "configs/web.local.json"
        values = json.loads(local.read_text(encoding="utf-8")) if local.is_file() else {}
        def setting(env, key, default):
            return os.getenv(env, values.get(key, default))
        root = PROJECT / "artifacts/deployment/sun397"
        deps = setting("SUN397_WORKER_DEPENDENCIES", "worker_dependencies", "")
        caption_cache = setting("SUN397_CAPTION_WEIGHT_CACHE", "caption_weight_cache", "")
        return cls(
            bundles={
                "tacs": Path(setting("SUN397_TACS_BUNDLE", "tacs_bundle", root / "tacs_visual")).resolve(),
                "mtacs": Path(setting("SUN397_MTACS_BUNDLE", "mtacs_bundle", root / "mtacs_full_both")).resolve(),
            },
            python=str(setting("SUN397_WORKER_PYTHON", "worker_python", sys.executable)),
            dependencies=Path(deps).resolve() if deps else None,
            runtime=Path(setting("SUN397_WEB_RUNTIME", "runtime", PROJECT / ".runtime/sun397_demo")).resolve(),
            device=str(setting("SUN397_DEVICE", "device", "auto")),
            autoload=str(setting("SUN397_AUTOLOAD", "autoload", "1")).lower() not in {"0", "false", "no"},
            startup_timeout=float(setting("SUN397_STARTUP_TIMEOUT", "startup_timeout", 1800)),
            prediction_timeout=float(setting("SUN397_PREDICTION_TIMEOUT", "prediction_timeout", 600)),
            threads=int(setting("SUN397_CPU_THREADS", "threads", 4)),
            caption_weight_cache=Path(caption_cache).resolve() if caption_cache else None,
            shared_process=str(setting("SUN397_SHARED_PROCESS", "shared_process", "0")).lower() in {"1","true","yes"},
        )


class ModelWorker:
    def __init__(self, model, settings):
        self.model, self.settings = model, settings
        self.process = None
        self.state = "loading" if settings.autoload else "disabled"
        self.message = "Đang chờ nạp model khi web khởi động."
        self.details = {}
        self._events = queue.Queue()
        self._lock = threading.Lock()
        self._log = None
        self._closed = threading.Event()

    def status(self):
        if self.state == "ready" and self.process is not None and self.process.poll() is not None:
            self.state, self.message = "error", "Worker đã dừng. Xem log và khởi động lại web."
        return {"available": self.state == "ready", "code": self.state,
                "message": self.message, **self.details}

    def _read_events(self):
        try:
            for line in self.process.stdout:
                if line.startswith(EVENT_PREFIX):
                    self._events.put(json.loads(line[len(EVENT_PREFIX):]))
                elif self._log is not None:
                    self._log.write(line)
                    self._log.flush()
        except (OSError, ValueError) as exc:
            self._events.put({"event": "error", "message": str(exc)})
        finally:
            try:
                code = self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                code = self.process.poll()
            self._events.put({"event": "exited", "message": f"Model worker đã kết thúc (exit code: {code})."})

    def _event(self, deadline):
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Hết thời gian chờ model; xem log trong thư mục runtime.")
            try:
                event = self._events.get(timeout=min(remaining, 1))
            except queue.Empty:
                continue
            if event["event"] == "progress":
                self.message = event["message"]
                LOG.info("%s: %s", self.model, self.message)
                continue
            if event["event"] in {"error", "exited"}:
                raise InferenceUnavailable(event.get("message", "Model worker thất bại."))
            return event

    def start(self):
        with self._lock:
            try:
                if self._closed.is_set():
                    return
                root = self.settings.bundles["mtacs" if self.model == "both" else self.model]
                if not (root / "manifest.json").is_file() or not (root / "model.pt").is_file():
                    raise FileNotFoundError(f"Không tìm thấy bundle model: {root}")
                self.settings.runtime.mkdir(parents=True, exist_ok=True)
                logs = self.settings.runtime / "logs"
                logs.mkdir(exist_ok=True)
                self._log = (logs / f"{self.model}.log").open("a", encoding="utf-8", buffering=1)
                self._log.write(f"\n--- New startup: {self.model}, device={self.settings.device} ---\n")
                env = dict(os.environ, PYTHONUTF8="1", PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1",
                           TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                           HF_HUB_DISABLE_TELEMETRY="1")
                env.pop("PYTHONPATH", None)
                if self.settings.dependencies:
                    env["PYTHONPATH"] = str(self.settings.dependencies)
                if self.settings.caption_weight_cache:
                    env["SUN397_CAPTION_WEIGHT_CACHE"] = str(self.settings.caption_weight_cache)
                command = [self.settings.python, "-u", str(Path(__file__).with_name("worker.py")),
                           "--model", self.model, "--bundle", str(root),
                           "--runtime", str(self.settings.runtime), "--device", self.settings.device,
                           "--threads", str(self.settings.threads), "--parent-pid", str(os.getpid())]
                if self.model == "both":
                    command += ["--tacs-bundle", str(self.settings.bundles["tacs"])]
                kwargs = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
                self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                                stderr=self._log, text=True, encoding="utf-8", bufsize=1,
                                                env=env, **kwargs)
                if self._closed.is_set():
                    self.stop()
                    return
                threading.Thread(target=self._read_events, daemon=True).start()
                event = self._event(time.monotonic() + self.settings.startup_timeout)
                if event["event"] != "ready":
                    raise InferenceUnavailable("Worker không xác nhận model đã nạp.")
                self.details = event.get("details", {})
                self.state, self.message = "ready", "Model đã nạp sẵn, có thể dự đoán."
                self._log.write("READY " + json.dumps(self.details) + "\n")
                LOG.info("%s ready: %s", self.model, self.details)
            except Exception as exc:
                self.state, self.message = "error", str(exc)
                LOG.exception("Cannot start %s", self.model)
                self.stop()

    def predict(self, content, model=None):
        with self._lock:
            if not self.status()["available"]:
                raise InferenceUnavailable(self.message)
            requests = self.settings.runtime / "requests"
            requests.mkdir(exist_ok=True)
            try:
                selected_model = model or self.model
                with request_directory(requests, selected_model) as tmp:
                    image, output = tmp / "upload.image", tmp / "result.json"
                    image.write_bytes(content)  # Preserve original bytes for exact self-exclusion/cache lookup.
                    self.process.stdin.write(json.dumps({"command": "predict", "model": selected_model, "image": str(image),
                                                         "output": str(output)}) + "\n")
                    self.process.stdin.flush()
                    event = self._event(time.monotonic() + self.settings.prediction_timeout)
                    if event["event"] == "prediction_error":
                        raise RuntimeError(event["message"])
                    if event["event"] != "result" or not output.is_file():
                        raise RuntimeError("Worker không trả kết quả hợp lệ.")
                    result = json.loads(output.read_text(encoding="utf-8"))
                    if result.get("model") != selected_model or not result.get("prediction"):
                        raise RuntimeError("Worker trả kết quả của model khác hoặc thiếu prediction.")
                    return result
            except (TimeoutError, BrokenPipeError, InferenceUnavailable) as exc:
                self.state, self.message = "error", str(exc)
                self.stop()
                raise InferenceUnavailable(str(exc)) from exc

    def stop(self):
        self._closed.set()
        process = self.process
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            for pipe in (process.stdin, process.stdout):
                if pipe is not None:
                    pipe.close()
        if self._log is not None:
            self._log.close()
            self._log = None


class ModelRoute:
    def __init__(self, model, worker):
        self.model, self.worker = model, worker

    def status(self):
        values = self.worker.status()
        per_model = values.pop("models", {}).get(self.model, {})
        return {**values, **per_model}

    def predict(self, content):
        return self.worker.predict(content, model=self.model)


class InferenceManager:
    def __init__(self, settings=None):
        self.settings = settings or Settings.from_environment()
        self.shared = ModelWorker("both", self.settings) if self.settings.shared_process else None
        self.workers = {name: ModelRoute(name, self.shared) if self.shared else ModelWorker(name, self.settings)
                        for name in ("tacs", "mtacs")}
        self._startup = None
        self._inference_lock = asyncio.Lock()

    async def start(self):
        if self.settings.autoload:
            # Keep HTTP responsive while both workers preload, sequentially to limit peak memory.
            async def preload():
                # Florence/CLIP initialization has the largest temporary peak.
                # Load it before the other L/16 is resident on small machines.
                async with self._inference_lock:
                    if self.shared:
                        await asyncio.to_thread(self.shared.start)
                    else:
                        for name in ("mtacs", "tacs"):
                            await asyncio.to_thread(self.workers[name].start)
            self._startup = asyncio.create_task(preload())

    def status(self, model):
        return self.workers[model].status()

    async def predict(self, model, content):
        if not self.status(model)["available"]:
            raise InferenceUnavailable(self.status(model)["message"])
        async with self._inference_lock:
            return await asyncio.to_thread(self.workers[model].predict, content)

    async def close(self):
        if self._startup is not None:
            self._startup.cancel()
        for worker in ([self.shared] if self.shared else self.workers.values()):
            await asyncio.to_thread(worker.stop)
