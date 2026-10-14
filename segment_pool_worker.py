"""
Picklable ProcessPool entry point for segment encoding.

Avoid `import test` — collides with Python stdlib `test` in child processes.

Optional env (auto-discovery works if unset):
  VIDEO_ENCODE_MODULE=create_video
  VIDEO_ENCODE_PATH=D:/Alchemy Mix/backend/tests.py
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import os
import sys

import env_loader  # noqa: F401 — VIDEO_ENCODE_PATH from .env

_MODULE_NAME: str | None = None
_RESOLVED_PATH: str | None = None
_RESOLVED = False
_LAST_IMPORT_ERROR: str | None = None

_SKIP_SCAN = {
    "segment_pool_worker.py",
    "global_render_scheduler.py",
    "video_job_controller.py",
    "worker_config.py",
    "wsgi.py",
    "app.py",
}

_PREFERRED_FILES = (
    "create_video.py",
    "tests.py",
    "video_engine.py",
    "video_pipeline.py",
    "test.py",
)


def _backend_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _module_has_worker(mod) -> bool:
    return hasattr(mod, "_encode_segment_pool_worker") or hasattr(
        mod, "encode_video_segment_with_retry"
    )


def _mod_key_for_path(path: str) -> str:
    path = os.path.normpath(os.path.abspath(path))
    return f"_video_encode_{abs(hash(path))}"


def _evict_loaded_path(path: str) -> None:
    """Drop a failed or partial exec_module from sys.modules (same-process reuse)."""
    sys.modules.pop(_mod_key_for_path(path), None)


def _load_module_from_path(path: str):
    global _LAST_IMPORT_ERROR
    path = os.path.normpath(os.path.abspath(path))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Video encode file not found: {path}")
    mod_key = _mod_key_for_path(path)
    cached = sys.modules.get(mod_key)
    if cached is not None:
        if _module_has_worker(cached):
            return cached, path
        _evict_loaded_path(path)
    spec = importlib.util.spec_from_file_location(mod_key, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_key] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        import traceback

        _evict_loaded_path(path)
        _LAST_IMPORT_ERROR = (
            f"{type(exc).__name__}: {exc}\n"
            f"{traceback.format_exc()[-2500:]}"
        )
        raise
    if not _module_has_worker(mod):
        _evict_loaded_path(path)
        raise ImportError(
            f"{path} finished exec_module but has no segment encode worker symbols"
        )
    _LAST_IMPORT_ERROR = None
    return mod, path


def _try_load_path(path: str) -> str | None:
    global _LAST_IMPORT_ERROR
    _LAST_IMPORT_ERROR = None
    try:
        mod, resolved = _load_module_from_path(path)
    except Exception as exc:
        _evict_loaded_path(path)
        logging.warning(
            "Encode module import failed for %s: %s",
            path,
            exc,
            exc_info=True,
        )
        return None
    if _module_has_worker(mod):
        return resolved
    _evict_loaded_path(path)
    logging.warning(
        "Encode file loaded but missing worker symbols: %s "
        "(need _encode_segment_pool_worker or encode_video_segment_with_retry)",
        path,
    )
    return None


def _discover_encode_file() -> str | None:
    base = _backend_dir()
    for fname in _PREFERRED_FILES:
        p = os.path.join(base, fname)
        if os.path.isfile(p):
            found = _try_load_path(p)
            if found:
                return found
    for fname in sorted(os.listdir(base)):
        if not fname.endswith(".py") or fname in _SKIP_SCAN:
            continue
        found = _try_load_path(os.path.join(base, fname))
        if found:
            return found
    return None


def _candidate_module_names() -> list[str]:
    env = os.environ.get("VIDEO_ENCODE_MODULE", "").strip()
    if env:
        return [env]
    return ["create_video", "video_engine", "video_pipeline"]


def _file_has_encode_worker(path: str) -> bool:
    """Check encode markers without importing the module (tail-read — worker is ~line 14k)."""
    markers = (
        "_encode_segment_pool_worker",
        "encode_video_segment_with_retry",
        "def encode_video_segment(",
    )
    try:
        size = os.path.getsize(path)
        read_bytes = min(size, 900_000)
        with open(path, encoding="utf-8", errors="ignore") as fh:
            if size > read_bytes:
                fh.seek(size - read_bytes)
            chunk = fh.read()
        return any(m in chunk for m in markers)
    except OSError:
        return False


def _resolve_env_encode_path() -> str | None:
    raw = os.environ.get("VIDEO_ENCODE_PATH", "").strip()
    if not raw:
        return None
    candidates = [raw, os.path.normpath(raw)]
    seen: set[str] = set()
    for p in candidates:
        if p in seen:
            continue
        seen.add(p)
        if os.path.isfile(p):
            return p
    return None


def validate_encode_module_path() -> str:
    """
    Verify encode module exists without exec_module (avoids loading 18k-line test.py
    in the parent Flask process at scheduler init).
    """
    env_path = _resolve_env_encode_path()
    if env_path:
        logging.info(f"🎬 Validating encode module: {env_path}")
        if _file_has_encode_worker(env_path):
            logging.info(f"🎬 Encode module validated: {env_path}")
            return env_path
        raise ImportError(
            f"VIDEO_ENCODE_PATH={env_path} missing segment encode worker "
            f"(need _encode_segment_pool_worker or encode_video_segment). "
            f"Copy latest test.py → tests.py"
        )
    raw = os.environ.get("VIDEO_ENCODE_PATH", "").strip()
    if raw:
        logging.warning(f"⚠️ VIDEO_ENCODE_PATH not found: {raw}")

    discovered = _discover_encode_file()
    if discovered:
        logging.info(f"🎬 Encode module validated (discovered): {discovered}")
        return discovered

    return resolve_encode_module_name()


def resolve_encode_module_name() -> str:
    """Find/load the module that exposes segment encode workers."""
    global _MODULE_NAME, _RESOLVED_PATH, _RESOLVED
    if _RESOLVED:
        return _RESOLVED_PATH or _MODULE_NAME or "<resolved>"

    env_path = os.environ.get("VIDEO_ENCODE_PATH", "").strip()
    if env_path:
        if os.path.isfile(env_path):
            found = _try_load_path(env_path)
            if found:
                _RESOLVED_PATH = found
                _RESOLVED = True
                logging.info(f"🎬 Segment worker loaded from VIDEO_ENCODE_PATH: {found}")
                return found
            if _LAST_IMPORT_ERROR:
                raise ImportError(
                    f"VIDEO_ENCODE_PATH={env_path} import failed during exec_module:\n"
                    f"{_LAST_IMPORT_ERROR}"
                )
            raise ImportError(
                f"VIDEO_ENCODE_PATH={env_path} imported OK but has no "
                f"_encode_segment_pool_worker / encode_video_segment_with_retry"
            )
        logging.warning(
            f"⚠️ VIDEO_ENCODE_PATH not found: {env_path} — auto-discovering in {_backend_dir()}"
        )

    discovered = _discover_encode_file()
    if discovered:
        _RESOLVED_PATH = discovered
        _RESOLVED = True
        logging.info(f"🎬 Segment worker auto-discovered: {discovered}")
        return discovered

    errors: list[str] = []
    for name in _candidate_module_names():
        try:
            mod = importlib.import_module(name)
        except ImportError as exc:
            errors.append(f"{name}: {exc}")
            continue
        if _module_has_worker(mod):
            _MODULE_NAME = name
            _RESOLVED = True
            logging.info(f"🎬 Segment worker module: {name}")
            return name
        errors.append(f"{name}: missing encode worker functions")

    py_files = [f for f in os.listdir(_backend_dir()) if f.endswith(".py")]
    raise ImportError(
        "Could not find video encode module. Set VIDEO_ENCODE_PATH to the .py file "
        f"that contains create_video() (e.g. tests.py). Python files in backend: {py_files}. "
        f"Module tries: {errors}"
    )


def _get_encode_module():
    if not _RESOLVED:
        resolve_encode_module_name()
    if _RESOLVED_PATH:
        mod_key = _mod_key_for_path(_RESOLVED_PATH)
        mod = sys.modules.get(mod_key)
        if mod is not None and _module_has_worker(mod):
            return mod
        if mod is not None:
            _evict_loaded_path(_RESOLVED_PATH)
        mod, _ = _load_module_from_path(_RESOLVED_PATH)
        return mod
    return importlib.import_module(_MODULE_NAME)


def get_encode_attr(name: str):
    """Load attribute from VIDEO_ENCODE_PATH module — never `from tests import`."""
    mod = _get_encode_module()
    obj = getattr(mod, name, None)
    if obj is None:
        raise ImportError(
            f"{_RESOLVED_PATH or _MODULE_NAME} has no {name!r}; "
            "set VIDEO_ENCODE_PATH to your tests.py"
        )
    return obj


def get_create_video():
    return get_encode_attr("create_video")


def run_segment_encode(segment_info):
    """Top-level worker invoked by ProcessPoolExecutor child processes."""
    os.environ["_SEGMENT_POOL_CHILD"] = "1"
    segment_index = segment_info[5]
    logging.info(f"👷 Worker start | seg {segment_index} | pid={os.getpid()}")
    mod = _get_encode_module()
    if hasattr(mod, "_encode_segment_pool_worker"):
        result = mod._encode_segment_pool_worker(segment_info)
    else:
        path = mod.encode_video_segment_with_retry(segment_info)
        result = segment_index, path
    logging.info(f"👷 Worker done | seg {segment_index} | pid={os.getpid()}")
    return result
