"""
Single gateway for loading the video encode module — ALWAYS by file path.

Never use:
  from tests import …
  import tests
  import test

Those collide with PyPI/stdlib packages in worker processes.

Set in .env:
  VIDEO_ENCODE_PATH=D:/Alchemy Mix/backend/video_engine.py
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import os
import sys

import env_loader  # noqa: F401

CANONICAL_ENCODE_FILENAME = "video_engine.py"
FORBIDDEN_ENCODE_FILENAMES = frozenset({"tests.py", "test.py"})

_RESOLVED_PATH: str | None = None
_RESOLVED = False

_SKIP_SCAN = frozenset({
    "segment_pool_worker.py",
    "global_render_scheduler.py",
    "video_job_controller.py",
    "worker_config.py",
    "wsgi.py",
    "app.py",
    "tests.py",
    "test.py",
})

_PREFERRED_FILES = (
    "video_engine.py",
    "create_video.py",
    "video_pipeline.py",
)


def backend_dir() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _module_key(path: str) -> str:
    return f"_alchemy_video_encode_{abs(hash(os.path.normpath(path)))}"


def warn_import_collisions() -> None:
    """Log warnings if PyPI/stdlib packages would shadow encode code."""
    try:
        spec = importlib.util.find_spec("tests")
        if spec and spec.origin and "site-packages" in spec.origin.replace("\\", "/"):
            logging.warning(
                "⚠️ PyPI package 'tests' is installed (%s). "
                "Run: pip uninstall tests -y — never use `from tests import`.",
                spec.origin,
            )
    except (ImportError, ValueError, ModuleNotFoundError):
        pass

    env_path = (os.environ.get("VIDEO_ENCODE_PATH") or "").strip()
    if env_path and os.path.basename(env_path).lower() in FORBIDDEN_ENCODE_FILENAMES:
        logging.warning(
            "⚠️ VIDEO_ENCODE_PATH points at %s — rename to %s and update .env "
            "to avoid import collisions forever.",
            os.path.basename(env_path),
            CANONICAL_ENCODE_FILENAME,
        )


def _load_module_from_path(path: str):
    path = os.path.normpath(os.path.abspath(path))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Video encode file not found: {path}")
    mod_key = _module_key(path)
    if mod_key in sys.modules:
        return sys.modules[mod_key], path
    spec = importlib.util.spec_from_file_location(mod_key, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_key] = mod
    spec.loader.exec_module(mod)
    return mod, path


def _module_has_worker(mod) -> bool:
    return hasattr(mod, "_encode_segment_pool_worker") or hasattr(
        mod, "encode_video_segment_with_retry"
    )


def _try_load_path(path: str) -> str | None:
    try:
        mod, resolved = _load_module_from_path(path)
    except Exception as exc:
        logging.debug(f"Skip encode path {path}: {exc}")
        return None
    if _module_has_worker(mod):
        return resolved
    return None


def _resolve_env_encode_path() -> str | None:
    raw = (os.environ.get("VIDEO_ENCODE_PATH") or "").strip()
    if not raw:
        return None
    for p in (raw, os.path.normpath(raw)):
        if os.path.isfile(p):
            return p
    return None


def _file_has_encode_worker(path: str) -> bool:
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


def _discover_encode_file() -> str | None:
    base = backend_dir()
    for fname in _PREFERRED_FILES:
        found = _try_load_path(os.path.join(base, fname))
        if found:
            return found
    for fname in sorted(os.listdir(base)):
        if not fname.endswith(".py") or fname in _SKIP_SCAN:
            continue
        found = _try_load_path(os.path.join(base, fname))
        if found:
            return found
    return None


def resolve_encode_path() -> str:
    """Return absolute path to validated encode module. Loads module on first call."""
    global _RESOLVED_PATH, _RESOLVED
    if _RESOLVED and _RESOLVED_PATH:
        return _RESOLVED_PATH

    warn_import_collisions()

    env_path = _resolve_env_encode_path()
    if env_path:
        found = _try_load_path(env_path)
        if found:
            _RESOLVED_PATH = found
            _RESOLVED = True
            logging.info(f"🎬 Video encode module: {found}")
            return found
        raise ImportError(
            f"VIDEO_ENCODE_PATH={env_path} missing encode worker "
            f"(need encode_video_segment_with_retry or _encode_segment_pool_worker)"
        )

    discovered = _discover_encode_file()
    if discovered:
        _RESOLVED_PATH = discovered
        _RESOLVED = True
        logging.info(f"🎬 Video encode module (auto): {discovered}")
        return discovered

    raise ImportError(
        f"Set VIDEO_ENCODE_PATH to .../{CANONICAL_ENCODE_FILENAME} "
        f"(copy test.py → video_engine.py on deploy). "
        f"Backend dir: {backend_dir()}"
    )


def validate_encode_module_path() -> str:
    """Lightweight validation for scheduler init — may not exec full module."""
    warn_import_collisions()
    env_path = _resolve_env_encode_path()
    if env_path:
        if _file_has_encode_worker(env_path):
            logging.info(f"🎬 Encode module validated: {env_path}")
            return env_path
        raise ImportError(f"VIDEO_ENCODE_PATH invalid: {env_path}")
    discovered = _discover_encode_file()
    if discovered and _file_has_encode_worker(discovered):
        logging.info(f"🎬 Encode module validated (discovered): {discovered}")
        return discovered
    return resolve_encode_path()


def get_encode_module():
    path = resolve_encode_path()
    mod_key = _module_key(path)
    mod = sys.modules.get(mod_key)
    if mod is not None:
        return mod
    mod, _ = _load_module_from_path(path)
    return mod


def get_encode_attr(name: str):
    mod = get_encode_module()
    obj = getattr(mod, name, None)
    if obj is None:
        raise ImportError(
            f"{resolve_encode_path()} has no attribute {name!r}. "
            f"Deploy latest {CANONICAL_ENCODE_FILENAME}."
        )
    return obj


def get_create_video():
    return get_encode_attr("create_video")
