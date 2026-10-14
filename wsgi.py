"""
WSGI entry point — MUST stay lightweight at import time.

On Windows, ProcessPoolExecutor uses spawn: child processes re-import the main
module (this file if you start with `python wsgi.py`). Never import Flask/app
at module level — that loads MongoDB, CUDA, etc. in every segment worker
and causes OOM when the pool recycles.

Production servers:
  waitress-serve --listen=*:5000 wsgi:app
  gunicorn wsgi:app
"""

from __future__ import annotations

import env_loader  # noqa: F401 — loads backend/.env before app import


def create_app():
    """Load Flask application (heavy imports happen here only)."""
    from app import app as flask_app

    return flask_app


def __getattr__(name: str):
    if name == "app":
        return create_app()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == "__main__":
    application = create_app()
    application.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
