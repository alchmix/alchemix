"""
Load D:\\Alchemy Mix\\backend\\.env once at process start.

Import this module (or `import env_loader`) before reading os.getenv().
No PowerShell $env: needed per session — put values in .env instead.
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

_loaded = False


def load_backend_env() -> None:
    global _loaded
    if _loaded:
        return
    backend_dir = Path(__file__).resolve().parent
    env_file = backend_dir / ".env"
    if env_file.is_file():
        load_dotenv(env_file)
    else:
        load_dotenv()  # fallback: cwd .env
    _loaded = True


# Auto-load on first import of this module
load_backend_env()
