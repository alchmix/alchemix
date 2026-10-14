"""
Single MongoDB access point per process — connect lazily, reuse MongoClient pool.

Importing this module must never block on Atlas or leave ``mongo_instance`` undefined.
Set MONGODB_URI (or MONGO_URI) in backend/.env — do not hardcode credentials here.
"""

from __future__ import annotations

import logging
import os
import threading

import certifi
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.errors import PyMongoError

logger = logging.getLogger(__name__)

_connect_lock = threading.Lock()


def _connection_uri() -> str:
    uri = (os.getenv("MONGODB_URI") or os.getenv("MONGO_URI") or "").strip()
    if not uri:
        raise RuntimeError(
            "MONGODB_URI or MONGO_URI is not set (check backend/.env)"
        )
    return uri


def _database_name() -> str:
    return (
        os.getenv("MONGODB_DB")
        or os.getenv("MONGO_DB_NAME")
        or "AlchemyMix"
    ).strip()


class MongoDB:
    """One lazy client per process — same idea as a shared MongoClient in Node."""

    def __init__(self) -> None:
        self._client: MongoClient | None = None
        self._db: Database | None = None

    def _ensure_connected(self) -> None:
        if self._db is not None:
            return
        with _connect_lock:
            if self._db is not None:
                return
            uri = _connection_uri()
            db_name = _database_name()
            timeout_ms = max(
                5000, int(os.getenv("MONGODB_SERVER_SELECTION_MS", "15000"))
            )
            pool_size = max(10, int(os.getenv("MONGODB_MAX_POOL_SIZE", "50")))
            socket_ms = max(
                timeout_ms,
                int(os.getenv("MONGODB_SOCKET_TIMEOUT_MS", "45000")),
            )
            client = MongoClient(
                uri,
                tls=True,
                tlsCAFile=certifi.where(),
                serverSelectionTimeoutMS=timeout_ms,
                connectTimeoutMS=timeout_ms,
                socketTimeoutMS=socket_ms,
                maxPoolSize=pool_size,
                retryWrites=True,
            )
            client.admin.command("ping")
            self._client = client
            self._db = client[db_name]
            logger.info("Successfully connected to MongoDB Atlas")
            print("✓ Successfully connected to MongoDB Atlas")

    def get_db(self) -> Database:
        try:
            self._ensure_connected()
        except PyMongoError as e:
            logger.error("Failed to connect to MongoDB: %s", e)
            print(f"❌ Failed to connect to MongoDB: {e}")
            raise
        except Exception as e:
            logger.exception("Unexpected error during MongoDB connection: %s", e)
            print(f"❌ Unexpected error during MongoDB connection: {e}")
            raise
        assert self._db is not None
        return self._db

    def get_client(self) -> MongoClient:
        self.get_db()
        assert self._client is not None
        return self._client


# Always defined — safe for ``from mongodb import mongo_instance`` in workers.
# No network I/O until the first get_db() / get_client() call.
mongo_instance = MongoDB()
