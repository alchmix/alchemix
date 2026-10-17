"""
The AlchemyMix side of the Visual Engine integration: login rules, ownership, error pass-through, callbacks.
The engine is faked here (its own tests live in the visual_engine repo).

Run:  python -m pytest -q ve_tests
(folder is not called `tests` on purpose: tests.py is a module in this repo)
"""

from __future__ import annotations

import hashlib
import hmac
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import mongomock
import pytest
from flask import Flask
from flask_jwt_extended import JWTManager, create_access_token

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import visual_engine_client as VC  # noqa: E402
import visual_engine_routes as VR  # noqa: E402

SECRET = "k" * 48


class FakeEngine:
    """Stands in for requests.Session against the engine. Records calls, answers from a script."""

    def __init__(self):
        self.calls = []
        self.answers = {}

    def request(self, method, url, json=None, params=None, headers=None, timeout=None, stream=False):
        path = url.split("127.0.0.1:5001", 1)[-1]
        self.calls.append(SimpleNamespace(method=method, path=path, json=json, headers=headers or {}))
        status, body = self.answers.get((method, path), (200, {}))
        if callable(body):
            body = body(json)
        return SimpleNamespace(status_code=status, json=lambda: body, text=str(body), headers={},
                               iter_content=lambda n: [b"x"])


@pytest.fixture()
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("VE_SHARED_SECRET", SECRET)
    db = mongomock.MongoClient()["AlchemyMix"]
    monkeypatch.setattr(VR, "_db", lambda: db)
    ana = db.users.insert_one({"email": "ana@x.com", "status": "user"}).inserted_id
    db.users.insert_one({"email": "bo@x.com", "status": "user"})
    db.users.insert_one({"email": "boss@x.com", "status": "admin"})
    ch = db.channels.insert_one({"userId": str(ana), "channelName": "Old Times", "niche": "history",
                                 "brandKit": {"voice": "nora", "look": "editorial", "accent": "#E2B36E"}}).inserted_id
    boss = db.users.find_one({"email": "boss@x.com"})["_id"]
    bch = db.channels.insert_one({"userId": str(boss), "channelName": "Boss TV", "niche": "news"}).inserted_id
    fake = FakeEngine()
    VC.set_client(VC.EngineClient("http://127.0.0.1:5001", session=fake))
    VR._catalog_cache.update(at=0.0, data=None)
    app = Flask(__name__)
    app.config.update(JWT_SECRET_KEY="j" * 40, TESTING=True)
    JWTManager(app)
    app.register_blueprint(VR.visual_engine_bp)
    monkeypatch.setattr(VR, "OUTPUT_FOLDER", str(tmp_path))
    with app.app_context():
        tok = {e: create_access_token(identity=e) for e in ("ana@x.com", "bo@x.com", "boss@x.com", "ghost@x.com")}
    yield SimpleNamespace(app=app, c=app.test_client(), db=db, fake=fake, tok=tok, channel=str(ch), out=tmp_path,
                          boss_channel=str(bch))
    VC.set_client(None)


def auth(env, who="ana@x.com"):
    return {"Authorization": f"Bearer {env.tok[who]}"}


def _job(env, _id="ve_1", user="ana@x.com", status="queued", **kw):
    now = datetime.utcnow()
    doc = {"_id": _id, "user_id": user, "title": "T", "status": status, "stage": status, "progress": 0.0,
           "priority": 0, "created_at": now, "queued_at": now, "updated_at": now,
           "request": {"script": "secret words", "voice": "cweb"}, "result": None, "logs": ["a"] * 30, **kw}
    env.db.ve_jobs.insert_one(doc)
    return doc


# ── login and roles ─────────────────────────────────────────────────────────────────────────────────────────────


def test_needs_login(env):
    assert env.c.get("/api/visual-engine/videos").status_code == 401
    assert env.c.get("/api/visual-engine/videos", headers=auth(env, "ghost@x.com")).status_code == 403


def test_service_token_role_comes_from_the_database(env):
    env.fake.answers[("POST", "/v1/videos")] = (202, lambda j: {"video": {"id": "ve_9"}, "created": True})
    env.c.post("/api/visual-engine/videos", json={"title": "T", "script": "s", "priority": 9, "dry_run": True,
                                                  "channel_id": env.channel}, headers=auth(env))
    call = env.fake.calls[-1]
    import jwt

    claims = jwt.decode(call.headers["Authorization"].split()[1], SECRET, algorithms=["HS256"],
                        audience="visual-engine")
    assert claims["sub"] == "ana@x.com" and claims["role"] == "user" and claims["exp"] - claims["iat"] <= 120
    assert "priority" not in call.json and "dry_run" not in call.json  # users can't sneak admin fields through
    env.c.post("/api/visual-engine/videos", json={"title": "T", "script": "s", "dry_run": True,
                                                  "channel_id": env.boss_channel}, headers=auth(env, "boss@x.com"))
    claims = jwt.decode(env.fake.calls[-1].headers["Authorization"].split()[1], SECRET, algorithms=["HS256"],
                        audience="visual-engine")
    assert claims["role"] == "admin" and env.fake.calls[-1].json["dry_run"] is True


def test_admin_routes_are_admin_only(env):
    assert env.c.get("/api/visual-engine/admin/videos", headers=auth(env)).status_code == 403
    assert env.c.get("/api/visual-engine/admin/videos", headers=auth(env, "boss@x.com")).status_code == 200


# ── creating ────────────────────────────────────────────────────────────────────────────────────────────────────


def test_create_uses_the_channel_brand_kit_with_overrides(env):
    env.fake.answers[("POST", "/v1/videos")] = (202, lambda j: {"video": {"id": "ve_9"}, "created": True})
    r = env.c.post("/api/visual-engine/videos", json={"title": "T", "script": "s", "channel_id": env.channel,
                                                      "genre": "history", "status": "admin", "user_id": "bo@x.com"},
                   headers={**auth(env), "Idempotency-Key": "abc-123-xyz"})
    assert r.status_code == 202
    call = env.fake.calls[-1]
    assert call.json == {"title": "T", "script": "s", "channel_id": env.channel, "channel_name": "Old Times",
                         "niche": "history", "voice": "nora", "look": "editorial", "accent": "#E2B36E",
                         "genre": "history"}
    assert call.headers["Idempotency-Key"] == "abc-123-xyz"
    assert env.db.channels.find_one({"channelName": "Old Times"})["brandKit"].get("genre") is None  # not saved


def test_channel_is_required_and_brand_kit_can_be_saved(env):
    r = env.c.post("/api/visual-engine/videos", json={"title": "T", "script": "s"}, headers=auth(env))
    assert r.status_code == 400 and r.get_json()["error"]["code"] == "channel_required"
    env.fake.answers[("POST", "/v1/videos")] = (202, lambda j: {"video": {"id": "ve_9"}, "created": True})
    env.c.post("/api/visual-engine/videos", json={"title": "T", "script": "s", "channel_id": env.channel,
                                                  "look": "tech", "save_brand_kit": True}, headers=auth(env))
    assert env.db.channels.find_one({"channelName": "Old Times"})["brandKit"]["look"] == "tech"
    chans = env.c.get("/api/visual-engine/channels", headers=auth(env)).get_json()["channels"]
    assert [c["name"] for c in chans] == ["Old Times"] and chans[0]["brand_kit"]["voice"] == "nora"
    r = env.c.put(f"/api/visual-engine/channels/{env.channel}/brand-kit", json={"voice": "liam", "bogus": "x"},
                  headers=auth(env))
    assert r.get_json()["channel"]["brand_kit"] == {"voice": "liam"}
    assert env.c.put(f"/api/visual-engine/channels/{env.channel}/brand-kit", json={},
                     headers=auth(env, "bo@x.com")).status_code == 404


def test_the_channels_own_voice_is_used(env):
    env.fake.answers[("POST", "/v1/videos")] = (202, lambda j: {"video": {"id": "ve_9"}, "created": True})
    env.db.channels.update_one({"channelName": "Old Times"}, {"$set": {"brandKit": {"look": "editorial"},
                                "voice": "henry", "speechify_voice_name": "Henry", "logo": "a_logo.png"}})
    ch = env.c.get("/api/visual-engine/channels", headers=auth(env)).get_json()["channels"][0]
    assert ch["voice"] == {"id": "henry", "name": "Henry", "provider": "speechify"}
    assert ch["logo_url"] == "/channel_media/a_logo.png"
    env.c.post("/api/visual-engine/videos", json={"title": "T", "script": "s", "channel_id": env.channel},
               headers=auth(env))
    assert env.fake.calls[-1].json["voice"] == "henry" and env.fake.calls[-1].json["voice_name"] == "Henry"
    # a voice chosen for this video wins over the channel's
    env.c.post("/api/visual-engine/videos", json={"title": "T", "script": "s", "channel_id": env.channel,
                                                  "voice": "liam", "voice_name": "Liam"}, headers=auth(env))
    assert env.fake.calls[-1].json["voice"] == "liam"


def test_a_cartesia_channel_voice_asks_for_a_speechify_one(env, monkeypatch):
    import types
    import sys

    monkeypatch.setitem(sys.modules, "helper.cartesia",
                        types.SimpleNamespace(CARTESIA_VOICES={"carson": "86e30c1d-714b-4074-a1f2-1cb6b552fb49"}))
    monkeypatch.setattr(VR, "_CARTESIA", None)
    env.db.channels.update_one({"channelName": "Old Times"}, {"$set": {"brandKit": {},
                                "voice": "86e30c1d-714b-4074-a1f2-1cb6b552fb49"}})
    ch = env.c.get("/api/visual-engine/channels", headers=auth(env)).get_json()["channels"][0]
    assert ch["voice"]["provider"] == "cartesia"
    r = env.c.post("/api/visual-engine/videos", json={"title": "T", "script": "s", "channel_id": env.channel},
                   headers=auth(env))
    assert r.status_code == 400 and r.get_json()["error"]["code"] == "voice_not_supported"


def test_someone_elses_channel_is_refused(env):
    r = env.c.post("/api/visual-engine/videos", json={"title": "T", "script": "s", "channel_id": env.channel},
                   headers=auth(env, "bo@x.com"))
    assert r.status_code == 404 and r.get_json()["error"]["code"] == "unknown_channel"
    r = env.c.post("/api/visual-engine/videos", json={"title": "T", "script": "s", "channel_id": "not-an-id"},
                   headers=auth(env))
    assert r.status_code == 404


def test_engine_errors_pass_through(env):
    env.fake.answers[("POST", "/v1/videos")] = (429, {"error": {"code": "daily_limit", "message": "Limit",
                                                                 "retry_after_sec": 3600}})
    r = env.c.post("/api/visual-engine/videos", json={"title": "T", "script": "s", "channel_id": env.channel},
                   headers=auth(env))
    assert r.status_code == 429 and r.get_json()["error"]["code"] == "daily_limit"
    assert r.headers["Retry-After"] == "3600"


def test_engine_down_is_a_clear_502(env):
    import requests

    class Down:
        def request(self, *a, **k):
            raise requests.ConnectionError("refused")

    VC.set_client(VC.EngineClient("http://127.0.0.1:5001", session=Down()))
    r = env.c.post("/api/visual-engine/videos", json={"title": "T", "script": "s", "channel_id": env.channel},
                   headers=auth(env))
    assert r.status_code == 502 and r.get_json()["error"]["code"] == "engine_unavailable"


def test_missing_secret_is_a_clear_503(env, monkeypatch):
    monkeypatch.setenv("VE_SHARED_SECRET", "short")
    r = env.c.get("/api/visual-engine/options", headers=auth(env))
    assert r.status_code == 503 and r.get_json()["error"]["code"] == "not_configured"


# ── reading ─────────────────────────────────────────────────────────────────────────────────────────────────────


def test_reads_are_owner_scoped_and_hide_the_script(env):
    _job(env, "ve_1")
    _job(env, "ve_2", user="bo@x.com")
    r = env.c.get("/api/visual-engine/videos", headers=auth(env)).get_json()
    assert [v["id"] for v in r["videos"]] == ["ve_1"]
    v = r["videos"][0]
    assert "script" not in v["request"] and "logs" not in v and v["queue_position"] == 1
    assert env.c.get("/api/visual-engine/videos/ve_2", headers=auth(env)).status_code == 404
    assert env.c.get("/api/visual-engine/videos/ve_2", headers=auth(env, "boss@x.com")).status_code == 200


def test_users_get_plain_errors_admins_get_details(env):
    _job(env, "ve_4", status="failed", error={"code": "render_failed", "message": "ffmpeg C:/vm/storage/x.mp4"})
    u = env.c.get("/api/visual-engine/videos/ve_4", headers=auth(env)).get_json()["video"]
    assert u["error"]["code"] == "render_failed" and "C:/" not in u["error"]["message"]
    a = env.c.get("/api/visual-engine/videos/ve_4", headers=auth(env, "boss@x.com")).get_json()["video"]
    assert "C:/vm" in a["error"]["message"] and a["logs"]


def test_completed_video_links_and_download(env):
    (env.out / "ve_3.mp4").write_bytes(b"0" * 10000)
    _job(env, "ve_3", status="completed", result={"publish": "shared", "video_file": "ve_3.mp4",
                                                  "thumbnail_file": "ve_3.jpg", "video_path": "C:/secret/path"})
    v = env.c.get("/api/visual-engine/videos/ve_3", headers=auth(env)).get_json()["video"]
    assert v["result"]["video_url"] == "/videos/ve_3.mp4" and "video_path" not in v["result"]
    r = env.c.get("/api/visual-engine/videos/ve_3/download", headers=auth(env))
    assert r.status_code == 200 and len(r.data) == 10000 and "attachment" in r.headers["Content-Disposition"]
    assert env.c.get("/api/visual-engine/videos/ve_3/download", headers=auth(env, "bo@x.com")).status_code == 404


# ── callbacks ───────────────────────────────────────────────────────────────────────────────────────────────────


def _signed(env, ev: dict, *, secret=SECRET, ts=None):
    raw = json.dumps(ev).encode()
    ts = str(int(time.time()) if ts is None else ts)
    mac = hmac.new(secret.encode(), f"{ts}.".encode() + raw, hashlib.sha256).hexdigest()
    return env.c.post("/api/visual-engine/callback", data=raw, content_type="application/json",
                      headers={"X-VE-Timestamp": ts, "X-VE-Signature": f"v1={mac}"})


def _completed_event(eid="evt_1"):
    return {"event_id": eid, "type": "video.completed", "occurred_at": "2026-10-17T10:00:00Z",
            "job": {"id": "ve_5", "user_id": "ana@x.com", "title": "The Crash", "status": "completed",
                    "result": {"publish": "shared", "video_file": "ve_5.mp4", "thumbnail_file": "ve_5.jpg",
                               "duration_sec": 312.0},
                    "request": {"niche": "history", "channel_id": "c1"}}}


def test_callback_records_the_video_once(env, monkeypatch):
    sent = []
    monkeypatch.setattr(VR, "_notify", lambda kind, job: sent.append(kind))
    assert _signed(env, _completed_event()).status_code == 200
    rec = env.db.completed_videos.find_one({"ve_job_id": "ve_5"})
    assert rec["video_path"] == "ve_5.mp4" and rec["user_id"] == "ana@x.com" and rec["source"] == "visual_engine"
    r = _signed(env, _completed_event())  # the engine retried a delivered event
    assert r.get_json()["duplicate"] is True
    _signed(env, _completed_event("evt_2"))  # a different event for the same video
    assert env.db.completed_videos.count_documents({"ve_job_id": "ve_5"}) == 1
    assert sent == ["video.completed", "video.completed"]


@pytest.mark.parametrize("kw", [{"secret": "z" * 48}, {"ts": int(time.time()) - 3600}])
def test_callback_rejects_bad_signatures_and_replays(env, kw):
    assert _signed(env, _completed_event(), **kw).status_code == 401
    assert env.db.completed_videos.count_documents({}) == 0


def test_callback_without_signature(env):
    r = env.c.post("/api/visual-engine/callback", json=_completed_event())
    assert r.status_code == 401


# ── the manual flow: package → plan → render ────────────────────────────────────────────────────────────────────


def test_package_plan_and_render_are_forwarded(env):
    _job(env, "ve_7", status="awaiting_plan", phase="voice", package={"beats": 3, "files": []})
    env.fake.answers[("GET", "/v1/videos/ve_7/package")] = (200, {"files": [{"name": "x"}], "copy_all": "all"})
    r = env.c.get("/api/visual-engine/videos/ve_7/package", headers=auth(env))
    assert r.get_json()["copy_all"] == "all"
    v = env.c.get("/api/visual-engine/videos/ve_7", headers=auth(env)).get_json()["video"]
    assert v["step"] == "plan" and v["package"]["beats"] == 3
    env.fake.answers[("POST", "/v1/videos/ve_7/plan")] = (200, {"video": {"id": "ve_7"}})
    env.c.post("/api/visual-engine/videos/ve_7/plan", json={"plan": "{\"beats\": []}"}, headers=auth(env))
    assert env.fake.calls[-1].json == {"plan": "{\"beats\": []}"}
    env.fake.answers[("POST", "/v1/videos/ve_7/render")] = (409, {"error": {"code": "plan_not_ready",
                                                                           "message": "Paste first"}})
    r = env.c.post("/api/visual-engine/videos/ve_7/render", headers=auth(env))
    assert r.status_code == 409 and r.get_json()["error"]["code"] == "plan_not_ready"


def test_steps(env):
    assert VR.step_of({"status": "running", "phase": "voice"}) == "voice"
    assert VR.step_of({"status": "awaiting_plan", "phase": "voice"}) == "plan"
    assert VR.step_of({"status": "running", "phase": "render"}) == "rendering"
    assert VR.step_of({"status": "completed", "phase": "render"}) == "done"
    assert VR.step_of({"status": "canceled", "phase": "render"}) == "plan"


def test_timings_ready_callback_notifies(env, monkeypatch):
    sent = []
    monkeypatch.setattr(VR, "_notify", lambda kind, job: sent.append(kind))
    ev = {"event_id": "evt_t", "type": "video.timings_ready", "job": {"id": "ve_8", "user_id": "ana@x.com",
                                                                      "title": "T", "status": "awaiting_plan"}}
    assert _signed(env, ev).status_code == 200
    assert sent == ["video.timings_ready"] and env.db.completed_videos.count_documents({}) == 0


def test_cartesia_ids_are_read_from_the_file_when_the_module_cannot_load(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "helper.cartesia", None)  # importing it now raises ImportError
    monkeypatch.setattr(VR, "_CARTESIA", None)
    assert VR._cartesia_voices().get("carson") == "86e30c1d-714b-4074-a1f2-1cb6b552fb49"
