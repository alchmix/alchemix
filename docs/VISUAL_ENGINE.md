# Visual Engine integration (documentary videos)

The Visual Engine is a separate service on the same machine (`http://127.0.0.1:5001`, repo `alchmix/visual_engine`).
This backend is the only thing that talks to it. Users log in here as usual; `visual_engine_routes.py` checks who
they are, forwards their request signed with `VE_SHARED_SECRET`, and turns the engine's signed callbacks into
`completed_videos` records and Telegram notices.

## .env (this backend)

```
VE_ENGINE_URL=http://127.0.0.1:5001
VE_SHARED_SECRET=<same long random value as the engine's .env>
# optional
VE_CALLBACK_SECRET=<defaults to VE_SHARED_SECRET>
VE_OUTPUT_FOLDER=outputs          # where the engine copies finished videos (its VE_PUBLISH_DIR)
```

## API for the frontend

All need the normal login token (`Authorization: Bearer <token>`). Errors are always
`{"error": {"code": "...", "message": "..."}}` — show `message`, branch on `code`.

The frontend pages are **Documentaries** (`/documentaries`, `/documentaries/new`, `/documentaries/<id>`).
Manual flow: script + channel → voice & timings → copy the package to Claude → paste Claude's JSON plan → render.

| method | path | notes |
|---|---|---|
| GET | `/api/visual-engine/options` | voices, looks, genres, visual modes, limits → build the form from this |
| GET | `/api/visual-engine/channels` | my channels with their saved brand kit (`brand_kit`: voice, look, genre, accent, notes, visual_mode) |
| PUT | `/api/visual-engine/channels/<id>/brand-kit` | save a channel's brand kit |
| POST | `/api/visual-engine/videos` | `{title, script, channel_id, voice?, look?, genre?, visual_mode?, accent?, brand_notes?, save_brand_kit?}` — `channel_id` required; missing fields come from the channel's brand kit. Send an `Idempotency-Key` header so a double click never makes two videos. 202 → `{video}` |
| GET | `/api/visual-engine/videos` | my videos, newest first. `?status=…`, `?before=<id>` for the next page |
| GET | `/api/visual-engine/videos/<id>` | poll every 4–10 s while `status` is `queued`/`running`. `step` tells the page what to show: `voice`, `plan`, `rendering`, `done`, `voice_failed`, `render_failed`, `canceled` |
| GET | `/api/visual-engine/videos/<id>/package` | `{files: [{name, title, content}], copy_all}` — what the user gives Claude (timings + rulebook always together) |
| GET | `/api/visual-engine/videos/<id>/package.zip` | the same as a zip |
| POST | `/api/visual-engine/videos/<id>/plan` | `{plan: "<pasted text>"}` → `{video}` with `plan: {version, ok, blocking[], warnings[], info[], scenes}` |
| POST | `/api/visual-engine/videos/<id>/render` | only when `plan.ok`; also "render again" after done |
| POST | `/api/visual-engine/videos/<id>/cancel` | queued: at once; running: within seconds |
| POST | `/api/visual-engine/videos/<id>/retry` | failed/canceled only, limited per video |
| GET | `/api/visual-engine/videos/<id>/download` | the MP4 as a download |

Telegram: "⏱ Timings ready" when the package is ready, then the usual done/failed messages.

A finished video also has `result.video_url` (`/videos/ve_….mp4`, for the player) and `result.thumbnail_url`.

Error codes worth a specific message: `too_many_active`, `hourly_limit`, `daily_limit` (429, with `Retry-After`),
`script_too_short`, `script_too_long`, `unknown_voice`, `unknown_channel`, `queue_full` (503),
`engine_unavailable` (502: the engine is not running), `retry_limit`, `not_cancelable`, `channel_required`,
`plan_not_ready`, `render_not_allowed`, `render_limit`, `plan_not_allowed`, `package_missing`.

Video `status`: `queued` → `running` → `completed` | `failed` | `canceled`. Failed videos carry
`error.code` + a plain `error.message` (raw details are admin-only).

## Admin

`GET /api/visual-engine/admin/status` (engine readiness + queue), `GET /api/visual-engine/admin/videos?user=…`,
`POST …/admin/videos/<id>/priority {"priority": 10}`, `POST …/admin/videos/<id>/fail {"reason"}`,
`POST …/admin/callbacks/resend`, `GET …/admin/audit?job_id=…`.
Admins may also send `"dry_run": true` with a video request: a 6-second test video through the whole system
(no voice, no AI, no downloads) — the quickest way to check the setup.

## Also changed in this branch

- `/api/register` now needs a logged-in user (the Sign-up page is already behind login), and only an admin can
  create another admin. Before, anyone on the internet could POST `"status": "admin"` without logging in.
  The Sign-up page must send the login token (frontend change in the same work). **Check for existing self-made
  admins:** `db.users.find({status: "admin"}, {email: 1})`.
- `DELETE /api/completed-videos/cleanup` now needs an admin (it was open to anyone).

## Tests

`python -m pytest -q ve_tests` (the folder is not called `tests`: `tests.py` is a module here).
