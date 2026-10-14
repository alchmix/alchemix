"""
Resolve Grok Imagine share links to downloadable MP4s.

Share URL: https://grok.com/imagine/post/{uuid}
API:       POST https://grok.com/rest/media/post/get  body {"id": "<uuid>"}

Public share posts work without login (same as opening the link in a browser).
Optional cookies/grok_cookies.json only for private or restricted posts.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import urllib.request
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlparse

GROK_API_BASE = "https://grok.com"
ASSETS_BASE = "https://assets.grok.com"

SHARE_PATTERNS = (
    r"grok\.com/imagine/post/",
    r"grok\.com/.+/imagine/post/",
)

DEFAULT_HEADERS_BASE = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://grok.com",
    "Referer": "https://grok.com/imagine",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
}


def is_grok_imagine_share_link(raw_url: str) -> bool:
    if not raw_url or not str(raw_url).strip().startswith("http"):
        return False
    low = raw_url.lower()
    return any(re.search(p, low) for p in SHARE_PATTERNS)


def parse_grok_post_id(raw_url: str) -> str | None:
    if not raw_url:
        return None
    parsed = urlparse(raw_url.strip())
    parts = [p for p in parsed.path.split("/") if p]
    for i, part in enumerate(parts):
        if part.lower() == "post" and i + 1 < len(parts):
            post_id = parts[i + 1].split("?")[0].strip()
            if re.fullmatch(r"[0-9a-fA-F-]{36}", post_id):
                return post_id
    match = re.search(
        r"imagine/post/([0-9a-fA-F-]{36})",
        raw_url,
        re.I,
    )
    return match.group(1) if match else None


def _cookie_file_path() -> Path:
    env = os.getenv("GROK_COOKIES_FILE", "").strip()
    if env:
        return Path(env)
    return Path(os.getenv("COOKIE_DIR", "cookies")) / "grok_cookies.json"


def _parse_cookie_header(header: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in header.split(";"):
        part = part.strip()
        if not part or "=" not in part:
            continue
        name, _, value = part.partition("=")
        name = name.strip()
        if name:
            out[name] = value.strip()
    return out


def load_grok_cookies(*, silent: bool = True) -> dict[str, str]:
    """
    Load cookies from grok_cookies.json (optional — public shares do not need this):
      { "cookies": { "name": "value", ... } }
      { "cookies": ["name=value; name2=value2"] }
      { "cookie_header": "..." }
    """
    path = _cookie_file_path()
    if not path.is_file():
        if not silent:
            logging.warning(
                f"⚠️ Grok cookies missing ({path}) — only needed for non-public posts"
            )
        return {}

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logging.error(f"❌ Failed to read Grok cookies {path}: {exc}")
        return {}

    if isinstance(data.get("cookie_header"), str):
        return _parse_cookie_header(data["cookie_header"])

    raw = data.get("cookies")
    if isinstance(raw, dict):
        return {str(k): str(v) for k, v in raw.items() if v}

    if isinstance(raw, list):
        merged: dict[str, str] = {}
        for item in raw:
            if isinstance(item, str):
                merged.update(_parse_cookie_header(item))
        return merged

    return {}


def _build_headers(cookies: dict[str, str], referer: str = "https://grok.com/imagine") -> dict[str, str]:
    headers = dict(DEFAULT_HEADERS_BASE)
    headers["Referer"] = referer
    headers["Content-Type"] = "application/json"
    if cookies:
        headers["Cookie"] = "; ".join(f"{k}={v}" for k, v in cookies.items() if v)
    return headers


def _deep_values(data: Any) -> Iterator[dict]:
    if isinstance(data, dict):
        yield data
        for value in data.values():
            yield from _deep_values(value)
    elif isinstance(data, list):
        for value in data:
            yield from _deep_values(value)


def _normalize_asset_url(url: str | None) -> str | None:
    if not url:
        return None
    url = url.strip()
    if url.startswith("http://") or url.startswith("https://"):
        return url
    if url.startswith("/"):
        return f"{ASSETS_BASE}{url}"
    return f"{ASSETS_BASE}/{url.lstrip('/')}"


def _is_video_url(value: str) -> bool:
    low = value.lower()
    return any(
        marker in low
        for marker in (".mp4", ".webm", "/videos/", "generated_video", "video/mp4")
    )


def collect_video_urls(post: dict[str, Any]) -> list[str]:
    """Walk post JSON and collect video CDN URLs (hd preferred first)."""
    hd: list[str] = []
    other: list[str] = []

    for node in _deep_values(post):
        for key in ("hdMediaUrl", "hdVideoUrl"):
            val = node.get(key)
            if isinstance(val, str) and val and _is_video_url(val):
                norm = _normalize_asset_url(val)
                if norm and norm not in hd:
                    hd.append(norm)
        for key in ("videoUrl", "mediaUrl", "url", "original"):
            val = node.get(key)
            if isinstance(val, str) and val and _is_video_url(val):
                norm = _normalize_asset_url(val)
                if norm and norm not in other and norm not in hd:
                    other.append(norm)

    return hd + other


def _post_get_request(post_id: str, cookies: dict[str, str]) -> dict[str, Any] | None:
    share_referer = f"{GROK_API_BASE}/imagine/post/{post_id}"
    api_url = f"{GROK_API_BASE}/rest/media/post/get"
    body = json.dumps({"id": post_id}).encode("utf-8")
    req = urllib.request.Request(
        api_url,
        data=body,
        headers=_build_headers(cookies, referer=share_referer),
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        logging.debug(f"Grok post/get attempt failed for {post_id}: {exc}")
        return None
    post = payload.get("post", payload)
    return post if isinstance(post, dict) else None


def fetch_media_post(post_id: str, cookies: dict[str, str] | None = None) -> dict[str, Any] | None:
    """Public share posts resolve without cookies; retry with cookies if configured."""
    post = _post_get_request(post_id, {})
    if post:
        return post

    extra = cookies if cookies is not None else load_grok_cookies(silent=True)
    if extra:
        post = _post_get_request(post_id, extra)
        if post:
            return post

    logging.error(
        f"❌ Grok post/get failed for {post_id} — post may be private or removed"
    )
    return None


def resolve_grok_imagine_video_url(raw_url: str) -> dict | None:
    """
    Returns dict with post_id, stream_url, title, video_urls — or None.
    """
    post_id = parse_grok_post_id(raw_url)
    if not post_id:
        logging.error(f"❌ Grok link missing post UUID: {raw_url}")
        return None

    post = fetch_media_post(post_id)
    if not post:
        return None

    video_urls = collect_video_urls(post)
    if not video_urls:
        logging.error(f"❌ Grok post {post_id} has no video URLs (image-only or not ready?)")
        return None

    title = (
        post.get("prompt")
        or post.get("title")
        or post.get("originalPrompt")
        or f"Grok {post_id[:8]}"
    )
    if isinstance(title, str) and len(title) > 200:
        title = title[:197] + "..."

    duration = 0.0
    try:
        duration = float(post.get("videoDuration") or 0)
    except (TypeError, ValueError):
        duration = 0.0

    return {
        "post_id": post_id,
        "stream_url": video_urls[0],
        "video_urls": video_urls,
        "title": str(title),
        "poster_url": post.get("thumbnailImageUrl") or "",
        "duration": duration,
    }


def _probe_duration_ffprobe(path: str) -> float:
    try:
        out = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", path,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return float(out.stdout.strip() or 0)
    except Exception:
        return 0.0


def get_grok_video_metadata(raw_url: str) -> dict:
    resolved = resolve_grok_imagine_video_url(raw_url)
    if not resolved:
        raise RuntimeError(f"Could not resolve Grok Imagine link: {raw_url}")
    return {
        "duration": float(resolved.get("duration") or 0),
        "title": resolved.get("title") or "",
    }


def download_grok_stream(
    stream_url: str,
    dest_path: str,
    cookies: dict[str, str] | None = None,
    *,
    referer: str = "https://grok.com/imagine",
) -> str | None:
    cookies = cookies if cookies is not None else load_grok_cookies(silent=True)
    headers = dict(DEFAULT_HEADERS_BASE)
    headers["Referer"] = referer
    if cookies:
        headers["Cookie"] = "; ".join(f"{k}={v}" for k, v in cookies.items() if v)

    try:
        req = urllib.request.Request(stream_url, headers=headers)
        with urllib.request.urlopen(req, timeout=300) as resp, open(dest_path, "wb") as out:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
    except Exception as exc:
        logging.error(f"❌ Grok video download failed: {exc}")
        return None

    if not os.path.isfile(dest_path) or os.path.getsize(dest_path) < 10_000:
        try:
            os.remove(dest_path)
        except OSError:
            pass
        logging.error("❌ Grok download too small (auth expired or moderation page?)")
        return None

    return dest_path


def download_grok_imagine_source(
    raw_url: str,
    dest_path: str,
) -> str | None:
    """Resolve share link and download best MP4 to dest_path."""
    resolved = resolve_grok_imagine_video_url(raw_url)
    if not resolved:
        return None

    referer = f"{GROK_API_BASE}/imagine/post/{resolved['post_id']}"
    logging.info(
        f"📥 Grok Imagine | post={resolved['post_id']} → {os.path.basename(dest_path)}"
    )
    return download_grok_stream(
        resolved["stream_url"],
        dest_path,
        referer=referer,
    )
