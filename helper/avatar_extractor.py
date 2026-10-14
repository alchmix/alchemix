"""
Resolve DreamFace share links to direct MP4 URLs.

Share page: https://www.dreamfaceapp.com/creation?previewId=...
API:       GET {base}/dw-server/work/get_work_detail_web?work_id={previewId}
"""
from __future__ import annotations

import json
import logging
import os
import re
import urllib.request
from urllib.parse import parse_qs, urlparse

DREAMFACE_API_BASES = (
    "https://aidreamface.com",
    "https://www.dreamfaceapp.com",
    "https://tools.dreamfaceapp.com",
)

SHARE_LINK_PATTERNS = (
    r"dreamfaceapp\.com/creation",
    r"dreamfaceapp\.com/.+previewId",
)

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.dreamfaceapp.com/",
}


def is_dreamface_share_link(raw_url: str) -> bool:
    if not raw_url or not str(raw_url).startswith("http"):
        return False
    return any(re.search(p, raw_url) for p in SHARE_LINK_PATTERNS)


def parse_preview_id(raw_url: str) -> str | None:
    if not raw_url:
        return None
    parsed = urlparse(raw_url.strip())
    qs = parse_qs(parsed.query)
    preview_vals = qs.get("previewId") or qs.get("previewid")
    if preview_vals and preview_vals[0]:
        return preview_vals[0].strip()
    match = re.search(r"previewId=([a-f0-9]+)", raw_url, re.I)
    return match.group(1) if match else None


def fetch_work_detail(preview_id: str) -> dict | None:
    if not preview_id:
        return None

    for base in DREAMFACE_API_BASES:
        api_url = f"{base}/dw-server/work/get_work_detail_web?work_id={preview_id}"
        try:
            req = urllib.request.Request(api_url, headers=DEFAULT_HEADERS)
            with urllib.request.urlopen(req, timeout=30) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            if payload.get("status_code") == "THS12140000000" and payload.get("data"):
                logging.info(f"✅ DreamFace work detail | previewId={preview_id} | base={base}")
                return payload["data"]
        except Exception as exc:
            logging.warning(f"⚠️ DreamFace API {base} failed: {exc}")

    return None


def resolve_dreamface_video_url(raw_url: str) -> dict | None:
    """
    Returns dict with stream_url, preview_id, work_name, poster_url — or None.
    """
    preview_id = parse_preview_id(raw_url)
    if not preview_id:
        logging.error(f"❌ DreamFace share link missing previewId: {raw_url}")
        return None

    detail = fetch_work_detail(preview_id)
    if not detail:
        return None

    stream_url = detail.get("nw_work_url") or detail.get("work_url")
    if not stream_url:
        logging.error(f"❌ DreamFace work detail has no video URL | previewId={preview_id}")
        return None

    return {
        "preview_id": preview_id,
        "stream_url": stream_url,
        "work_name": detail.get("work_name") or "",
        "poster_url": detail.get("work_webp_path") or detail.get("user_image_url") or "",
        "work_type": detail.get("work_type") or detail.get("work_detail_type") or "",
    }


def extract_avatar_from_share_link(
    raw_url: str,
    output_dir: str = "static/avatars",
    url_key: str | None = None,
    *,
    apply_tag_strip: bool = True,
) -> str | None:
    """Download DreamFace avatar MP4 to a local path for the render pipeline."""
    resolved = resolve_dreamface_video_url(raw_url)
    if not resolved:
        return None

    os.makedirs(output_dir, exist_ok=True)
    key_suffix = url_key or resolved["preview_id"]
    filename = f"dreamface_{resolved['preview_id']}_{key_suffix}.mp4"
    save_path = os.path.join(output_dir, filename)

    if os.path.exists(save_path) and os.path.getsize(save_path) >= 100_000:
        size_mb = os.path.getsize(save_path) / (1024 * 1024)
        logging.info(
            f"♻️ DreamFace avatar cache hit | {save_path} ({size_mb:.1f}MB)"
        )
        if apply_tag_strip:
            from helper.dreamface_tag_strip import maybe_strip_dreamface_avatar
            return maybe_strip_dreamface_avatar(save_path)
        return save_path

    try:
        req = urllib.request.Request(resolved["stream_url"], headers=DEFAULT_HEADERS)
        with urllib.request.urlopen(req, timeout=180) as resp, open(save_path, "wb") as out:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
    except Exception as exc:
        logging.error(f"❌ DreamFace download failed: {exc}")
        return None

    if not os.path.exists(save_path) or os.path.getsize(save_path) < 100_000:
        try:
            os.remove(save_path)
        except OSError:
            pass
        logging.error("❌ DreamFace download too small or missing")
        return None

    size_mb = os.path.getsize(save_path) / (1024 * 1024)
    logging.info(f"✅ DreamFace avatar saved | {save_path} ({size_mb:.1f}MB)")
    if apply_tag_strip:
        from helper.dreamface_tag_strip import maybe_strip_dreamface_avatar
        return maybe_strip_dreamface_avatar(save_path)
    return save_path
