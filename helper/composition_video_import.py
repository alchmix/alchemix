"""Resolve remote video URLs for composition bulk import (duration + title)."""

from __future__ import annotations

import logging
import subprocess

import yt_dlp


def _ffprobe_duration(media_url: str) -> float:
    if not media_url:
        return 0.0
    try:
        out = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                media_url,
            ],
            stderr=subprocess.STDOUT,
            timeout=120,
        )
        return float(out.decode().strip() or 0)
    except Exception as exc:
        logging.warning("ffprobe duration failed for %s: %s", media_url[:80], exc)
        return 0.0


def resolve_video_import_metadata(raw_url: str) -> dict:
    """
    Return { url, title, duration } for a composition Videos / timeline import.
    `url` is always the original link the backend will download at render time.
    """
    url = (raw_url or "").strip()
    if not url:
        raise ValueError("url is required")

    from helper.avatar_extractor import (
        is_dreamface_share_link,
        resolve_dreamface_video_url,
    )
    from helper.grok_imagine_extractor import (
        get_grok_video_metadata,
        is_grok_imagine_share_link,
    )

    if is_grok_imagine_share_link(url):
        meta = get_grok_video_metadata(url)
        duration = float(meta.get("duration") or 0)
        if duration <= 0.05:
            raise RuntimeError("Could not read Grok video duration")
        return {
            "url": url,
            "title": (meta.get("title") or "").strip(),
            "duration": duration,
        }

    if is_dreamface_share_link(url):
        resolved = resolve_dreamface_video_url(url)
        if not resolved:
            raise RuntimeError("Could not resolve DreamFace share link")
        stream = resolved.get("stream_url") or ""
        duration = _ffprobe_duration(stream)
        if duration <= 0.05:
            raise RuntimeError("Could not read DreamFace video duration")
        return {
            "url": url,
            "title": (resolved.get("work_name") or "").strip(),
            "duration": duration,
        }

    ydl_opts = {
        "skip_download": True,
        "quiet": True,
        "no_warnings": True,
        "ignore_no_formats_error": True,
    }
    last_err = None
    for attempt in range(2):
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
            duration = float(info.get("duration") or 0)
            if duration <= 0.05 and url.lower().startswith("http"):
                duration = _ffprobe_duration(url)
            if duration <= 0.05:
                raise RuntimeError("Could not determine video duration")
            return {
                "url": url,
                "title": (info.get("title") or "").strip(),
                "duration": duration,
            }
        except Exception as exc:
            last_err = exc
            if attempt == 0 and url.lower().startswith("http"):
                duration = _ffprobe_duration(url)
                if duration > 0.05:
                    return {"url": url, "title": "", "duration": duration}

    raise RuntimeError(str(last_err or "Unsupported or unreachable URL"))
