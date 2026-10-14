"""
YouTube download quality demo — current pipeline vs proposed pipeline.

Downloads the SAME short range of a YouTube video twice and reports, for each:
  * which stream yt-dlp actually picked (format id, resolution, codec)
  * download time, encode time, total time
  * the final 1920x1080 clip (same filter graph + NVENC settings as test.py)

Then saves a side-by-side frame so you can see the difference.

Usage (from C:\\Users\\HP 50\\Pictures\\test):
    python scripts\\yt_quality_demo.py
    python scripts\\yt_quality_demo.py "https://www.youtube.com/watch?v=..." --start 60 --length 10
    python scripts\\yt_quality_demo.py URL --list-only      # just show available resolutions

Nothing here touches the production code; output goes to scripts\\yt_demo_out\\.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

try:
    import yt_dlp
except ImportError:
    print("yt-dlp is not installed:  pip install -U yt-dlp")
    raise SystemExit(1)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_ROOT = os.path.join(HERE, "yt_demo_out")


# ─────────────────────────────────────────────────────────────── shared bits
def _find(name, fallbacks):
    path = shutil.which(name)
    if path:
        return path
    for fb in fallbacks:
        p = os.path.expandvars(os.path.expanduser(fb))
        if os.path.isfile(p):
            return p
    return None


def js_runtimes() -> dict:
    """Same detection as test.py (_js_runtime_opts)."""
    rt = {}
    deno = _find("deno", (r"%USERPROFILE%\.deno\bin\deno.exe", r"C:\Program Files\deno\deno.exe",
                          r"C:\Program Files (x86)\deno\deno.exe"))
    if deno:
        rt["deno"] = {"path": deno}
    node = _find("node", (r"C:\Program Files\nodejs\node.exe", r"C:\Program Files (x86)\nodejs\node.exe"))
    if node:
        rt["node"] = {"path": node}
    return rt or {"deno": {}}


def common_opts(cookies: bool) -> dict:
    o = {
        "retries": 3,
        "fragment_retries": 10,
        "concurrent_fragment_downloads": 8,
        "noprogress": True,
        "quiet": True,
        "no_warnings": False,
        "js_runtimes": js_runtimes(),
        "remote_components": {"ejs:github"},
    }
    if cookies:
        o["cookiesfrombrowser"] = ("firefox",)
    return o


# CURRENT — copied from test.py _base_ydl_opts + download_video_segment
CURRENT_FORMAT = (
    "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/"
    "bestvideo[height<=1080]+bestaudio/"
    "best[height<=1080][ext=mp4]/best"
)
CURRENT_CLIENTS = ["web", "mweb", "default"]

# PROPOSED — best stream up to 1080p, prefer H.264 (fast NVDEC/CPU decode),
# the 360p muxed file is only a last resort; yt-dlp picks the player clients.
NEW_FORMAT = "bv*[height<=1080]+ba/b[height<=1080]/bv*+ba/b"
NEW_SORT = ["res:1080", "vcodec:h264", "fps", "acodec:m4a"]
# tried in order until one gives >= 720p: (player clients, use cookies)
# None = let yt-dlp choose. Some clients (android_vr, ios) only work WITHOUT cookies.
NEW_CLIENT_RETRIES = [
    (None, True),
    (["tv"], True),
    (["web_safari"], True),
    (None, False),
    (["android_vr"], False),
    (["ios"], False),
    (["tv"], False),
]
# client setups probed by the format listing
PROBE_CLIENTS = [
    ("CURRENT web+mweb+default", CURRENT_CLIENTS),
    ("yt-dlp default", None),
    ("tv", ["tv"]),
    ("web_safari", ["web_safari"]),
    ("android_vr", ["android_vr"]),
    ("ios", ["ios"]),
]
PAD_SEC = 2.0  # downloaded around the cut; exact trim happens in the single encode


def nvenc_available() -> bool:
    try:
        r = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=20)
        if "h264_nvenc" not in r.stdout:
            return False
        t = subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=s=256x256:d=0.2",
                            "-c:v", "h264_nvenc", "-f", "null", "-"], capture_output=True, text=True, timeout=30)
        return t.returncode == 0
    except Exception:
        return False


def encoder_args(use_nvenc: bool):
    if use_nvenc:  # identical to test.py _nvenc_args()
        return ["-c:v", "h264_nvenc", "-preset", "p3", "-tune", "hq",
                "-rc", "vbr", "-b:v", "8M", "-maxrate", "12M", "-bufsize", "24M",
                "-profile:v", "high", "-pix_fmt", "yuv420p",
                "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709"]
    return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p"]


# same graph as test.py _build_filter_graph (no crop / effect / logo)
FILTER = (
    "[0:v]split=2[main][blur];"
    "[blur]scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,"
    "boxblur=luma_radius=min(h\\,w)/20:luma_power=1:"
    "chroma_radius=min(cw\\,ch)/20:chroma_power=1[bg];"
    "[main]null[processed];"
    "[processed]scale=1920:1080:force_original_aspect_ratio=decrease[scaled];"
    "[bg][scaled]overlay=(W-w)/2:(H-h)/2,crop=h=iw*9/16,"
    "eq=contrast=1.05:saturation=1.15:brightness=0.01[video]"
)


def encode(src, dst, start, duration, use_nvenc):
    pre = max(0.0, start - 3.0)
    fine = start - pre
    cmd = ["ffmpeg", "-y", "-v", "error", "-ss", f"{pre:.3f}", "-i", src, "-ss", f"{fine:.3f}",
           "-t", f"{duration:.3f}", "-filter_complex", FILTER, "-map", "[video]", "-map", "0:a?",
           *encoder_args(use_nvenc), "-c:a", "aac", "-b:a", "192k",
           "-movflags", "+faststart", "-shortest", dst]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-1200:])
    return time.time() - t0


def probe(path):
    r = subprocess.run(["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", "-show_format", path],
                       capture_output=True, text=True)
    data = json.loads(r.stdout or "{}")
    v = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), {})
    return {
        "width": v.get("width"), "height": v.get("height"), "codec": v.get("codec_name"),
        "duration": float(data.get("format", {}).get("duration") or 0),
        "size_mb": round(os.path.getsize(path) / 1e6, 2) if os.path.exists(path) else 0,
    }


def picked(info):
    fmts = info.get("requested_formats") or [info]
    v = next((f for f in fmts if f.get("vcodec") not in (None, "none")), fmts[0])
    return {
        "format_id": info.get("format_id"),
        "height": v.get("height"), "width": v.get("width"),
        "vcodec": v.get("vcodec"), "fps": v.get("fps"),
    }


def ydl_download(url, opts):
    with yt_dlp.YoutubeDL(opts) as ydl:
        return ydl.extract_info(url, download=True)


def find_output(base):
    for ext in (".mp4", ".mkv", ".webm"):
        if os.path.exists(base + ext):
            return base + ext
    if os.path.exists(base):
        return base
    return None


# ─────────────────────────────────────────────────────────────── runs
def run_current(url, start, length, out_dir, cookies, use_nvenc):
    raw_base = os.path.join(out_dir, "current_raw")
    opts = common_opts(cookies)
    opts.update({
        "format": CURRENT_FORMAT,
        "merge_output_format": "mp4",
        "outtmpl": raw_base + ".%(ext)s",
        "extractor_args": {"youtube": {"player_client": CURRENT_CLIENTS}},
        "download_ranges": lambda _, __: [{"start_time": start, "end_time": start + length}],
        "force_keyframes_at_cuts": True,
    })
    t0 = time.time()
    info = ydl_download(url, opts)
    dl = time.time() - t0
    raw = find_output(raw_base)
    if not raw:
        raise RuntimeError("current: no file downloaded")
    dur = probe(raw)["duration"]
    final = os.path.join(out_dir, "current_final.mp4")
    enc = encode(raw, final, 0.0, dur, use_nvenc)
    return {"picked": picked(info), "raw": probe(raw), "final": final,
            "download_s": dl, "encode_s": enc, "notes": "exact cut + keyframe re-encode inside yt-dlp, then NVENC"}


def run_new(url, start, length, out_dir, cookies, use_nvenc):
    a = max(0.0, start - PAD_SEC)
    b = start + length + PAD_SEC
    tries = []
    info = raw = None
    best_h = -1
    t0 = time.time()
    seen = set()
    for clients, want_cookies in NEW_CLIENT_RETRIES:
        use_ck = cookies and want_cookies
        key = (tuple(clients or ()), use_ck)
        if key in seen:
            continue
        seen.add(key)
        raw_base = os.path.join(out_dir, "new_raw_" + ("default" if clients is None else clients[0])
                                + ("_ck" if use_ck else "_nock"))
        opts = common_opts(use_ck)
        opts.update({
            "format": NEW_FORMAT,
            "format_sort": NEW_SORT,
            "merge_output_format": "mp4",
            "outtmpl": raw_base + ".%(ext)s",
            "download_ranges": lambda _, __, a=a, b=b: [{"start_time": a, "end_time": b}],
        })
        if clients:
            opts["extractor_args"] = {"youtube": {"player_client": clients}}
        label = ("yt-dlp default" if clients is None else "+".join(clients)) + (" +cookies" if use_ck else " no-cookies")
        try:
            got = ydl_download(url, opts)
            path = find_output(raw_base)
            h = picked(got).get("height") or (probe(path)["height"] if path else 0) or 0
            tries.append(f"{label}: {h}p")
            if path and h > best_h:
                info, raw, best_h = got, path, h
            if path and h >= 720:
                break
        except Exception as e:
            tries.append(f"{label}: failed ({str(e).splitlines()[0][:90]})")
    dl = time.time() - t0
    if not raw:
        raise RuntimeError("new: no file downloaded (" + "; ".join(tries) + ")")
    final = os.path.join(out_dir, "new_final.mp4")
    enc = encode(raw, final, start - a, length, use_nvenc)
    return {"picked": picked(info), "raw": probe(raw), "final": final,
            "download_s": dl, "encode_s": enc, "notes": "padded range, exact trim in the one NVENC pass",
            "client_tries": tries}


def _max_heights(url, clients, use_cookies):
    opts = common_opts(use_cookies)
    opts.update({"skip_download": True, "ignore_no_formats_error": True, "quiet": True, "no_warnings": True})
    if clients:
        opts["extractor_args"] = {"youtube": {"player_client": clients}}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    vids = [f for f in info.get("formats", []) if f.get("vcodec") not in (None, "none") and f.get("height")]
    return max([f["height"] for f in vids] or [0]), sorted({f["height"] for f in vids})


def list_formats(url, cookies):
    print("\nBest available resolution per player client  (cookies on | cookies off):")
    best = (0, "")
    for label, clients in PROBE_CLIENTS:
        cells = []
        for ck in ((True, False) if cookies else (False,)):
            try:
                top, heights = _max_heights(url, clients, ck)
                cells.append(f"{str(top) + 'p' if top else 'none':>6s}")
                if top > best[0]:
                    best = (top, f"{label} ({'cookies' if ck else 'no cookies'})")
            except Exception as e:
                msg = str(e).splitlines()[0].replace("ERROR: ", "")
                msg = msg.split(": ", 2)[-1][:38]
                cells.append(f"  fail ({msg})")
        print(f"  {label:26s} " + " | ".join(cells))
    if best[0]:
        print(f"  -> best: {best[0]}p via {best[1]}")
    else:
        print("  -> no client returned real video streams (see tips at the end)")
    return best[0]


def runtime_report():
    """JS runtime health — YouTube's challenge solver needs Deno, or Node 22+."""
    lines = []
    for name, info in js_runtimes().items():
        path = info.get("path") or shutil.which(name)
        ver = "?"
        if path:
            try:
                ver = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=15).stdout.strip().splitlines()[0]
            except Exception:
                pass
        lines.append(f"{name} {ver}")
        if name == "node":
            try:
                major = int(ver.lstrip("v").split(".")[0])
                if major < 22:
                    lines.append("WARNING: Node < 22 cannot run the YouTube challenge solver")
            except ValueError:
                pass
    if "deno" not in js_runtimes():
        lines.append("tip: Deno is the runtime yt-dlp recommends (winget install DenoLand.Deno)")
    return lines


TIPS = """
If every client above shows 360p / none, YouTube is withholding the real streams. In order:
  1. pip install -U yt-dlp            (YouTube changes weekly; old builds lose formats)
  2. winget install DenoLand.Deno     (then open a NEW terminal) - yt-dlp's preferred JS runtime
  3. Re-run with --no-cookies         (stale/flagged browser cookies cause "page needs to be reloaded")
  4. Still stuck: add a PO-token provider plugin, see
     https://github.com/yt-dlp/yt-dlp/wiki/PO-Token-Guide
"""


def side_by_side(a, b, t, dst):
    cmd = ["ffmpeg", "-y", "-v", "error", "-ss", f"{t:.2f}", "-i", a, "-ss", f"{t:.2f}", "-i", b,
           "-filter_complex",
           "[0:v]drawtext=text='CURRENT':x=30:y=30:fontsize=56:fontcolor=white:box=1:boxcolor=black@0.6[l];"
           "[1:v]drawtext=text='PROPOSED':x=30:y=30:fontsize=56:fontcolor=white:box=1:boxcolor=black@0.6[r];"
           "[l][r]hstack=inputs=2",
           "-frames:v", "1", dst]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:  # drawtext may lack a font on some builds
        cmd = ["ffmpeg", "-y", "-v", "error", "-ss", f"{t:.2f}", "-i", a, "-ss", f"{t:.2f}", "-i", b,
               "-filter_complex", "[0:v][1:v]hstack=inputs=2", "-frames:v", "1", dst]
        subprocess.run(cmd, capture_output=True, text=True)
    return os.path.exists(dst)


def show(name, r):
    p, raw = r["picked"], r["raw"]
    res = f"{p.get('width')}x{p.get('height')}" if p.get("height") else "size not reported"
    print(f"\n  {name}")
    print(f"    stream picked : format {p.get('format_id')}  {res}  "
          f"{p.get('vcodec') or ''}  {str(p.get('fps')) + 'fps' if p.get('fps') else ''}")
    print(f"    downloaded    : {raw['width']}x{raw['height']} {raw['codec']}  "
          f"{raw['duration']:.1f}s  {raw['size_mb']} MB")
    if r.get("client_tries"):
        print(f"    client tries  : {' | '.join(r['client_tries'])}")
    print(f"    download time : {r['download_s']:.1f}s")
    print(f"    encode time   : {r['encode_s']:.1f}s")
    print(f"    TOTAL         : {r['download_s'] + r['encode_s']:.1f}s   ({r['notes']})")


def main():
    ap = argparse.ArgumentParser(description="Compare current vs proposed YouTube download quality.")
    ap.add_argument("url", nargs="?")
    ap.add_argument("--start", type=float, default=None, help="clip start in seconds (default 30)")
    ap.add_argument("--length", type=float, default=None, help="clip length in seconds (default 10)")
    ap.add_argument("--no-cookies", action="store_true", help="don't read Firefox cookies")
    ap.add_argument("--list-only", action="store_true", help="only list available resolutions")
    args = ap.parse_args()

    url = args.url or input("YouTube URL: ").strip()
    if not url:
        print("No URL given.")
        return 1
    if args.start is None and not args.list_only and not args.url:
        s = input("Start second [30]: ").strip()
        args.start = float(s) if s else 30.0
        n = input("Length in seconds [10]: ").strip()
        args.length = float(n) if n else 10.0
    start = 30.0 if args.start is None else args.start
    length = 10.0 if args.length is None else args.length
    cookies = not args.no_cookies

    print(f"yt-dlp {yt_dlp.version.__version__} | Firefox cookies: {'on' if cookies else 'off'}")
    for line in runtime_report():
        print("  " + line)
    best_available = list_formats(url, cookies)
    if args.list_only:
        if best_available < 720:
            print(TIPS)
        return 0

    out_dir = os.path.join(OUT_ROOT, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    use_nvenc = nvenc_available()
    print(f"\nEncoder: {'h264_nvenc (same as production)' if use_nvenc else 'libx264 (NVENC not available here)'}")
    print(f"Clip: {start:.1f}s -> {start + length:.1f}s\nOutput: {out_dir}")

    results = {}
    for name, fn in (("CURRENT", run_current), ("PROPOSED", run_new)):
        print(f"\nRunning {name}...")
        try:
            results[name] = fn(url, start, length, out_dir, cookies, use_nvenc)
        except Exception as e:
            print(f"  {name} FAILED: {e}")

    print("\n" + "=" * 72 + "\nRESULTS")
    for name, r in results.items():
        show(name, r)

    if len(results) == 2:
        png = os.path.join(out_dir, "compare_side_by_side.png")
        if side_by_side(results["CURRENT"]["final"], results["PROPOSED"]["final"], min(length / 2, 5), png):
            print(f"\n  Side-by-side frame: {png}")
        hc = results["CURRENT"]["raw"].get("height") or 0
        hn = results["PROPOSED"]["raw"].get("height") or 0
        tc = results["CURRENT"]["download_s"] + results["CURRENT"]["encode_s"]
        tn = results["PROPOSED"]["download_s"] + results["PROPOSED"]["encode_s"]
        print(f"\n  Source resolution: {hc}p -> {hn}p   |   time: {tc:.1f}s -> {tn:.1f}s ({tn - tc:+.1f}s)")

    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump({"url": url, "start": start, "length": length, "results": results}, f, indent=2, default=str)
    print(f"\nFiles: {out_dir}")
    if best_available < 720:
        print(TIPS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())