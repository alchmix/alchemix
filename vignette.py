import cv2
import numpy as np
import torch
import subprocess
import threading
from pathlib import Path
from queue import Queue
from multiprocessing import Process, Queue as MPQueue, cpu_count
import math
import os
import time

# ============================================================
# CONFIG
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_FOLDER = Path("outputs")
OUTPUT_FOLDER.mkdir(exist_ok=True)

BATCH_SIZE = 120
NUM_READ_PROCS = max(2, cpu_count() // 2)
PREFETCH_BATCHES = 4
FP16 = True   # enable mixed precision for faster processing

print(f"🚀 Using device: {DEVICE} | FP16={FP16} | CPU decoders={NUM_READ_PROCS}")

# ============================================================
# GPU VIGNETTE MASK
# ============================================================
def make_vignette_mask_gpu(w, h, vignette_strength):
    inner = 0.40
    edge_alpha = min(0.99, max(0.0, vignette_strength))
    y = torch.linspace(0, h - 1, h, device=DEVICE)
    x = torch.linspace(0, w - 1, w, device=DEVICE)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    cx, cy = w / 2.0, h / 2.0
    r = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    max_radius = math.sqrt(cx**2 + cy**2)
    rN = r / max_radius
    ramp = torch.clamp((rN - inner) / (1.0 - inner), 0, 1)
    alpha = edge_alpha * ramp
    mask = 1.0 - alpha
    mask = mask.unsqueeze(2).repeat(1, 1, 3)
    return mask.half() if FP16 else mask

# ============================================================
# GPU OVERLAY PRELOAD
# ============================================================
def preload_overlays_gpu(overlays, w, h):
    gpu_overlays = []
    for ov in overlays:
        src = ov.get("path") or ov.get("url")
        if not src or not os.path.exists(src):
            print(f"⚠️ Skipping invalid overlay: {src}")
            continue
        img = cv2.imread(src)
        if img is None:
            print(f"⚠️ Could not load overlay: {src}")
            continue
        img = cv2.resize(img, (w, h))
        t = torch.from_numpy(img.astype(np.float32) / 255.0).to(DEVICE)
        if FP16:
            t = t.half()
        gpu_overlays.append({
            "tensor": t,
            "start": ov.get("start", 0),
            "end": ov.get("end", float("inf")),
            "opacity": ov.get("opacity", 1.0)
        })
    return gpu_overlays

# ============================================================
# CPU FRAME READER (MULTIPROCESS)
# ============================================================
def frame_reader(video_path, start, end, batch_size, q: MPQueue):
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames, ids = [], []
    for fid in range(start, end):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        ids.append(fid)
        if len(frames) >= batch_size:
            q.put((frames.copy(), ids.copy()))
            frames, ids = [], []
    if frames:
        q.put((frames.copy(), ids.copy()))
    cap.release()
    q.put(None)

# ============================================================
# GPU PROCESSING
# ============================================================
def apply_overlays_gpu(batch_gpu, overlay_tensors, frame_ids, fps):
    for i, fid in enumerate(frame_ids):
        t = fid / fps
        frame_gpu = batch_gpu[i]
        for ov in overlay_tensors:
            if not (ov["start"] <= t < ov["end"]):
                continue
            opacity = ov["opacity"]
            frame_gpu = frame_gpu * (1 - opacity) + ov["tensor"] * opacity
        batch_gpu[i] = frame_gpu
    return batch_gpu

def process_batch_gpu(batch_frames, batch_ids, mask_gpu, overlays, fps):
    np_batch = np.stack(batch_frames).astype(np.float32) / 255.0
    batch_gpu = torch.from_numpy(np_batch).to(DEVICE)
    if FP16:
        batch_gpu = batch_gpu.half()
    with torch.cuda.stream(torch.cuda.Stream()):
        batch_gpu *= mask_gpu.unsqueeze(0)
        if overlays:
            batch_gpu = apply_overlays_gpu(batch_gpu, overlays, batch_ids, fps)
    batch_cpu = (batch_gpu.float().cpu().numpy() * 255).astype(np.uint8)
    return batch_cpu

# ============================================================
# AUDIO MERGE
# ============================================================
def add_audio_track(original_video, processed_video, output_path):
    cmd = [
        "ffmpeg", "-y",
        "-i", processed_video,
        "-i", original_video,
        "-c:v", "copy",
        "-c:a", "aac",
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-shortest",
        output_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# ============================================================
# MAIN PIPELINE
# ============================================================
def apply_edits_gpu_ultrafast(
    video_path,
    overlays=[],
    vignette=0.6,
    output_name_hint="ultrafast_gpu_out.mp4",
):
    base_path = Path(video_path)
    if not base_path.exists():
        raise FileNotFoundError(video_path)

    cap = cv2.VideoCapture(str(base_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    print(f"🎞️ {w}x{h} @ {fps:.1f}fps, {total_frames} frames")

    mask_gpu = make_vignette_mask_gpu(w, h, vignette)
    overlay_tensors = preload_overlays_gpu(overlays, w, h)

    out_temp = OUTPUT_FOLDER / f"temp_{output_name_hint}"
    out_final = OUTPUT_FOLDER / output_name_hint

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{w}x{h}",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "h264_nvenc",
        "-preset", "p4",
        "-cq", "23",
        "-pix_fmt", "yuv420p",
        str(out_temp)
    ]

    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    print(f"⚡ NVENC encoding started → {out_temp}")

    # --------------------------------------------------------
    # Multiprocess readers
    # --------------------------------------------------------
    frame_queue = MPQueue(maxsize=PREFETCH_BATCHES)
    chunk = total_frames // NUM_READ_PROCS
    readers = []
    for i in range(NUM_READ_PROCS):
        start = i * chunk
        end = total_frames if i == NUM_READ_PROCS - 1 else (i + 1) * chunk
        p = Process(target=frame_reader, args=(video_path, start, end, BATCH_SIZE, frame_queue))
        p.daemon = True
        p.start()
        readers.append(p)

    finished_procs = 0
    processed = 0
    t0 = time.time()

    while True:
        item = frame_queue.get()
        if item is None:
            finished_procs += 1
            if finished_procs >= NUM_READ_PROCS:
                break
            continue

        batch_frames, batch_ids = item
        batch_out = process_batch_gpu(batch_frames, batch_ids, mask_gpu, overlay_tensors, fps)
        for frame in batch_out:
            proc.stdin.write(frame.tobytes())

        processed += len(batch_out)
        if processed % 300 == 0:
            pct = processed / total_frames * 100
            print(f"⏳ {processed}/{total_frames} ({pct:.1f}%)")

    proc.stdin.close()
    proc.wait()
    dt = time.time() - t0
    print(f"✅ Done processing {processed} frames in {dt:.1f}s")

    print("🔊 Adding audio track...")
    add_audio_track(video_path, str(out_temp), str(out_final))
    os.remove(out_temp)
    print(f"🎉 Final output: {out_final}")
    return str(out_final)

# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available!")

    print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")

    overlays = [
        # Example overlay
        # {"path": "overlay.png", "start": 0, "end": 5, "opacity": 0.5},
    ]

    apply_edits_gpu_ultrafast(
        video_path="outputs/final_20251012_155916.mp4",
        overlays=overlays,
        vignette=0.75,
        output_name_hint="ultra_full_parallel.mp4"
    )
