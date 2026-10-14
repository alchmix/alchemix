"""Quick local test — imports from gpu_kenburns_simple (same as pipeline)."""
from __future__ import annotations

import os
import sys
import time

from PIL import Image, ImageDraw

from gpu_kenburns_simple import VALID_EFFECTS, create_video_gpu_direct


def _make_test_image(path: str, w: int = 1600, h: int = 900) -> None:
    img = Image.new("RGB", (w, h), (30, 60, 120))
    draw = ImageDraw.Draw(img)
    draw.rectangle([w // 4, h // 4, 3 * w // 4, 3 * h // 4], outline=(255, 200, 0), width=8)
    draw.text((w // 2 - 80, h // 2), "KEN BURNS TEST", fill=(255, 255, 255))
    img.save(path, quality=95)


def main() -> int:
    out_dir = os.path.join(os.path.dirname(__file__), "kb_test_out")
    os.makedirs(out_dir, exist_ok=True)
    img_path = os.path.join(out_dir, "test_photo.jpg")
    _make_test_image(img_path)

    for effect in VALID_EFFECTS:
        out = os.path.join(out_dir, f"kb_{effect}.mp4")
        t0 = time.perf_counter()
        create_video_gpu_direct(
            image_path=img_path,
            duration=8.0,
            effect_type=effect,
            width=1920,
            height=1080,
            fps=30,
            output_path=out,
            speed_multiplier=0.55,
            head_freeze_sec=0.5 if effect == "zoom_in" else 0.0,
        )
        print(f"OK {effect}: {time.perf_counter() - t0:.2f}s -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
