# render.py

from pycaps import TemplateLoader, PictexSubtitleRenderer

def main():
    # 🔁 Change this path to whatever input video you want to process
    input_path = "final_20251201_031654.mp4"
    template_name = "minimalist"   # or any other template you want

    print("🟦 Loading template in builder mode...")
    # ❗ IMPORTANT: load(False) → returns a CapsPipelineBuilder, not a pre-built pipeline
    builder = (
        TemplateLoader(template_name)
        .with_input_video(input_path)
        .load(False)   # <- DO NOT remove the False
    )

    print(f"Loaded builder: {builder}")

    # Attach PictexSubtitleRenderer so it does NOT use Playwright/Chromium
    print("🟩 Attaching PictexSubtitleRenderer (no browser)...")
    if hasattr(builder, "with_custom_subtitle_renderer"):
        builder = builder.with_custom_subtitle_renderer(PictexSubtitleRenderer())
    else:
        raise AttributeError(
            "Builder has no 'with_custom_subtitle_renderer' method. "
            "Make sure you're using the GitHub pycaps (git+https...) version."
        )

    # Build pipeline
    print("🟧 Building pipeline...")
    if hasattr(builder, "build"):
        pipeline = builder.build()
    else:
        # Fallback in case load(False) ever changes behaviour
        pipeline = builder

    # Run pipeline
    print("▶️ Running pipeline...")
    output_path = pipeline.run()

    print(f"✅ Done! Output video: {output_path}")

if __name__ == "__main__":
    main()
