import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import subprocess
import math
import random
import logging
from typing import Literal
import threading
import queue

logger = logging.getLogger(__name__)

# ============================================================================
# GPU HELPER FUNCTIONS
# ============================================================================

@torch.no_grad()
def gaussian_blur_gpu(img_tensor: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    """Apply Gaussian blur on GPU using conv2d"""
    channels = img_tensor.shape[1]
    
    kernel = _get_gaussian_kernel2d(kernel_size, sigma).to(img_tensor.device)
    kernel = kernel.repeat(channels, 1, 1, 1)
    
    padding = kernel_size // 2
    blurred = F.conv2d(img_tensor, kernel, padding=padding, groups=channels)
    
    return blurred


def _get_gaussian_kernel2d(kernel_size: int, sigma: float) -> torch.Tensor:
    """Generate 2D Gaussian kernel"""
    kernel_1d = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2
    kernel_1d = torch.exp(-kernel_1d.pow(2) / (2 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    return kernel_2d.reshape(1, 1, kernel_size, kernel_size)


@torch.no_grad()
def create_composite_gpu(image_path: str, target_w: int, target_h: int) -> torch.Tensor:
    """
    🚀 GPU-accelerated composite - returns TENSOR at EXACT target size
    - If aspect ratio matches: just resize to target
    - If aspect ratio differs: blur background + overlay foreground
    """
    img = Image.open(image_path).convert("RGB")
    iw, ih = img.size
    
    img_tensor = transforms.ToTensor()(img).unsqueeze(0).cuda()
    
    # Check if aspect ratios match (within 1% tolerance)
    needs_bg = abs((iw / ih) - (target_w / target_h)) > 0.01
    
    if not needs_bg:
        # Perfect aspect ratio - just resize
        logger.info(f"✅ Aspect ratio matches - simple resize to {target_w}x{target_h}")
        return F.interpolate(img_tensor, size=(target_h, target_w), 
                           mode='bilinear', align_corners=False)
    
    logger.info(f"⚠️ Aspect ratio mismatch - creating blurred composite")
    
    # Create blurred background
    blurred = F.interpolate(img_tensor, size=(target_h, target_w), 
                           mode='bilinear', align_corners=False)
    blurred = gaussian_blur_gpu(blurred, kernel_size=41, sigma=13.0)
    
    # Create foreground - fit to height while maintaining aspect ratio
    scale = target_h / ih
    new_w = int(iw * scale)
    
    if new_w <= target_w:
        # Image is narrower - center it on blurred background (pillarbox)
        fore = F.interpolate(img_tensor, size=(target_h, new_w), 
                            mode='bilinear', align_corners=False)
        
        x = (target_w - new_w) // 2
        composite = blurred.clone()
        composite[:, :, :, x:x+new_w] = fore
        logger.info(f"  → Pillarbox: foreground {new_w}x{target_h} centered in {target_w}x{target_h}")
    else:
        # Image is wider - fit to width instead (letterbox)
        scale = target_w / iw
        new_h = int(ih * scale)
        
        fore = F.interpolate(img_tensor, size=(new_h, target_w), 
                            mode='bilinear', align_corners=False)
        
        y = (target_h - new_h) // 2
        composite = blurred.clone()
        composite[:, :, y:y+new_h, :] = fore
        logger.info(f"  → Letterbox: foreground {target_w}x{new_h} centered in {target_w}x{target_h}")
    
    return composite


# ============================================================================
# GOD MODE: DIRECT GPU → FFMPEG PIPELINE
# ============================================================================

class GPUToFFmpegPipeline:
    """Zero-copy GPU → FFmpeg pipeline"""
    
    def __init__(self, 
                 width: int, 
                 height: int, 
                 fps: int = 30,
                 output_path: str = None,
                 preset: str = "p3",
                 crf: int = 23):
        
        self.width = width
        self.height = height
        self.fps = fps
        self.output_path = output_path or f"gpu_render_{random.randint(1000,9999)}.mp4"
        
        self.ffmpeg_cmd = [
            'ffmpeg',
            '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}',
            '-pix_fmt', 'rgb24',
            '-r', str(fps),
            '-i', '-',
            '-c:v', 'h264_nvenc',
            '-preset', preset,
            '-rc', 'vbr',
            '-cq', str(crf),
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            self.output_path
        ]
        
        self.process = None
        self.stderr_queue = queue.Queue()
        self.stderr_thread = None
        self.frame_count = 0
        
    def _stderr_reader(self):
        for line in iter(self.process.stderr.readline, b''):
            self.stderr_queue.put(line.decode('utf-8', errors='ignore'))
        
    def start(self):
        self.process = subprocess.Popen(
            self.ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            bufsize=10**8
        )
        
        self.stderr_thread = threading.Thread(target=self._stderr_reader, daemon=True)
        self.stderr_thread.start()
        
        logger.info(f"✅ FFmpeg NVENC pipeline started → {self.output_path}")
        
    def write_frame_gpu(self, frame_tensor: torch.Tensor):
        """Write frame: [1, 3, H, W] GPU tensor → FFmpeg"""
        try:
            # Verify dimensions
            _, _, h, w = frame_tensor.shape
            if h != self.height or w != self.width:
                logger.warning(f"⚠️ Resizing frame from {h}x{w} to {self.height}x{self.width}")
                frame_tensor = F.interpolate(frame_tensor, size=(self.height, self.width), 
                                            mode='bilinear', align_corners=False)
            
            # Convert to numpy RGB uint8
            frame_np = frame_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            frame_np = np.clip(frame_np * 255.0, 0, 255).astype(np.uint8)
            
            if not frame_np.flags['C_CONTIGUOUS']:
                frame_np = np.ascontiguousarray(frame_np)
            
            self.process.stdin.write(frame_np.tobytes())
            self.frame_count += 1
            
        except BrokenPipeError:
            logger.error("❌ FFmpeg pipe broken")
            self._print_stderr()
            raise
        
    def _print_stderr(self):
        while not self.stderr_queue.empty():
            try:
                line = self.stderr_queue.get_nowait()
                if 'error' in line.lower() or 'invalid' in line.lower():
                    logger.error(line.strip())
            except queue.Empty:
                break
        
    def finish(self):
        if self.process:
            try:
                logger.info(f"🎬 Finalizing ({self.frame_count} frames)...")
                self.process.stdin.close()
                returncode = self.process.wait(timeout=60)
                self._print_stderr()
                
                if returncode == 0:
                    logger.info(f"✅ Video rendered: {self.output_path}")
                else:
                    logger.error(f"❌ FFmpeg error code {returncode}")
                    
            except subprocess.TimeoutExpired:
                logger.error("❌ Timeout - killing FFmpeg")
                self.process.kill()
                self.process.wait()
                self._print_stderr()
                
        return self.output_path


# ============================================================================
# GPU EFFECT PROCESSOR
# ============================================================================

class GPUEffectProcessor:
    """All effects computed entirely on GPU - starts at NORMAL size"""
    
    def __init__(self, base_tensor: torch.Tensor, duration: float):
        """
        base_tensor: [1, 3, H, W] - the NORMAL sized composite image
        """
        self.base_tensor = base_tensor.cuda() if not base_tensor.is_cuda else base_tensor
        self.duration = duration
        self.h, self.w = base_tensor.shape[2:]
        
        logger.info(f"📐 Effect processor: {self.h}x{self.w}")
        
    @torch.no_grad()
    def zoom_in(self, t: float, zoom_ratio: float = 0.04) -> torch.Tensor:
        """
        Zoom IN effect: starts NORMAL, gradually zooms in
        Returns [1, 3, H, W] at original dimensions
        """
        # Calculate zoom level (starts at 1.0, increases)
        zoom = 1.0 + (zoom_ratio * t)
        
        # Calculate enlarged dimensions
        zoomed_h = int(self.h * zoom)
        zoomed_w = int(self.w * zoom)
        zoomed_h += zoomed_h % 2
        zoomed_w += zoomed_w % 2
        
        # Enlarge the image
        enlarged = F.interpolate(self.base_tensor, size=(zoomed_h, zoomed_w), 
                                mode='bilinear', align_corners=False)
        
        # Center crop back to original size
        y = (zoomed_h - self.h) // 2
        x = (zoomed_w - self.w) // 2
        
        return enlarged[:, :, y:y+self.h, x:x+self.w]
    
    @torch.no_grad()
    def zoom_out(self, t: float, zoom_start: float = 1.3, zoom_end: float = 1.0) -> torch.Tensor:
        """
        Zoom OUT effect: starts ZOOMED IN (1.3x), gradually zooms out to normal (1.0x)
        Returns [1, 3, H, W] at original dimensions
        """
        # Calculate zoom level (starts at zoom_start, decreases to zoom_end)
        progress = t / self.duration
        zoom = zoom_start - (progress * (zoom_start - zoom_end))
        zoom = max(zoom, zoom_end)  # Don't go below 1.0
        
        # Calculate enlarged dimensions
        zoomed_h = int(self.h * zoom)
        zoomed_w = int(self.w * zoom)
        zoomed_h += zoomed_h % 2
        zoomed_w += zoomed_w % 2
        
        # Enlarge the image
        enlarged = F.interpolate(self.base_tensor, size=(zoomed_h, zoomed_w), 
                                mode='bilinear', align_corners=False)
        
        # Center crop back to original size
        y = (zoomed_h - self.h) // 2
        x = (zoomed_w - self.w) // 2
        
        return enlarged[:, :, y:y+self.h, x:x+self.w]
    
    @torch.no_grad()
    def pan_left_to_right(self, t: float, pan_scale: float = 1.2) -> torch.Tensor:
        """
        Pan LEFT to RIGHT: image is 1.2x wide, camera pans from left to right
        Returns [1, 3, H, W] at original dimensions
        """
        # Create wider canvas
        wide_w = int(self.w * pan_scale)
        wide_canvas = F.interpolate(self.base_tensor, size=(self.h, wide_w), 
                                    mode='bilinear', align_corners=False)
        
        # Calculate pan position (0 to max_offset)
        max_offset = wide_w - self.w
        progress = t / self.duration
        x_offset = int(max_offset * progress)
        
        # Crop moving window
        return wide_canvas[:, :, :, x_offset:x_offset+self.w]
    
    @torch.no_grad()
    def pan_right_to_left(self, t: float, pan_scale: float = 1.2) -> torch.Tensor:
        """
        Pan RIGHT to LEFT: image is 1.2x wide, camera pans from right to left
        Returns [1, 3, H, W] at original dimensions
        """
        # Create wider canvas
        wide_w = int(self.w * pan_scale)
        wide_canvas = F.interpolate(self.base_tensor, size=(self.h, wide_w), 
                                    mode='bilinear', align_corners=False)
        
        # Calculate pan position (max_offset to 0)
        max_offset = wide_w - self.w
        progress = t / self.duration
        x_offset = int(max_offset * (1.0 - progress))
        
        # Crop moving window
        return wide_canvas[:, :, :, x_offset:x_offset+self.w]


# ============================================================================
# MAIN FUNCTION: TRUE GOD MODE
# ============================================================================

def create_video_gpu_direct(
    image_path: str,
    duration: float,
    effect_type: Literal['zoom_in', 'zoom_out', 'pan_left', 'pan_right', 'random'] = 'random',
    width: int = 1920,
    height: int = 1080,
    fps: int = 30,
    output_path: str = None,
) -> str:
    """
    🔥 TRUE GOD MODE: 100% GPU, smooth effects starting from NORMAL size
    
    Effect behaviors:
    - zoom_in: Starts normal, zooms in gradually
    - zoom_out: Starts zoomed in (1.3x), zooms out to normal
    - pan_left/right: Starts normal, pans across 1.2x canvas
    
    Returns: path to rendered video file
    """
    
    logger.info("🚀 Starting FULL GPU rendering pipeline...")
    
    # 1. Create composite at EXACT target size (normal size, not pre-zoomed)
    base_tensor = create_composite_gpu(image_path, width, height)
    logger.info(f"✅ Base composite: {base_tensor.shape}")
    
    # 2. Initialize processor (base tensor is NORMAL size)
    processor = GPUEffectProcessor(base_tensor, duration)
    pipeline = GPUToFFmpegPipeline(width, height, fps, output_path)
    pipeline.start()
    
    # 3. Choose effect
    if effect_type == 'random':
        effect_type = random.choice(['zoom_in', 'zoom_out', 'pan_left', 'pan_right'])
    
    effect_map = {
        'zoom_in': processor.zoom_in,
        'zoom_out': processor.zoom_out,
        'pan_left': processor.pan_left_to_right,
        'pan_right': processor.pan_right_to_left,
    }
    
    effect_fn = effect_map[effect_type]
    logger.info(f"🎬 Rendering with effect: {effect_type}")
    
    # 4. Render frames
    total_frames = int(duration * fps)
    
    for i in range(total_frames):
        t = i / fps
        frame_tensor = effect_fn(t)
        pipeline.write_frame_gpu(frame_tensor)
        
        if i % 30 == 0 or i == total_frames - 1:
            logger.info(f"⏳ Frame {i+1}/{total_frames} ({(i+1)/total_frames*100:.1f}%)")
    
    # 5. Finalize
    output = pipeline.finish()
    logger.info(f"✅ COMPLETE! {output}")
    
    return output


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
    
    # Test all effects
    effects = ['zoom_in', 'zoom_out', 'pan_left', 'pan_right']
    
    for effect in effects:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing effect: {effect}")
        logger.info(f"{'='*60}\n")
        
        video_path = create_video_gpu_direct(
            image_path="bg_strip.png",
            duration=5.0,
            effect_type=effect,
            width=1920,
            height=1080,
            fps=30,
            output_path=f"test_{effect}.mp4"
        )
        
        print(f"✅ {effect}: {video_path}\n")