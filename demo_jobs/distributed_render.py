"""
CampusGrid: Distributed Procedural Render Farm
Renders stunning procedural animation frames using Pillow + numpy.
Each worker renders a slice of the total frame range and returns base64 PNG data.

Supported scenes:
  - plasma_wave   : Animated plasma / interference wave patterns
  - mandelbrot    : Mandelbrot set zoom animation
  - aurora        : Northern lights style aurora borealis animation
"""
import time
import math
import base64
import io
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# ── Frame resolution ────────────────────────────────────────────────────────
RENDER_WIDTH  = 1280
RENDER_HEIGHT = 720


# ─────────────────────────────────────────────────────────────────────────────
# Scene Renderers
# ─────────────────────────────────────────────────────────────────────────────

def _render_plasma_wave(frame_idx: int, total_frames: int) -> Image.Image:
    """Animated plasma wave interference pattern — vibrant and hypnotic."""
    t = (frame_idx / total_frames) * 2 * math.pi
    W, H = RENDER_WIDTH, RENDER_HEIGHT

    xs = np.linspace(0, 4 * math.pi, W)
    ys = np.linspace(0, 4 * math.pi, H)
    x, y = np.meshgrid(xs, ys)

    # Composite wave field
    wave  = np.sin(x + t)
    wave += np.sin(y + t * 1.3)
    wave += np.sin((x + y + t) * 0.7)
    wave += np.sin(np.sqrt(x ** 2 + y ** 2 + 1) * 0.5 + t)
    wave += np.sin(x * 0.5 + t * 0.8) * 0.6
    wave += np.cos(y * 0.7 - t * 1.1) * 0.4

    norm = (wave - wave.min()) / (wave.max() - wave.min())  # 0..1

    # Map to vivid HSV-style color using sin/cos of phase-shifted value
    angle = norm * 2 * math.pi
    r = ((np.sin(angle + 0.0) + 1) * 127.5).astype(np.uint8)
    g = ((np.sin(angle + 2.094) + 1) * 127.5).astype(np.uint8)  # +120°
    b = ((np.sin(angle + 4.189) + 1) * 127.5).astype(np.uint8)  # +240°

    rgb = np.stack([r, g, b], axis=-1)
    img = Image.fromarray(rgb, mode="RGB")

    # Subtle bloom: slight blur then lighten
    bloom = img.filter(ImageFilter.GaussianBlur(radius=4))
    img = Image.blend(img, bloom, 0.25)
    return img


def _render_mandelbrot_zoom(frame_idx: int, total_frames: int) -> Image.Image:
    """Animated Mandelbrot set zoom towards a visually interesting point."""
    # Zoom target: mini-brot near (-0.7269, 0.1889)
    cx, cy = -0.7269, 0.1889
    zoom_start, zoom_end = 0.8, 0.004
    t = frame_idx / max(total_frames - 1, 1)
    zoom = zoom_start * ((zoom_end / zoom_start) ** t)

    W, H = RENDER_WIDTH, RENDER_HEIGHT
    max_iter = 80 + int(t * 120)  # increase detail as we zoom

    x0 = np.linspace(cx - zoom, cx + zoom, W)
    y0 = np.linspace(cy - zoom * H / W, cy + zoom * H / W, H)
    x, y = np.meshgrid(x0, y0)

    zr, zi = x.copy(), y.copy()
    iteration = np.zeros((H, W), dtype=np.float32)
    mask = np.ones((H, W), dtype=bool)

    for i in range(max_iter):
        if not mask.any():
            break
        zr2, zi2 = zr * zr, zi * zi
        escaped = mask & (zr2 + zi2 > 4)
        iteration[escaped] = i + 1 - np.log2(np.log2(zr2[escaped] + zi2[escaped] + 1e-6))
        mask &= ~escaped
        new_zr = zr2 - zi2 + x
        zi = 2 * zr * zi + y
        zr = new_zr

    # Normalize and colorize
    norm = iteration / max_iter
    # Vivid cycling palette
    hue_shift = t * math.pi
    r = ((np.sin(norm * 6 * math.pi + hue_shift) + 1) * 127.5).astype(np.uint8)
    g = ((np.sin(norm * 5 * math.pi + hue_shift + 1.0) + 1) * 127.5).astype(np.uint8)
    b = ((np.cos(norm * 7 * math.pi + hue_shift * 0.5) + 1) * 127.5).astype(np.uint8)
    # Interior = black
    interior = iteration == 0
    r[interior] = 0; g[interior] = 0; b[interior] = 10

    rgb = np.stack([r, g, b], axis=-1)
    return Image.fromarray(rgb, mode="RGB")


def _render_aurora(frame_idx: int, total_frames: int) -> Image.Image:
    """Northern lights / aurora borealis animation — deep space dark bg with ribbons."""
    t = (frame_idx / total_frames) * 2 * math.pi
    W, H = RENDER_WIDTH, RENDER_HEIGHT

    img_arr = np.zeros((H, W, 3), dtype=np.float32)

    # Star field (static per frame for speed, jitter on t)
    rng = np.random.default_rng(42)
    sx = (rng.random(300) * W).astype(int)
    sy = (rng.random(300) * H).astype(int)
    brightness = rng.random(300) * 0.8 + 0.2
    img_arr[sy, sx] += brightness[:, np.newaxis]

    xs = np.linspace(0, W, W)
    ys = np.arange(H)
    _, yg = np.meshgrid(xs, ys)

    # Aurora ribbons — multiple layers
    for layer in range(4):
        freq = 0.5 + layer * 0.4
        speed = 0.7 + layer * 0.3
        amp = 60 + layer * 20
        center_y = H * 0.35 + layer * 15

        xline = np.linspace(0, W, W)
        ribbon_y = center_y + amp * np.sin(freq * xline / W * 2 * math.pi + t * speed + layer)
        ribbon_y2 = ribbon_y + 80 + 30 * np.sin(xline / W * math.pi * 3 + t + layer * 1.5)

        # Gaussian falloff from ribbon center
        ribbon_mid = (ribbon_y + ribbon_y2) / 2
        half_width = (ribbon_y2 - ribbon_y) / 2 + 1
        dist = np.abs(yg - ribbon_mid[np.newaxis, :]) / (half_width[np.newaxis, :] + 1)
        intensity = np.exp(-dist ** 2 * 0.6) * (0.5 + 0.5 * math.sin(t + layer))

        # Color each layer differently
        colors = [
            (0.0, 0.8, 0.4),   # emerald green
            (0.1, 0.4, 0.9),   # electric blue
            (0.6, 0.1, 0.8),   # violet
            (0.0, 0.9, 0.7),   # cyan-green
        ]
        cr, cg, cb = colors[layer]
        img_arr[:, :, 0] += intensity * cr
        img_arr[:, :, 1] += intensity * cg
        img_arr[:, :, 2] += intensity * cb

    img_arr = np.clip(img_arr, 0, 1)
    img_arr = (img_arr * 255).astype(np.uint8)

    img = Image.fromarray(img_arr, mode="RGB")
    img = img.filter(ImageFilter.GaussianBlur(radius=2))

    # Add subtle vignette
    vignette = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(vignette)
    for i in range(min(W, H) // 2):
        alpha = int(255 * (i / (min(W, H) // 2)) ** 0.5)
        draw.ellipse([i, i, W - i, H - i], fill=alpha)
    vignette = vignette.filter(ImageFilter.GaussianBlur(radius=40))
    img = Image.composite(img, Image.new("RGB", (W, H), (0, 0, 10)), vignette)

    return img


# ─────────────────────────────────────────────────────────────────────────────
# Main Entry Point (called by worker.py)
# ─────────────────────────────────────────────────────────────────────────────

SCENE_RENDERERS = {
    "plasma_wave":       _render_plasma_wave,
    "mandelbrot":        _render_mandelbrot_zoom,
    "aurora":            _render_aurora,
}


def run(chunk_index: int, total_chunks: int, progress_callback=None, extra_params: dict = None) -> list:
    """
    Renders assigned frame range for this chunk.
    Returns a list of dicts: [{frame_idx: int, png_b64: str}, ...]
    """
    extra_params = extra_params or {}
    scene = extra_params.get("scene", "plasma_wave")
    total_frames = int(extra_params.get("total_frames", 120))

    renderer = SCENE_RENDERERS.get(scene, _render_plasma_wave)

    # Compute this chunk's frame range
    chunk_size = total_frames // total_chunks
    remainder  = total_frames % total_chunks
    start_frame = chunk_index * chunk_size + min(chunk_index, remainder)
    end_frame   = start_frame + chunk_size + (1 if chunk_index < remainder else 0) - 1

    frames_to_render = list(range(start_frame, end_frame + 1))
    results = []

    for idx, frame_idx in enumerate(frames_to_render):
        img = renderer(frame_idx, total_frames)

        # Encode as base64 PNG
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=False)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        results.append({"frame_idx": frame_idx, "png_b64": b64})

        if progress_callback:
            progress_callback(int(((idx + 1) / len(frames_to_render)) * 100))

    return results
