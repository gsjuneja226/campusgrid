"""
CampusGrid: Distributed Video Processing Worker
Receives a base64-encoded video segment, applies an ffmpeg-based effect,
returns the processed segment as base64.

Effects:
  cinematic_grade  : LUT-style color grading, vignette, film grain
  film_noir        : B&W high contrast + dramatic shadows
  neon_glow        : Vivid oversaturated neon + glow
  stabilize_blur   : Smart blur + edge enhancement (fake stabilisation look)
  upscale_4k       : Bicubic upscale simulation + sharpening
"""
import os
import time
import base64
import subprocess
import tempfile


def _ffmpeg_path() -> str:
    """Returns the ffmpeg executable path."""
    return "ffmpeg"  # assumes ffmpeg in PATH (verified on master machine)


def _run_ffmpeg(args: list, timeout: int = 120) -> tuple:
    """Runs ffmpeg with given args. Returns (returncode, stderr)."""
    cmd = [_ffmpeg_path(), "-y"] + args
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout
    )
    return result.returncode, result.stderr


# ── Effect filter strings (ffmpeg -vf) ──────────────────────────────────────

EFFECT_FILTERS = {
    "cinematic_grade": (
        "eq=contrast=1.15:brightness=-0.02:saturation=0.85,"
        "curves=r='0/0 0.1/0.05 0.5/0.45 1/0.9':g='0/0 0.1/0.08 0.5/0.5 1/0.92':b='0/0.05 0.5/0.55 1/1',"
        "vignette=PI/5,"
        "noise=alls=3:allf=t+u"   # film grain
    ),
    "film_noir": (
        "hue=s=0,"                 # desaturate
        "eq=contrast=1.5:brightness=-0.05:gamma=0.85,"
        "curves=r='0/0 0.2/0.05 0.7/0.6 1/1',"
        "vignette=PI/4"
    ),
    "neon_glow": (
        "eq=saturation=2.5:contrast=1.2,"
        "split[main][glow];"
        "[glow]boxblur=10:10,eq=brightness=0.2[blurred];"
        "[main][blurred]blend=all_mode=screen"
    ),
    "stabilize_blur": (
        "unsharp=5:5:1.5:5:5:0.0,"
        "smartblur=lr=1.0:ls=-1.0:lt=-4.5:cr=0.8:cs=-2.0:ct=-4.5,"
        "eq=brightness=0.03:contrast=1.08"
    ),
    "upscale_4k": (
        "scale=iw*1.5:ih*1.5:flags=bicubic,"
        "unsharp=5:5:2.0:5:5:0.0,"
        "eq=contrast=1.1:saturation=1.1"
    ),
}


def run(chunk_index: int, total_chunks: int, progress_callback=None, extra_params: dict = None) -> str:
    """
    Processes a video segment with an ffmpeg filter.

    extra_params must contain:
        segment_b64 (str)  : base64-encoded input video segment bytes
        effect      (str)  : one of the keys in EFFECT_FILTERS
        segment_ext (str)  : file extension e.g. 'mp4'

    Returns:
        base64-encoded processed video segment (str)
    """
    extra_params = extra_params or {}
    segment_b64 = extra_params.get("segment_b64", "")
    effect = extra_params.get("effect", "cinematic_grade")
    seg_ext = extra_params.get("segment_ext", "mp4")

    if not segment_b64:
        raise ValueError("No video segment data received (segment_b64 is empty)")

    vf_filter = EFFECT_FILTERS.get(effect, EFFECT_FILTERS["cinematic_grade"])

    if progress_callback:
        progress_callback(5)

    # Write input segment to temp file
    with tempfile.TemporaryDirectory() as tmpdir:
        in_path  = os.path.join(tmpdir, f"in_seg_{chunk_index}.{seg_ext}")
        out_path = os.path.join(tmpdir, f"out_seg_{chunk_index}.{seg_ext}")

        in_bytes = base64.b64decode(segment_b64)
        with open(in_path, "wb") as f:
            f.write(in_bytes)

        if progress_callback:
            progress_callback(15)

        # Build ffmpeg command
        if ";" in vf_filter:
            # Complex filter with split/blend
            ffmpeg_args = [
                "-i", in_path,
                "-filter_complex", vf_filter,
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "22",
                "-c:a", "copy",
                out_path
            ]
        else:
            ffmpeg_args = [
                "-i", in_path,
                "-vf", vf_filter,
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "22",
                "-c:a", "copy",
                out_path
            ]

        if progress_callback:
            progress_callback(20)

        rc, stderr = _run_ffmpeg(ffmpeg_args, timeout=180)

        if progress_callback:
            progress_callback(85)

        if rc != 0:
            raise RuntimeError(f"ffmpeg processing failed (exit {rc}):\n{stderr[-800:]}")

        if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
            raise RuntimeError("ffmpeg produced empty output file")

        # Read and encode result
        with open(out_path, "rb") as f:
            out_bytes = f.read()

        if progress_callback:
            progress_callback(100)

        return base64.b64encode(out_bytes).decode("ascii")
