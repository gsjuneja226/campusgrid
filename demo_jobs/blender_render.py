"""
CampusGrid: Blender Render Farm Worker
Receives a .blend file, renders an assigned frame range using Blender's CLI,
and returns the rendered frames as base64-encoded PNG data.

Requires Blender to be installed on the worker machine.
Blender CLI: blender --background file.blend --render-output /path/ --frame-start N --frame-end M -a
"""
import os
import time
import base64
import subprocess
import tempfile
import glob
import shutil


def _find_blender() -> str:
    """
    Searches for Blender executable in common installation paths.
    Returns path if found, else raises RuntimeError.
    """
    import platform
    candidates = []

    if platform.system() == "Windows":
        # Search common Windows install locations
        import glob as g
        patterns = [
            r"C:\Program Files\Blender Foundation\Blender*\blender.exe",
            r"C:\Program Files (x86)\Blender Foundation\Blender*\blender.exe",
            r"D:\Program Files\Blender Foundation\Blender*\blender.exe",
        ]
        for pattern in patterns:
            candidates.extend(g.glob(pattern))
        candidates.extend(["blender", "blender.exe"])

    elif platform.system() == "Darwin":  # macOS
        candidates = [
            "/Applications/Blender.app/Contents/MacOS/Blender",
            "blender",
        ]
    else:  # Linux
        candidates = [
            "/usr/bin/blender",
            "/usr/local/bin/blender",
            "blender",
        ]

    for c in candidates:
        try:
            result = subprocess.run(
                [c, "--version"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                return c
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue

    raise RuntimeError(
        "Blender not found on this worker machine. "
        "Please install Blender from https://www.blender.org/download/ "
        "and make sure it's accessible."
    )


def run(chunk_index: int, total_chunks: int, progress_callback=None, extra_params: dict = None) -> list:
    """
    Renders a range of frames from a .blend file using Blender's headless CLI.

    extra_params must contain:
        blend_b64    (str)  : base64-encoded .blend file bytes
        frame_start  (int)  : first frame to render (1-indexed)
        frame_end    (int)  : last frame to render (inclusive)
        total_frames (int)  : total frames in animation (for UI display)
        render_engine (str) : 'CYCLES' or 'EEVEE' or 'WORKBENCH' (default: WORKBENCH for speed)
        samples      (int)  : render samples (default: 32 for speed demo)

    Returns:
        list of {frame_idx: int, png_b64: str} dicts
    """
    extra_params = extra_params or {}
    blend_b64 = extra_params.get("blend_b64", "")
    frame_start = int(extra_params.get("frame_start", 1))
    frame_end = int(extra_params.get("frame_end", 1))
    render_engine = extra_params.get("render_engine", "WORKBENCH")
    samples = int(extra_params.get("samples", 32))

    if not blend_b64:
        raise ValueError("No .blend file data received (blend_b64 is empty)")

    if progress_callback:
        progress_callback(3)

    # Find Blender
    blender_exe = _find_blender()

    with tempfile.TemporaryDirectory() as tmpdir:
        blend_path = os.path.join(tmpdir, "scene.blend")
        frames_dir = os.path.join(tmpdir, "frames")
        os.makedirs(frames_dir, exist_ok=True)

        # Write .blend file to disk
        blend_bytes = base64.b64decode(blend_b64)
        with open(blend_path, "wb") as f:
            f.write(blend_bytes)

        if progress_callback:
            progress_callback(8)

        # Build Blender Python script to configure engine + samples
        config_script = f"""
import bpy
scene = bpy.context.scene
scene.render.engine = '{render_engine}'
scene.render.image_settings.file_format = 'PNG'
scene.render.resolution_percentage = 100
"""
        if render_engine == "CYCLES":
            config_script += f"scene.cycles.samples = {samples}\n"
        elif render_engine == "WORKBENCH":
            config_script += "scene.display.shading.light = 'STUDIO'\n"

        config_script_path = os.path.join(tmpdir, "config.py")
        with open(config_script_path, "w") as f:
            f.write(config_script)

        # Output path pattern for frames: /tmpdir/frames/frame_####.png
        output_pattern = os.path.join(frames_dir, "frame_####")

        # Build Blender CLI command
        blender_cmd = [
            blender_exe,
            "--background",
            blend_path,
            "--python", config_script_path,
            "--render-output", output_pattern,
            "--render-format", "PNG",
            "--frame-start", str(frame_start),
            "--frame-end", str(frame_end),
            "--render-anim",
        ]

        total_frames_range = frame_end - frame_start + 1
        last_reported_pct = 8

        if progress_callback:
            progress_callback(10)

        # Run Blender and stream progress by monitoring output files
        proc = subprocess.Popen(
            blender_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=tmpdir
        )

        # Monitor progress by counting rendered frames in output dir
        start_time = time.time()
        timeout = 600  # 10 minutes max

        while proc.poll() is None:
            if time.time() - start_time > timeout:
                proc.kill()
                raise TimeoutError(f"Blender render timed out after {timeout}s")

            rendered_count = len(glob.glob(os.path.join(frames_dir, "*.png")))
            if total_frames_range > 0:
                pct = 10 + int((rendered_count / total_frames_range) * 85)
                if pct > last_reported_pct and progress_callback:
                    progress_callback(min(pct, 94))
                    last_reported_pct = pct
            time.sleep(0.5)

        rc = proc.returncode
        if rc != 0:
            stdout = proc.stdout.read() if proc.stdout else ""
            raise RuntimeError(f"Blender render failed (exit {rc}): {stdout[-1000:]}")

        if progress_callback:
            progress_callback(95)

        # Collect rendered PNG frames
        png_files = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
        if not png_files:
            raise RuntimeError("Blender produced no output frames. Check the .blend file and render settings.")

        results = []
        for idx, png_path in enumerate(png_files):
            # Extract frame index from filename (frame_0001.png → 1)
            basename = os.path.basename(png_path)
            try:
                frame_num = int(basename.replace("frame_", "").replace(".png", ""))
            except ValueError:
                frame_num = frame_start + idx

            with open(png_path, "rb") as f:
                png_b64 = base64.b64encode(f.read()).decode("ascii")
            results.append({"frame_idx": frame_num, "png_b64": png_b64})

        if progress_callback:
            progress_callback(100)

        return results
