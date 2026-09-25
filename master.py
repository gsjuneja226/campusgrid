"""
CampusGrid Master Server
Coordinated by Flask + Flask-SocketIO. Handles job distribution, chunk scheduling,
fault tolerance, real-time socket broadcasts, and dashboard visualization.
"""
import os
import sys
import time
import json
import sqlite3
import random
import contextlib
import subprocess
import base64
import threading
import tempfile
import glob
import shutil
import struct
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory, Response
from flask_socketio import SocketIO, emit

# Add current dir to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import chunker
import reducer

app = Flask(__name__)
# Using simple threading async mode for maximum Windows compatibility without eventlet compilation requirements
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading", max_http_buffer_size=512 * 1024 * 1024)

DB_FILE = "campusgrid.db"
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
VIDEO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "video_output")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)

# In-memory tracking of online workers and active job coordination
# active_workers keys are session IDs (sid)
active_workers = {}
active_job = None
ping_loop_started = False

# ── AI Inference Pipeline State ───────────────────────────────────────────────
# Sequential pipeline: we pass hidden states through workers one at a time
ai_pipeline_state = {
    "running": False,
    "job_id": None,
    "prompt": "",
    "tokens_generated": [],
    "current_worker_idx": 0,
    "hidden_states_b64": None,
    "input_ids_b64": None,
    "layer_assignments": [],   # list of {worker_sid, layer_start, layer_end}
    "tokens_per_sec": 0.0,
    "start_time": 0,
    "max_tokens": 50,
    "lock": None,
}
ai_pipeline_state["lock"] = threading.Lock()

def log_event(message):
    """Prints to stdout and streams to SocketIO clients for the dashboard log terminal"""
    print(message)
    # Stream the log message to dashboard client connections
    socketio.emit("system_log", {"timestamp": time.time(), "message": str(message)})

def ping_workers_loop():
    """Background thread running every 5 seconds to measure connection latency to each node"""
    while True:
        socketio.sleep(5.0)
        # Send a ping request to every active worker node
        for sid in list(active_workers.keys()):
            socketio.emit("ping_node", {"sent_at": time.time()}, to=sid)

@socketio.on("pong_node")
def handle_pong_node(data):
    sid = request.sid
    sent_at = data.get("sent_at")
    if sid in active_workers and sent_at:
        # Calculate round-trip time latency in milliseconds
        rtt = (time.time() - float(sent_at)) * 1000
        active_workers[sid]["latency"] = round(rtt, 1)
        emit_workers_update()

@contextlib.contextmanager
def get_db():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        with conn:
            yield conn
    finally:
        conn.close()

def init_db():
    with get_db() as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                job_type TEXT NOT NULL,
                status TEXT NOT NULL,
                params TEXT,
                worker_count INTEGER DEFAULT 0,
                start_time REAL,
                end_time REAL,
                parallel_time REAL,
                single_time_est REAL,
                speedup REAL,
                result_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        db.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                chunk_data TEXT,
                status TEXT NOT NULL,
                worker_id TEXT,
                progress INTEGER DEFAULT 0,
                time_taken REAL DEFAULT 0.0,
                error_msg TEXT,
                FOREIGN KEY (job_id) REFERENCES jobs (id) ON DELETE CASCADE
            )
        """)
        db.commit()

init_db()

# Large text source for the Word Count demo job
LARGE_TEXT_SOURCE = [
    "The CampusGrid project is a distributed computing platform designed to harness the power of idle laboratory computers.",
    "By dividing complex tasks into smaller chunks, we can execute computations simultaneously and merge the results.",
    "This parallel execution model is inspired by MapReduce, which divides work into Map and Reduce phases.",
    "In the Map phase, the master node partitions the workload and sends individual chunks to various worker nodes.",
    "In the Execute phase, workers process their assigned chunks simultaneously and stream real-time progress updates.",
    "Every 500 milliseconds, workers report their progress, system resource usage, and active tasks back to the master.",
    "If a worker suddenly disconnects, the master immediately detects the failure and reassigns the chunk to another worker.",
    "This fault tolerance mechanism guarantees that long-running jobs are executed reliably, even on unstable networks.",
    "In the Reduce phase, the master collects all chunk responses, aggregates them, and writes the final output to disk.",
    "The SQLite database is used to track job execution histories, worker registration specs, and individual chunk states.",
    "Performance statistics, such as speedup ratio and estimated single-machine time, are displayed dynamically in real time.",
    "A clean dark mode dashboard gives laboratory administrators a birds-eye view of CPU, memory, and storage resources.",
    "Numpy is used in demo jobs to demonstrate heavy scientific computations, such as parallel matrix multiplications.",
    "Distributed computing is crucial for modern scientific workflows, including Monte Carlo simulations and cryptography.",
    "Using socket connections, we establish low-latency bidirectional pipelines that scale to dozens of computer lab nodes."
] * 1000  # Multiplied to create 15,000 paragraphs (~2.5 MB of textual data)

# ----------------------------------------------------
# DB Helper Functions
# ----------------------------------------------------

def db_create_job(name, job_type, params_dict, status="RUNNING", start_time=None):
    with get_db() as db:
        cursor = db.cursor()
        cursor.execute(
            "INSERT INTO jobs (name, job_type, status, params, start_time) VALUES (?, ?, ?, ?, ?)",
            (name, job_type, status, json.dumps(params_dict), start_time)
        )
        job_id = cursor.lastrowid
        db.commit()
        return job_id

def db_create_chunk(job_id, chunk_index, chunk_data):
    with get_db() as db:
        db.execute(
            "INSERT INTO chunks (job_id, chunk_index, chunk_data, status) VALUES (?, ?, ?, ?)",
            (job_id, chunk_index, json.dumps(chunk_data), "PENDING")
        )
        db.commit()

def db_update_chunk(job_id, chunk_index, status, worker_id=None, progress=0, time_taken=0.0, error_msg=None):
    with get_db() as db:
        if worker_id is not None:
            db.execute(
                "UPDATE chunks SET status = ?, worker_id = ?, progress = ?, time_taken = ?, error_msg = ? WHERE job_id = ? AND chunk_index = ?",
                (status, worker_id, progress, time_taken, error_msg, job_id, chunk_index)
            )
        else:
            db.execute(
                "UPDATE chunks SET status = ?, progress = ?, time_taken = ?, error_msg = ? WHERE job_id = ? AND chunk_index = ?",
                (status, progress, time_taken, error_msg, job_id, chunk_index)
            )
        db.commit()

def db_complete_job(job_id, worker_count, parallel_time, single_time_est, speedup, result_path):
    with get_db() as db:
        db.execute(
            "UPDATE jobs SET status = 'COMPLETED', worker_count = ?, end_time = ?, parallel_time = ?, single_time_est = ?, speedup = ?, result_path = ? WHERE id = ?",
            (worker_count, time.time(), parallel_time, single_time_est, speedup, result_path, job_id)
        )
        db.commit()

def db_fail_job(job_id):
    with get_db() as db:
        db.execute(
            "UPDATE jobs SET status = 'FAILED', end_time = ? WHERE id = ?",
            (time.time(), job_id)
        )
        db.commit()

# ----------------------------------------------------
# Real-Time Event Emitters
# ----------------------------------------------------

def emit_workers_update():
    """Broadcasts current online worker nodes list to the UI dashboard"""
    workers_list = []
    for sid, info in active_workers.items():
        workers_list.append({
            "worker_id": info["worker_id"],
            "ip": info["ip"],
            "specs": info["specs"],
            "status": info["status"],
            "stats": info.get("stats", {}),
            "latency": info.get("latency", 0.0)
        })
    socketio.emit("workers_list", workers_list)

def emit_job_update():
    """Broadcasts current job progress and chunk list to the dashboard"""
    global active_job
    
    # Always fetch the list of queued jobs from database
    with get_db() as db:
        queued_rows = db.execute(
            "SELECT id, name, job_type FROM jobs WHERE status = 'QUEUED' ORDER BY id ASC"
        ).fetchall()
    queued_jobs = []
    for r in queued_rows:
        queued_jobs.append({
            "job_id": r["id"],
            "name": r["name"],
            "job_type": r["job_type"]
        })
        
    if not active_job:
        socketio.emit("job_status", {
            "active": False,
            "queued_jobs": queued_jobs
        })
        return

    # Calculate overall progress
    total_chunks = len(active_job["chunks"])
    completed_chunks = sum(1 for c in active_job["chunks"] if c["status"] == "COMPLETED")
    
    # Calculate sum of progresses
    total_progress_pct = sum(c["progress"] for c in active_job["chunks"]) // total_chunks if total_chunks > 0 else 0
    
    elapsed = time.time() - active_job["start_time"]
    
    # Simple remaining time estimate
    est_remaining = 0
    if completed_chunks > 0:
        avg_chunk_time = sum(c["time_taken"] for c in active_job["chunks"] if c["status"] == "COMPLETED") / completed_chunks
        # Active concurrent worker threads speed
        num_workers = max(1, len(active_workers))
        remaining_chunks = total_chunks - completed_chunks
        est_remaining = (remaining_chunks * avg_chunk_time) / num_workers
    
    # Compute active worker speedup estimation on the fly
    sum_worker_times = sum(c["time_taken"] for c in active_job["chunks"])
    speedup = sum_worker_times / elapsed if elapsed > 0 else 1.0

    socketio.emit("job_status", {
        "active": True,
        "job_id": active_job["job_id"],
        "name": active_job["name"],
        "job_type": active_job["job_type"],
        "status": active_job["status"],
        "total_chunks": total_chunks,
        "completed_chunks": completed_chunks,
        "progress": total_progress_pct,
        "elapsed_time": round(elapsed, 1),
        "est_remaining": round(est_remaining, 1),
        "speedup_est": round(speedup, 1),
        "chunks": active_job["chunks"],
        "queued_jobs": queued_jobs
    })

def trigger_next_job():
    """Fetches the next queued job from SQLite and initiates execution"""
    global active_job
    if active_job:
        return
        
    with get_db() as db:
        next_job = db.execute(
            "SELECT * FROM jobs WHERE status = 'QUEUED' ORDER BY id ASC LIMIT 1"
        ).fetchone()
        
    if not next_job:
        # Broadcast grid idle status
        emit_job_update()
        return
        
    job_id = next_job["id"]
    job_type = next_job["job_type"]
    job_name = next_job["name"]
    params_dict = json.loads(next_job["params"]) if next_job["params"] else {}
    
    # Update job status to RUNNING in SQLite
    with get_db() as db:
        db.execute(
            "UPDATE jobs SET status = 'RUNNING', start_time = ? WHERE id = ?",
            (time.time(), job_id)
        )
        db.commit()
        
    # Read chunks from SQLite
    with get_db() as db:
        chunk_rows = db.execute(
            "SELECT chunk_index, chunk_data FROM chunks WHERE job_id = ? ORDER BY chunk_index ASC",
            (job_id,)
        ).fetchall()
        
    chunks = []
    for row in chunk_rows:
        chunks.append({
            "chunk_index": row["chunk_index"],
            "chunk_data": json.loads(row["chunk_data"]),
            "status": "PENDING",
            "worker_id": None,
            "progress": 0,
            "time_taken": 0.0,
            "retries": 0
        })
        
    extra_params = params_dict.get("extra_params", {})
    
    active_job = {
        "job_id": job_id,
        "name": job_name,
        "job_type": job_type,
        "status": "RUNNING",
        "start_time": time.time(),
        "worker_count": len(active_workers),
        "extra_params": extra_params,
        "chunks": chunks,
        "results": {}
    }
    
    log_event(f"[*] Started queued job {job_id}: {job_name}")
    emit_job_update()
    assign_pending_chunks()

# ----------------------------------------------------
# Scheduling & Execution Logic
# ----------------------------------------------------

def assign_pending_chunks():
    """Finds idle workers and assigns pending chunks to them"""
    global active_job, active_workers
    if not active_job or active_job["status"] != "RUNNING":
        return
        
    # Get session IDs of all idle workers
    idle_sids = [sid for sid, w in active_workers.items() if w["status"] == "idle"]
    if not idle_sids:
        return
        
    for sid in idle_sids:
        # Find a PENDING chunk
        pending_chunk = None
        for chunk in active_job["chunks"]:
            if chunk["status"] == "PENDING":
                pending_chunk = chunk
                break
                
        if not pending_chunk:
            break  # No more pending chunks
            
        worker = active_workers[sid]
        worker["status"] = "working"
        
        pending_chunk["status"] = "RUNNING"
        pending_chunk["worker_id"] = worker["worker_id"]
        pending_chunk["progress"] = 0
        
        # Update SQLite Chunk state
        db_update_chunk(
            active_job["job_id"], 
            pending_chunk["chunk_index"], 
            "RUNNING", 
            worker_id=worker["worker_id"]
        )
        
        # Notify dashboard of updates
        emit_workers_update()
        emit_job_update()
        
        # Emit work assignment to the worker node
        log_event(f"[*] Assigned Chunk {pending_chunk['chunk_index'] + 1}/{len(active_job['chunks'])} to worker {worker['worker_id']}")
        socketio.emit("assign_chunk", {
            "job_id": active_job["job_id"],
            "chunk_index": pending_chunk["chunk_index"],
            "total_chunks": len(active_job["chunks"]),
            "job_type": active_job["job_type"],
            "chunk_data": pending_chunk["chunk_data"],
            "extra_params": active_job.get("extra_params", {})
        }, to=sid)

def finish_job():
    """Aggregates chunk results, calculates performance metrics, and completes the active job"""
    global active_job
    if not active_job or active_job["status"] != "RUNNING":
        return
        
    active_job["status"] = "COMPLETED"
    job_id = active_job["job_id"]
    
    parallel_time = time.time() - active_job["start_time"]
    
    # Extract results ordered by chunk index
    sorted_chunks = sorted(active_job["chunks"], key=lambda c: c["chunk_index"])
    results_ordered = [active_job["results"][c["chunk_index"]] for c in sorted_chunks]
    single_time_est = sum(c["time_taken"] for c in sorted_chunks)
    
    # Calculate speedup
    speedup = single_time_est / parallel_time if parallel_time > 0 else 1.0
    # Safeguard against extreme values (e.g. sub-millisecond computations)
    speedup = min(speedup, len(active_workers) * 2) if len(active_workers) > 0 else 1.0
    if speedup < 0.1:
        speedup = 1.0
        
    # Reduce results
    job_type = active_job["job_type"]
    final_output = None
    summary_text = ""
    
    if job_type == "prime_finder":
        final_output = reducer.concat_reduce(results_ordered)
        # Sort primes
        final_output = sorted(final_output)
        total_primes = len(final_output)
        first_10 = ", ".join(map(str, final_output[:10]))
        last_10 = ", ".join(map(str, final_output[-10:]))
        summary_text = f"Found {total_primes} prime numbers in total.\nFirst 10 primes: {first_10}\nLast 10 primes: {last_10}"

    elif job_type == "distributed_render":
        # Flatten all frame dicts from all chunks
        all_frames = []
        for chunk_result in results_ordered:
            if isinstance(chunk_result, list):
                all_frames.extend(chunk_result)
            elif isinstance(chunk_result, dict):
                all_frames.append(chunk_result)
        all_frames.sort(key=lambda x: x["frame_idx"])
        total_frames = len(all_frames)
        summary_text = f"Rendered {total_frames} frames across {active_job['worker_count']} workers."
        final_output = f"{total_frames} frames rendered"

        # Stitch frames into MP4 using ffmpeg
        try:
            stitch_path = _stitch_frames_to_video(
                all_frames, active_job["job_id"],
                active_job.get("extra_params", {}).get("scene", "plasma_wave")
            )
            if stitch_path:
                summary_text += f"\nVideo saved: {os.path.basename(stitch_path)}"
                # Broadcast video ready event
                socketio.emit("render_video_ready", {
                    "job_id": active_job["job_id"],
                    "filename": os.path.basename(stitch_path),
                    "total_frames": total_frames
                })
        except Exception as ve:
            summary_text += f"\nVideo stitch error: {ve}"
            log_event(f"[!] Frame stitch error: {ve}")

    elif job_type in ["blender_render"]:
        # Same as distributed_render — collect frames and stitch
        all_frames = []
        for chunk_result in results_ordered:
            if isinstance(chunk_result, list):
                all_frames.extend(chunk_result)
            elif isinstance(chunk_result, dict):
                all_frames.append(chunk_result)
        all_frames.sort(key=lambda x: x["frame_idx"])
        total_frames = len(all_frames)
        summary_text = f"Blender rendered {total_frames} frames across {active_job['worker_count']} workers."
        final_output = f"{total_frames} Blender frames rendered"
        try:
            stitch_path = _stitch_frames_to_video(
                all_frames, active_job["job_id"], "blender_render"
            )
            if stitch_path:
                summary_text += f"\nVideo: {os.path.basename(stitch_path)}"
                socketio.emit("render_video_ready", {
                    "job_id": active_job["job_id"],
                    "filename": os.path.basename(stitch_path),
                    "total_frames": total_frames
                })
        except Exception as ve:
            summary_text += f"\nBlender stitch error: {ve}"

    elif job_type == "video_processing":
        # Collect processed segments in order and concatenate
        try:
            out_path = _stitch_video_segments(
                results_ordered, active_job["job_id"],
                active_job.get("extra_params", {}).get("segment_ext", "mp4")
            )
            summary_text = f"Processed {len(results_ordered)} video segments across {active_job['worker_count']} workers."
            if out_path:
                summary_text += f"\nFinal video: {os.path.basename(out_path)}"
                socketio.emit("video_processing_ready", {
                    "job_id": active_job["job_id"],
                    "filename": os.path.basename(out_path)
                })
            final_output = summary_text
        except Exception as ve:
            summary_text = f"Video segment stitch error: {ve}"
            final_output = summary_text
            log_event(f"[!] Video segment stitch error: {ve}")
        
    elif job_type == "monte_carlo_pi":
        final_output = reducer.average_reduce(results_ordered)
        diff_from_actual = abs(final_output - np.pi)
        summary_text = (
            f"Estimated Pi: {final_output:.6f}\n"
            f"Actual Pi:    {np.pi:.6f}\n"
            f"Accuracy Error: {diff_from_actual:.6f}\n\n"
            f"Individual worker Pi estimates: "
            + ", ".join([f"{r:.4f}" for r in results_ordered])
        )
        
    elif job_type == "matrix_multiply":
        # Stacks vertically
        final_output = reducer.matrix_stack_reduce(results_ordered)
        rows = len(final_output)
        cols = len(final_output[0]) if rows > 0 else 0
        first_row_str = str(final_output[0][:5]) + "..." if rows > 0 else "[]"
        summary_text = (
            f"Completed multiplication of Matrix A and Matrix B.\n"
            f"Final stacked matrix dimensions: {rows} x {cols}\n"
            f"First row sample (first 5 elements): {first_row_str}"
        )
        
    elif job_type == "word_count":
        final_output = reducer.dict_merge_reduce(results_ordered)
        top_20 = list(final_output.items())[:20]
        top_words_str = "\n".join([f"  '{word}': {count}" for word, count in top_20])
        summary_text = (
            f"Completed word parsing on large paragraphs.\n"
            f"Total unique words found: {len(final_output)}\n\n"
            f"Top 20 most frequent words:\n{top_words_str}"
        )
        
    elif job_type in ["custom_script", "render_frames", "blur_images", "process_portrait", "federated_learning", "grid_search_rf", "process_logs", "llm_inference"]:
        if len(results_ordered) == 1:
            final_output = results_ordered[0] if results_ordered else "No execution result returned"
            summary_text = final_output
        else:
            final_output = results_ordered
            mode = active_job.get("extra_params", {}).get("mode", "ensemble")
            model = active_job.get("extra_params", {}).get("model", "?")
            
            if job_type == "llm_inference":
                mode_labels = {
                    "batch_qa":  "Batch Q&A",
                    "document":  "Document Analysis",
                    "ensemble":  "Ensemble Reasoning",
                }
                summary_text = (
                    f"LLM Inference complete — {mode_labels.get(mode, mode)} mode\n"
                    f"Model: {model} · {len(results_ordered)} worker(s)\n"
                    f"{'=' * 60}\n\n"
                )
                if mode == "ensemble":
                    summary_text += "=== ENSEMBLE REASONING RESULTS ===\n"
                    summary_text += "(Each worker took an independent reasoning path — compare their answers below)\n\n"
                for idx, res in enumerate(results_ordered):
                    summary_text += f"{res}\n"
            else:
                summary_text = f"Compiled results from {len(results_ordered)} parallel workers:\n\n"
                for idx, res in enumerate(results_ordered):
                    summary_text += f"--- Worker Chunk {idx+1} Output ---\n{res}\n\n"
        
    # Write final result to folder
    filename = f"job_{job_id}_result.txt"
    file_path = os.path.join(RESULTS_DIR, filename)
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"CAMPUSGRID JOB REPORT: {active_job['name'].upper()}\n")
        f.write("="*60 + "\n")
        f.write(f"Job ID:             {job_id}\n")
        f.write(f"Job Type:           {job_type}\n")
        f.write(f"Workers Deployed:   {active_job['worker_count']}\n")
        f.write(f"Parallel Execution: {parallel_time:.3f} seconds\n")
        f.write(f"Single-Machine Est: {single_time_est:.3f} seconds\n")
        f.write(f"Calculated Speedup: {speedup:.2f}x\n")
        f.write("="*60 + "\n\n")
        f.write("EXECUTION RESULT SUMMARY:\n")
        f.write(summary_text)
        f.write("\n\n" + "="*60 + "\n")
        f.write("RAW RESULT:\n")
        if job_type in ["prime_finder", "matrix_multiply"]:
            f.write(f"(Total output size: {len(final_output)} items)\n")
            f.write(json.dumps(final_output, indent=2))
        else:
            f.write(json.dumps(final_output, indent=2))
            
    # Save statistics in SQLite database
    db_complete_job(
        job_id, 
        worker_count=active_job["worker_count"],
        parallel_time=parallel_time, 
        single_time_est=single_time_est, 
        speedup=speedup, 
        result_path=filename
    )
    
    # Broadcast final state to dashboard UI
    socketio.emit("job_complete", {
        "job_id": job_id,
        "name": active_job["name"],
        "job_type": job_type,
        "parallel_time": round(parallel_time, 2),
        "single_time_est": round(single_time_est, 2),
        "speedup": round(speedup, 2),
        "summary": summary_text,
        "filename": filename
    })
    log_event(f"[+] Job {job_id} ({job_type}) completed successfully in {parallel_time:.2f} seconds (Speedup: {speedup:.1f}x)")
    active_job = None
    
    # Ensure all connected active workers are reset to idle
    for sid, info in active_workers.items():
        if info["status"] == "working":
            info["status"] = "idle"
            
    emit_workers_update()
    emit_job_update()
    
    # Trigger execution of the next job in the queue
    trigger_next_job()

# ----------------------------------------------------
# Video / Frame Stitching Helpers
# ----------------------------------------------------

def _stitch_frames_to_video(frames: list, job_id: int, scene_name: str) -> str:
    """Takes a list of {frame_idx, png_b64} dicts, writes PNGs, stitches MP4 via ffmpeg."""
    import tempfile, glob as gglob
    with tempfile.TemporaryDirectory() as tmpdir:
        for item in frames:
            png_bytes = base64.b64decode(item["png_b64"])
            fname = os.path.join(tmpdir, f"frame_{item['frame_idx']:06d}.png")
            with open(fname, "wb") as f:
                f.write(png_bytes)

        out_filename = f"render_job_{job_id}_{scene_name}.mp4"
        out_path = os.path.join(VIDEO_DIR, out_filename)

        # Build frame list file for ffmpeg concat (use forward slashes for ffmpeg on Windows)
        png_files = sorted(gglob.glob(os.path.join(tmpdir, "frame_*.png")))
        list_path = os.path.join(tmpdir, "frames.txt")
        with open(list_path, "w") as f:
            for pf in png_files:
                pf_fwd = pf.replace("\\", "/")
                f.write(f"file '{pf_fwd}'\n")
                f.write("duration 0.0417\n")  # 24 fps

        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", list_path,
            "-vsync", "vfr",
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-pix_fmt", "yuv420p",
            out_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            log_event(f"[!] ffmpeg stitch error: {result.stderr[-500:]}")
            return None
        log_event(f"[+] Video stitched: {out_filename} ({len(frames)} frames)")
        return out_path


def _stitch_video_segments(segments_b64: list, job_id: int, ext: str = "mp4") -> str:
    """Concatenates base64-encoded processed video segments into final output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        seg_paths = []
        for idx, seg_b64 in enumerate(segments_b64):
            if not seg_b64:
                continue
            seg_path = os.path.join(tmpdir, f"seg_{idx:04d}.{ext}")
            with open(seg_path, "wb") as f:
                f.write(base64.b64decode(seg_b64))
            seg_paths.append(seg_path)

        out_filename = f"vidproc_job_{job_id}_output.mp4"
        out_path = os.path.join(VIDEO_DIR, out_filename)

        # Write concat list
        list_path = os.path.join(tmpdir, "concat.txt")
        with open(list_path, "w") as f:
            for sp in seg_paths:
                f.write(f"file '{sp}'\n")

        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", list_path,
            "-c", "copy",
            out_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            log_event(f"[!] ffmpeg segment concat error: {result.stderr[-500:]}")
            return None
        log_event(f"[+] Video segments stitched: {out_filename}")
        return out_path


def _split_video_into_segments(video_path: str, num_segments: int, tmpdir: str) -> list:
    """Splits a video file into N equal-duration segments using ffmpeg."""
    # Get duration
    probe_cmd = [
        "ffmpeg", "-i", video_path,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1"
    ]
    # Use ffprobe if available, else estimate from ffmpeg stderr
    probe_result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        capture_output=True, text=True, timeout=30
    )
    try:
        duration = float(probe_result.stdout.strip())
    except Exception:
        duration = 30.0  # fallback

    seg_duration = duration / num_segments
    segments = []
    ext = os.path.splitext(video_path)[1].lstrip(".")

    for i in range(num_segments):
        ss = i * seg_duration
        seg_path = os.path.join(tmpdir, f"input_seg_{i:04d}.{ext}")
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(ss),
            "-i", video_path,
            "-t", str(seg_duration),
            "-c", "copy",
            seg_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0 and os.path.exists(seg_path):
            with open(seg_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("ascii")
            segments.append({"seg_b64": b64, "ext": ext, "seg_idx": i})
        else:
            log_event(f"[!] Segment {i} split failed: {result.stderr[-300:]}")

    return segments


# ----------------------------------------------------
# REST API Endpoints
# ----------------------------------------------------

@app.route("/")
def index():
    return render_template("dashboard.html")

@app.route("/api/history")
def get_history():
    with get_db() as db:
        rows = db.execute("""
            SELECT id, name, job_type, status, params, worker_count, start_time, end_time, 
                   parallel_time, single_time_est, speedup, result_path, 
                   strftime('%Y-%m-%dT%H:%M:%SZ', created_at) AS created_at 
            FROM jobs ORDER BY id DESC
        """).fetchall()
        jobs_list = [dict(r) for r in rows]
    return jsonify(jobs_list)

@app.route("/api/results/<filename>")
def download_result(filename):
    # Security: ensure file resides inside results folder
    safe_filename = os.path.basename(filename)
    return send_from_directory(RESULTS_DIR, safe_filename, as_attachment=True)


@app.route("/api/render_output/<filename>")
def serve_render_output(filename):
    """Serves stitched render farm videos for in-browser playback."""
    safe_filename = os.path.basename(filename)
    return send_from_directory(VIDEO_DIR, safe_filename, mimetype="video/mp4")


@app.route("/api/upload_video", methods=["POST"])
def upload_video():
    """Receives an uploaded video, splits it into N segments, creates a video_processing job."""
    global active_job, active_workers

    num_workers = len(active_workers)
    if num_workers == 0:
        return jsonify({"success": False, "error": "No workers online to process video"}), 400

    video_file = request.files.get("video_file")
    if not video_file or video_file.filename == "":
        return jsonify({"success": False, "error": "No video file uploaded"}), 400

    effect = request.form.get("effect", "cinematic_grade")
    num_segments = int(request.form.get("segments") or num_workers)
    num_segments = max(1, min(num_segments, num_workers))

    original_filename = video_file.filename
    ext = os.path.splitext(original_filename)[1].lstrip(".") or "mp4"

    # Save uploaded video to temp file
    with tempfile.TemporaryDirectory() as tmpdir:
        in_path = os.path.join(tmpdir, f"input.{ext}")
        video_file.save(in_path)

        file_size_mb = os.path.getsize(in_path) / (1024 * 1024)
        log_event(f"[*] Uploaded video: {original_filename} ({file_size_mb:.1f} MB), splitting into {num_segments} segments...")

        # Split into segments
        try:
            segments = _split_video_into_segments(in_path, num_segments, tmpdir)
        except Exception as e:
            return jsonify({"success": False, "error": f"Video split failed: {e}"}), 500

    if not segments:
        return jsonify({"success": False, "error": "Video splitting produced no segments"}), 500

    job_name = f"Video Interceptor: {original_filename} ({effect})"
    extra_params = {"effect": effect, "segment_ext": ext, "original_filename": original_filename}
    params_dict = {"chunks": len(segments), "input_params": {"effect": effect}, "extra_params": extra_params}

    is_queued = active_job is not None
    job_status = "QUEUED" if is_queued else "RUNNING"
    job_start_time = None if is_queued else time.time()

    job_id = db_create_job(job_name, "video_processing", params_dict, status=job_status, start_time=job_start_time)
    log_event(f"[*] Created video_processing job {job_id}: {job_name}")

    # Create chunks with segment data embedded
    chunks_list = []
    for seg in segments:
        chunk_data = {
            "chunk_index": seg["seg_idx"],
            "total_chunks": len(segments),
            "extra_segment": {
                "segment_b64": seg["seg_b64"],
                "effect": effect,
                "segment_ext": ext
            }
        }
        chunks_list.append(chunk_data)

    for idx, c_data in enumerate(chunks_list):
        db_create_chunk(job_id, idx, c_data)

    if not is_queued:
        active_job = {
            "job_id": job_id,
            "name": job_name,
            "job_type": "video_processing",
            "status": "RUNNING",
            "start_time": job_start_time,
            "worker_count": num_workers,
            "extra_params": extra_params,
            "chunks": [],
            "results": {}
        }
        for idx, c_data in enumerate(chunks_list):
            active_job["chunks"].append({
                "chunk_index": idx,
                "chunk_data": c_data,
                "status": "PENDING",
                "worker_id": None,
                "progress": 0,
                "time_taken": 0.0,
                "retries": 0
            })
        emit_job_update()
        assign_pending_chunks()
    else:
        emit_job_update()

    return jsonify({"success": True, "job_id": job_id, "queued": is_queued, "segments": len(segments)})


@app.route("/api/upload_blend", methods=["POST"])
def upload_blend():
    """Receives an uploaded .blend file, distributes frame rendering across workers."""
    global active_job, active_workers

    num_workers = len(active_workers)
    if num_workers == 0:
        return jsonify({"success": False, "error": "No workers online"}), 400

    blend_file = request.files.get("blend_file")
    if not blend_file or blend_file.filename == "":
        return jsonify({"success": False, "error": "No .blend file uploaded"}), 400

    frame_start = int(request.form.get("frame_start", 1))
    frame_end = int(request.form.get("frame_end", 120))
    render_engine = request.form.get("render_engine", "WORKBENCH")
    samples = int(request.form.get("samples", 32))
    num_chunks = int(request.form.get("chunks") or num_workers)
    num_chunks = max(1, min(num_chunks, num_workers))

    blend_bytes = blend_file.read()
    blend_b64 = base64.b64encode(blend_bytes).decode("ascii")
    original_filename = blend_file.filename

    total_frames = frame_end - frame_start + 1
    frames_per_chunk = total_frames // num_chunks
    remainder = total_frames % num_chunks

    job_name = f"Blender Farm: {original_filename} (frames {frame_start}-{frame_end})"
    extra_params = {
        "blend_b64": blend_b64,
        "render_engine": render_engine,
        "samples": samples,
        "total_frames": total_frames,
        "frame_start": frame_start,
        "frame_end": frame_end
    }
    params_dict = {"chunks": num_chunks, "input_params": {"engine": render_engine, "samples": samples}, "extra_params": extra_params}

    is_queued = active_job is not None
    job_status = "QUEUED" if is_queued else "RUNNING"
    job_start_time = None if is_queued else time.time()

    job_id = db_create_job(job_name, "blender_render", params_dict, status=job_status, start_time=job_start_time)
    log_event(f"[*] Created blender_render job {job_id}: {job_name}")

    chunks_list = []
    for i in range(num_chunks):
        chunk_fs = frame_start + i * frames_per_chunk + min(i, remainder)
        chunk_fe = chunk_fs + frames_per_chunk + (1 if i < remainder else 0) - 1
        chunk_fe = min(chunk_fe, frame_end)
        chunks_list.append({
            "chunk_index": i,
            "total_chunks": num_chunks,
            "frame_start": chunk_fs,
            "frame_end": chunk_fe
        })

    for idx, c_data in enumerate(chunks_list):
        db_create_chunk(job_id, idx, c_data)

    if not is_queued:
        active_job = {
            "job_id": job_id,
            "name": job_name,
            "job_type": "blender_render",
            "status": "RUNNING",
            "start_time": job_start_time,
            "worker_count": num_workers,
            "extra_params": extra_params,
            "chunks": [],
            "results": {}
        }
        for idx, c_data in enumerate(chunks_list):
            active_job["chunks"].append({
                "chunk_index": idx,
                "chunk_data": c_data,
                "status": "PENDING",
                "worker_id": None,
                "progress": 0,
                "time_taken": 0.0,
                "retries": 0
            })
        emit_job_update()
        assign_pending_chunks()
    else:
        emit_job_update()

    return jsonify({"success": True, "job_id": job_id, "queued": is_queued})

@app.route("/api/cancel_job", methods=["POST"])
def cancel_job():
    global active_job, active_workers
    
    if request.is_json:
        data = request.json or {}
    else:
        data = request.form or {}
        
    job_id_val = data.get("job_id")
    job_id = int(job_id_val) if job_id_val else None
    
    # If no job_id is provided, cancel the active job
    if not job_id and active_job:
        job_id = active_job["job_id"]
        
    if not job_id:
        return jsonify({"success": False, "error": "No active job to cancel"}), 400
        
    # Check if the job to cancel is the active job
    if active_job and active_job["job_id"] == job_id:
        # Mark active job as failed/cancelled
        db_fail_job(job_id)
        
        # Notify workers to abort
        socketio.emit("abort_job", {"job_id": job_id})
        
        log_event(f"[*] Aborted active job {job_id}")
        
        # Reset workers to idle
        for sid, worker in active_workers.items():
            if worker["status"] in ["working", "benchmarking"]:
                worker["status"] = "idle"
            
        active_job = None
        
        emit_workers_update()
        emit_job_update()
        
        # Start next queued job
        trigger_next_job()
        
        return jsonify({"success": True, "message": f"Job {job_id} aborted successfully"})
        
    # Otherwise, check if it's a queued job
    with get_db() as db:
        job_row = db.execute("SELECT status FROM jobs WHERE id = ?", (job_id,)).fetchone()
        
    if job_row and job_row["status"] == "QUEUED":
        with get_db() as db:
            db.execute("UPDATE jobs SET status = 'FAILED' WHERE id = ?", (job_id,))
            db.commit()
        log_event(f"[*] Removed queued job {job_id} from queue")
        emit_job_update()
        return jsonify({"success": True, "message": f"Queued job {job_id} cancelled successfully"})
        
    return jsonify({"success": False, "error": "Job not found or not in running/queued state"}), 400

@app.route("/api/run_job", methods=["POST"])
def run_job():
    global active_job, active_workers
        
    # Check if there are active workers
    num_workers = len(active_workers)
    if num_workers == 0:
        return jsonify({"success": False, "error": "No laboratory computers are currently registered/online"}), 400
        
    if request.is_json:
        data = request.json or {}
    else:
        data = request.form or {}
        
    job_type = data.get("job_type")
    
    # User can optionally choose custom chunks, otherwise default to active workers count
    chunks_val = data.get("chunks")
    user_chunks = int(chunks_val) if chunks_val else num_workers
    num_chunks = max(1, user_chunks)
    
    job_name = ""
    chunks_list = []
    extra_params = {}
    
    try:
        if job_type == "prime_finder":
            start_val = int(data.get("prime_start") or 1)
            end_val = int(data.get("prime_end") or 1000000)
            if start_val >= end_val:
                return jsonify({"success": False, "error": "Start range must be smaller than end range"}), 400
            job_name = f"Prime Finder ({start_val:,} to {end_val:,})"
            chunks_list = chunker.chunk_range(start_val, end_val, num_chunks)
            
        elif job_type == "monte_carlo_pi":
            iterations = int(data.get("iterations") or 10000000)
            if iterations <= 0:
                return jsonify({"success": False, "error": "Iterations must be greater than zero"}), 400
            job_name = f"Monte Carlo Pi ({iterations:,} iterations)"
            chunks_list = chunker.chunk_monte_carlo(iterations, num_chunks)
            
        elif job_type == "matrix_multiply":
            matrix_size = int(data.get("matrix_size") or 500)
            if matrix_size < 10 or matrix_size > 1500:
                return jsonify({"success": False, "error": "Matrix dimension size must be between 10 and 1500"}), 400
                
            job_name = f"Matrix Multiply ({matrix_size}x{matrix_size})"
            
            # Generate Matrix A and Matrix B on Master
            # A and B are matrix size x matrix size
            A = np.random.randint(1, 10, size=(matrix_size, matrix_size)).tolist()
            B = np.random.randint(1, 10, size=(matrix_size, matrix_size)).tolist()
            
            # Chunk A by rows
            chunked_rows = chunker.chunk_data(A, num_chunks)
            chunks_list = [{"A_chunk": chunk} for chunk in chunked_rows]
            # Send B in extra params to all workers
            extra_params = {"B": B}
            
        elif job_type == "word_count":
            job_name = f"Word Count ({len(LARGE_TEXT_SOURCE):,} paragraphs)"
            # Chunk paragraphs array
            chunked_paragraphs = chunker.chunk_data(LARGE_TEXT_SOURCE, num_chunks)
            chunks_list = [{"paragraphs": chunk} for chunk in chunked_paragraphs]
            
        elif job_type == "custom_script":
            custom_file = request.files.get("custom_file")
            if not custom_file or custom_file.filename == "":
                return jsonify({"success": False, "error": "No Python script file selected"}), 400
            
            # Read script contents
            try:
                script_code = custom_file.read().decode("utf-8")
            except Exception as read_err:
                return jsonify({"success": False, "error": f"Failed to read file: {str(read_err)}"}), 400
                
            filename = custom_file.filename
            job_name = f"Custom Script ({filename})"
            
            chunks_list = []
            for idx in range(num_chunks):
                chunks_list.append({
                    "code": script_code,
                    "filename": filename,
                    "chunk_index": idx,
                    "total_chunks": num_chunks
                })
            
        elif job_type == "distributed_render":
            scene = data.get("scene", "plasma_wave")
            total_frames = int(data.get("total_frames", 120))
            job_name = f"Render Farm: {scene.replace('_',' ').title()} ({total_frames} frames, {num_chunks} workers)"
            extra_params = {"scene": scene, "total_frames": total_frames}
            chunks_list = [{"chunk_index": i, "total_chunks": num_chunks} for i in range(num_chunks)]

        elif job_type in ["render_frames", "blur_images", "process_portrait", "federated_learning", "grid_search_rf", "process_logs"]:
            job_names = {
                "render_frames":     "Animation Frame Render",
                "blur_images":       "Batch Image Blur",
                "process_portrait":  "Gaussian Portrait Blur",
                "federated_learning":"Federated AI Training Simulation",
                "grid_search_rf":    "Random Forest Hyperparameter Grid Search",
                "process_logs":      "Log Analyzer (42k log lines)",
            }
            job_name = f"{job_names[job_type]} ({num_chunks} Chunks)"
            chunks_list = []
            for idx in range(num_chunks):
                chunks_list.append({
                    "chunk_index": idx,
                    "total_chunks": num_chunks
                })
            
        elif job_type == "llm_inference":
            mode         = data.get("llm_mode", "ensemble")
            model        = data.get("llm_model", "deepseek-r1:7b").strip()
            ollama_url   = data.get("ollama_url", "http://localhost:11434").strip()
            system_prompt = data.get("system_prompt", "").strip()
            
            mode_labels = {
                "batch_qa": "Batch Q&A",
                "document": "Document Analysis",
                "ensemble": "Ensemble Reasoning",
            }
            job_name = f"LLM Inference [{model}] — {mode_labels.get(mode, mode)}"
            
            extra_params = {
                "model":       model,
                "ollama_url":  ollama_url,
                "mode":        mode,
            }
            if system_prompt:
                extra_params["system_prompt"] = system_prompt
            
            if mode == "batch_qa":
                raw_questions = data.get("questions", "")
                questions = [q.strip() for q in raw_questions.split("\n") if q.strip()]
                if not questions:
                    return jsonify({"success": False, "error": "No questions provided (one per line)"}), 400
                extra_params["questions"] = questions
                # Each chunk gets a share of questions
                num_chunks = min(num_chunks, len(questions))
                chunks_list = [{"chunk_index": i, "total_chunks": num_chunks} for i in range(num_chunks)]
                
            elif mode == "document":
                document_text = data.get("document", "").strip()
                if not document_text:
                    return jsonify({"success": False, "error": "No document text provided"}), 400
                extra_params["document"] = document_text
                chunks_list = [{"chunk_index": i, "total_chunks": num_chunks} for i in range(num_chunks)]
                
            elif mode == "ensemble":
                prompt_text = data.get("prompt", "").strip()
                if not prompt_text:
                    return jsonify({"success": False, "error": "No prompt provided"}), 400
                extra_params["prompt"] = prompt_text
                chunks_list = [{"chunk_index": i, "total_chunks": num_chunks} for i in range(num_chunks)]
            else:
                return jsonify({"success": False, "error": f"Unknown LLM mode: {mode}"}), 400

        else:
            return jsonify({"success": False, "error": "Invalid job type provided"}), 400
            
    except Exception as e:
        return jsonify({"success": False, "error": f"Error splitting job: {str(e)}"}), 400
        
    # Write job to SQLite database
    is_queued = active_job is not None
    job_status = "QUEUED" if is_queued else "RUNNING"
    job_start_time = None if is_queued else time.time()
    
    params_dict = {
        "chunks": num_chunks,
        "input_params": {k: v for k, v in data.items() if k not in ["job_type", "chunks"]},
        "extra_params": extra_params
    }
    job_id = db_create_job(job_name, job_type, params_dict, status=job_status, start_time=job_start_time)
    log_event(f"[*] Submitted Job {job_id}: {job_name} ({num_chunks} chunks, queued: {is_queued})")
    
    for idx, c_data in enumerate(chunks_list):
        db_create_chunk(job_id, idx, c_data)
        
    if not is_queued:
        # Store in-memory tracking structure
        active_job = {
            "job_id": job_id,
            "name": job_name,
            "job_type": job_type,
            "status": "RUNNING",
            "start_time": job_start_time,
            "worker_count": num_workers,
            "extra_params": extra_params,
            "chunks": [],
            "results": {}
        }
        
        for idx, c_data in enumerate(chunks_list):
            active_job["chunks"].append({
                "chunk_index": idx,
                "chunk_data": c_data,
                "status": "PENDING",
                "worker_id": None,
                "progress": 0,
                "time_taken": 0.0,
                "retries": 0
            })
            
        # Broadcast status to UI
        emit_job_update()
        
        # Distribute parallel tasks to workers
        assign_pending_chunks()
    else:
        # Broadcast that queue list has updated
        emit_job_update()
        
    return jsonify({"success": True, "job_id": job_id, "queued": is_queued})

# ----------------------------------------------------
# SocketIO Handlers for Workers & Dashboard UI
# ----------------------------------------------------

@socketio.on("connect")
def handle_connect():
    global ping_loop_started
    if not ping_loop_started:
        socketio.start_background_task(ping_workers_loop)
        ping_loop_started = True

@socketio.on("disconnect")
def handle_disconnect():
    global active_job
    sid = request.sid
    if sid in active_workers:
        worker = active_workers[sid]
        w_id = worker["worker_id"]
        log_event(f"[-] Worker disconnected: {w_id}")
        
        # If the disconnected worker was processing a chunk, reset that chunk to PENDING
        if active_job and active_job["status"] == "RUNNING":
            for chunk in active_job["chunks"]:
                if chunk["status"] == "RUNNING" and chunk["worker_id"] == w_id:
                    chunk["status"] = "PENDING"
                    chunk["worker_id"] = None
                    chunk["progress"] = 0
                    db_update_chunk(active_job["job_id"], chunk["chunk_index"], "PENDING", worker_id="")
                    log_event(f"[*] Fault Tolerance: Chunk {chunk['chunk_index']} reassigned from disconnected worker {w_id}")
                    
        # Remove from active workers list
        del active_workers[sid]
        emit_workers_update()
        
        # Trigger rescheduling to other connected workers
        if active_job and active_job["status"] == "RUNNING":
            emit_job_update()
            assign_pending_chunks()

@socketio.on("register_worker")
def handle_register_worker(data):
    # data: { worker_id, specs: { cores, ram_total, ram_free, disk_total, disk_free, os } }
    sid = request.sid
    worker_id = data.get("worker_id")
    ip_addr = request.remote_addr or "127.0.0.1"
    
    active_workers[sid] = {
        "worker_id": worker_id,
        "ip": ip_addr,
        "specs": data.get("specs", {}),
        "status": "idle"
    }
    log_event(f"[+] Lab worker registered: {worker_id} (IP: {ip_addr})")
    
    emit_workers_update()
    
    # If no job is running, check if there is a queued job and run it!
    if not active_job:
        trigger_next_job()
    elif active_job["status"] == "RUNNING":
        assign_pending_chunks()

@socketio.on("stats_update")
def handle_stats_update(data):
    sid = request.sid
    if sid in active_workers:
        active_workers[sid]["stats"] = data
        emit_workers_update()
    else:
        log_event(f"[WARNING] stats_update received from unregistered session {sid}: {data}")

@socketio.on("chunk_progress")
def handle_chunk_progress(data):
    # data: { job_id, chunk_index, progress }
    global active_job
    job_id = data.get("job_id")
    chunk_index = int(data.get("chunk_index"))
    progress = int(data.get("progress", 0))
    
    if active_job and active_job["job_id"] == job_id:
        for chunk in active_job["chunks"]:
            if chunk["chunk_index"] == chunk_index:
                chunk["progress"] = progress
                # Optional: update DB, but throttle to keep DB fast
                # We update in-memory and stream to UI
                break
        emit_job_update()

@socketio.on("chunk_complete")
def handle_chunk_complete(data):
    # data: { job_id, chunk_index, result, time_taken }
    global active_job
    sid = request.sid
    job_id = data.get("job_id")
    chunk_index = int(data.get("chunk_index"))
    result = data.get("result")
    time_taken = float(data.get("time_taken", 0.0))
    
    # Free the worker node
    if sid in active_workers:
        active_workers[sid]["status"] = "idle"
        emit_workers_update()
        
    if active_job and active_job["job_id"] == job_id:
        for chunk in active_job["chunks"]:
            if chunk["chunk_index"] == chunk_index:
                chunk["status"] = "COMPLETED"
                chunk["progress"] = 100
                chunk["time_taken"] = time_taken
                active_job["results"][chunk_index] = result
                
                # Update SQLite Chunk state
                db_update_chunk(
                    job_id, 
                    chunk_index, 
                    "COMPLETED", 
                    progress=100, 
                    time_taken=time_taken
                )
                log_event(f"[+] Chunk {chunk_index + 1}/{len(active_job['chunks'])} completed by worker {active_workers[sid]['worker_id']} in {time_taken:.2f}s")
                break
                
        emit_job_update()
        
        # Check if all chunks completed
        all_done = all(c["status"] == "COMPLETED" for c in active_job["chunks"])
        if all_done:
            finish_job()
        else:
            # Re-assign remaining pending chunks
            assign_pending_chunks()

@socketio.on("chunk_failed")
def handle_chunk_failed(data):
    # data: { job_id, chunk_index, error }
    global active_job, active_workers
    sid = request.sid
    job_id = data.get("job_id")
    chunk_index = int(data.get("chunk_index"))
    err_msg = data.get("error", "Unknown execution error")
    
    # Free the worker node
    if sid in active_workers:
        active_workers[sid]["status"] = "idle"
        emit_workers_update()
        
    if active_job and active_job["job_id"] == job_id:
        should_fail_job = False
        
        for chunk in active_job["chunks"]:
            if chunk["chunk_index"] == chunk_index:
                # Initialize retries if not present
                if "retries" not in chunk:
                    chunk["retries"] = 0
                    
                chunk["retries"] += 1
                
                # Fail the job immediately if it is a custom script or exceeds max retries
                if active_job["job_type"] == "custom_script" or chunk["retries"] >= 3:
                    should_fail_job = True
                    # Log error details inside SQLite as FAILED
                    db_update_chunk(job_id, chunk_index, "FAILED", worker_id="", error_msg=err_msg)
                    log_event(f"[!] Worker reported critical chunk failure: {err_msg}. Failing job {job_id}...")
                else:
                    chunk["status"] = "PENDING"
                    chunk["worker_id"] = None
                    chunk["progress"] = 0
                    db_update_chunk(job_id, chunk_index, "PENDING", worker_id="", error_msg=err_msg)
                    log_event(f"[!] Worker reported transient chunk failure: {err_msg}. Requeuing (Try {chunk['retries']}/3)...")
                break
                
        if should_fail_job:
            db_fail_job(job_id)
            
            # Broadcast failure details to UI
            socketio.emit("job_failed", {
                "job_id": job_id,
                "name": active_job["name"],
                "error": err_msg
            })
            
            # Reset active job state and release other workers
            active_job = None
            for w_sid, info in active_workers.items():
                if info["status"] == "working":
                    info["status"] = "idle"
            emit_workers_update()
            emit_job_update()
            
            # Start next queued job
            trigger_next_job()
        else:
            emit_job_update()
            assign_pending_chunks()

@socketio.on("trigger_benchmark")
def handle_trigger_benchmark(data):
    global active_workers
    worker_id = data.get("worker_id")
    
    # Find sid for this worker_id
    target_sid = None
    for sid, info in active_workers.items():
        if info["worker_id"] == worker_id:
            target_sid = sid
            break
            
    if target_sid:
        active_workers[target_sid]["status"] = "benchmarking"
        emit_workers_update()
        socketio.emit("run_benchmark", {}, to=target_sid)
        log_event(f"[*] Sent benchmark request to worker {worker_id}")

@socketio.on("benchmark_result")
def handle_benchmark_result(data):
    global active_workers
    sid = request.sid
    worker_id = data.get("worker_id")
    gflops = float(data.get("gflops", 0.0))
    
    if sid in active_workers:
        active_workers[sid]["status"] = "idle"
        active_workers[sid]["specs"]["gflops"] = gflops
        emit_workers_update()
        log_event(f"[+] Worker {worker_id} benchmark complete: {gflops:.2f} GFLOPS")

if __name__ == "__main__":
    print("="*60)
    print("CAMPUSGRID MASTER SERVER STARTING")
    print("LAN access url: http://localhost:5000")
    print("="*60)
    socketio.run(app, host="0.0.0.0", port=5000, debug=False, allow_unsafe_werkzeug=True)
