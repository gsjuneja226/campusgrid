"""
CampusGrid Worker Agent
Runs on lab computers, registers with the Master, reports hardware specs,
streams resource usage stats, and executes job chunks in parallel threads.
"""
import os
import sys
import time
import socket
import platform
import argparse
import threading
import socketio
import psutil
import ast

def verify_code_safety(code_str):
    """
    Parses code using AST (Abstract Syntax Tree) to check for restricted modules and functions.
    Prevents workers from executing dangerous system-level commands.
    """
    blocked_modules = {'subprocess', 'ctypes', 'sys', 'os', 'importlib', 'builtins', 'socket', 'urllib', 'requests'}
    blocked_functions = {'eval', 'exec', 'compile', 'getattr', 'setattr', 'delattr', 'globals', 'locals', 'dir', 'vars', 'open', '__import__'}
    blocked_attributes = {
        'system', 'popen', 'kill', 'spawn', 'rmtree',
        'execl', 'execle', 'execlp', 'execlpe', 'execv', 'execve', 'execvp', 'execvpe',
        'remove', 'unlink', 'rmdir'
    }
    
    try:
        tree = ast.parse(code_str)
    except SyntaxError as e:
        return False, f"Syntax Error: {e}"
        
    for node in ast.walk(tree):
        # 1. Check Name nodes (blocks direct access or aliasing of blocked functions/built-ins and dunder names)
        if isinstance(node, ast.Name):
            if node.id in blocked_functions:
                return False, f"Access to restricted function/built-in '{node.id}' is blocked."
            if node.id.startswith('__') or node.id.endswith('__'):
                return False, f"Access to restricted name/dunder '{node.id}' is blocked."
                
        # 2. Check Attribute nodes (blocks access to dangerous attributes and any dunder attribute like __dict__)
        elif isinstance(node, ast.Attribute):
            if node.attr in blocked_attributes:
                return False, f"Access to restricted attribute '{node.attr}' is blocked."
            if node.attr.startswith('__') or node.attr.endswith('__'):
                return False, f"Access to restricted attribute/dunder '{node.attr}' is blocked."
                
        # 3. Check direct imports
        elif isinstance(node, ast.Import):
            for alias in node.names:
                root_module = alias.name.split('.')[0]
                if root_module in blocked_modules:
                    return False, f"Import of module '{alias.name}' is restricted."
                if alias.name.startswith('__') or alias.name.endswith('__'):
                    return False, f"Import of restricted name '{alias.name}' is blocked."
                if alias.asname and (alias.asname.startswith('__') or alias.asname.endswith('__')):
                    return False, f"Import alias '{alias.asname}' is restricted."
                    
        # 4. Check from-imports
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root_module = node.module.split('.')[0]
                if root_module in blocked_modules:
                    return False, f"Import from module '{node.module}' is restricted."
                if node.module.startswith('__') or node.module.endswith('__'):
                    return False, f"Import from restricted module name '{node.module}' is blocked."
            for alias in node.names:
                if alias.name in blocked_attributes or alias.name in blocked_functions:
                    return False, f"Import of restricted name '{alias.name}' is blocked."
                if alias.name.startswith('__') or alias.name.endswith('__'):
                    return False, f"Import of restricted name '{alias.name}' is blocked."
                if alias.asname and (alias.asname.startswith('__') or alias.asname.endswith('__')):
                    return False, f"Import alias '{alias.asname}' is restricted."
                    
    return True, "Verified safe"

def _run_isolated_script_process(code, filename, chunk_index, total_chunks, result_queue, progress_queue):
    """Runs in a separate process to execute custom script code with isolation"""
    import sys
    import os
    import io
    import contextlib
    import importlib.util
    import inspect
    
    clean_filename = os.path.basename(filename)
    try:
        with open(clean_filename, "w", encoding="utf-8") as f:
            f.write(code)
    except Exception as e:
        result_queue.put((False, f"Failed to save script: {str(e)}", None))
        return

    def isolated_progress_callback(pct):
        try:
            progress_queue.put(pct)
        except Exception:
            pass

    stdout_capture = io.StringIO()
    run_result = None
    success = True
    module_name = f"custom_script_module_{chunk_index}"
    
    try:
        spec = importlib.util.spec_from_file_location(module_name, clean_filename)
        custom_module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = custom_module
        
        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stdout_capture):
            spec.loader.exec_module(custom_module)
            
            if hasattr(custom_module, "run"):
                run_func = getattr(custom_module, "run")
                sig = inspect.signature(run_func)
                
                required_params = []
                for param_name, param in sig.parameters.items():
                    if param_name in ["progress_callback", "chunk_index", "total_chunks"]:
                        continue
                    if param.default == inspect.Parameter.empty and param.kind in (
                        inspect.Parameter.POSITIONAL_ONLY,
                        inspect.Parameter.POSITIONAL_OR_KEYWORD
                    ):
                        required_params.append(param_name)
                        
                if len(required_params) == 0:
                    run_args = {}
                    if "progress_callback" in sig.parameters:
                        run_args["progress_callback"] = isolated_progress_callback
                    if "chunk_index" in sig.parameters:
                        run_args["chunk_index"] = chunk_index
                    if "total_chunks" in sig.parameters:
                        run_args["total_chunks"] = total_chunks
                    
                    run_result = run_func(**run_args)
                else:
                    stdout_capture.write(
                        f"\n[!] Note: Exposing a run() function that requires arguments {required_params}.\n"
                        "Since this is executed as a generic custom script, the run() call was skipped.\n"
                        "Make sure required parameters have default values, or put execution code at the top-level.\n"
                    )
    except Exception as script_err:
        success = False
        run_result = str(script_err)
        stdout_capture.write(f"\nRuntime Error: {script_err}\n")
    finally:
        if module_name in sys.modules:
            del sys.modules[module_name]
        try:
            if os.path.exists(clean_filename):
                os.remove(clean_filename)
        except Exception:
            pass
            
    captured_stdout = stdout_capture.getvalue()
    result_queue.put((success, captured_stdout, run_result))


# Add directories to path to ensure demo_jobs imports correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from demo_jobs import prime_finder, monte_carlo_pi, matrix_multiply, word_count

# Setup socketio client
sio = socketio.Client()

# Global variables
worker_id = ""
master_url = ""
active_job_thread = None
running_chunk = None

abort_flag = False
current_job_id = None

reporter_running = False
reporter_lock = threading.Lock()
donated_cores = 0

def get_system_specs():
    """Gathers hardware and OS specifications of the computer"""
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage(os.path.abspath(os.sep))
    
    total_cores = psutil.cpu_count(logical=True)
    actual_donated = donated_cores if 0 < donated_cores <= total_cores else total_cores
    
    return {
        "cores": total_cores,
        "donated_cores": actual_donated,
        "ram_total": round(mem.total / (1024 ** 3), 1),  # GB
        "ram_free": round(mem.available / (1024 ** 3), 1),
        "disk_total": round(disk.total / (1024 ** 3), 1), # GB
        "disk_free": round(disk.free / (1024 ** 3), 1),
        "os": f"{platform.system()} {platform.release()}"
    }

def get_system_stats():
    """Gathers real-time performance statistics (CPU, RAM, Disk usage)"""
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage(os.path.abspath(os.sep))
    
    return {
        "cpu_usage": int(psutil.cpu_percent(interval=None)),
        "ram_usage": int(mem.percent),
        "disk_usage": int(disk.percent)
    }

def resource_reporter_loop():
    """Background thread that streams CPU/RAM updates to the master every 2 seconds"""
    global reporter_running
    with reporter_lock:
        if reporter_running:
            return
        reporter_running = True
        
    # Wait briefly for connection to be fully established and acknowledged
    time.sleep(1.0)
    try:
        while True:
            # If temporarily disconnected, wait for reconnection
            if not sio.connected:
                time.sleep(1.0)
                continue
            try:
                stats = get_system_stats()
                sio.emit("stats_update", stats)
            except Exception as e:
                print(f"[!] Error gathering stats: {e}")
            time.sleep(2.0)
    finally:
        with reporter_lock:
            reporter_running = False

# ----------------------------------------------------
# SocketIO Client Event Handlers
# ----------------------------------------------------

@sio.event
def connect():
    print(f"[+] Connected to CampusGrid Master at {master_url}")
    # Register this worker node with its specs
    specs = get_system_specs()
    sio.emit("register_worker", {
        "worker_id": worker_id,
        "specs": specs
    })
    print(f"[*] Worker '{worker_id}' registered successfully.")
    
    # Start the system stats reporting thread
    t = threading.Thread(target=resource_reporter_loop, daemon=True)
    t.start()

@sio.event
def disconnect():
    print("[-] Disconnected from Master server.")

@sio.on("ping_node")
def handle_ping_node(data):
    try:
        sio.emit("pong_node", {"sent_at": data.get("sent_at")})
    except Exception:
        pass

def execute_chunk_task(data):
    """Runs inside a separate thread to prevent blocking SocketIO network loop"""
    global running_chunk, abort_flag, current_job_id
    job_id = data.get("job_id")
    chunk_index = data.get("chunk_index")
    total_chunks = data.get("total_chunks", 1)
    job_type = data.get("job_type")
    chunk_data = data.get("chunk_data", {})
    extra_params = data.get("extra_params", {})
    
    abort_flag = False
    current_job_id = job_id
    running_chunk = chunk_index
    print(f"\n[*] Received Chunk {chunk_index} of Job {job_id} ({job_type})")
    print(f"[*] Executing...")
    
    start_time = time.time()
    
    # Progress callback helper to stream percentage back to master
    def progress_callback(pct):
        global abort_flag
        if abort_flag:
            raise InterruptedError("Job execution aborted by master")
            
        print(f"Processing chunk {chunk_index} -> {pct}% done", end="\r")
        try:
            sio.emit("chunk_progress", {
                "job_id": job_id,
                "chunk_index": chunk_index,
                "progress": pct
            })
        except Exception:
            pass

    try:
        result = None
        if job_type == "prime_finder":
            start = chunk_data["start"]
            end = chunk_data["end"]
            result = prime_finder.run(start, end, progress_callback)
            
        elif job_type == "monte_carlo_pi":
            iterations = chunk_data["iterations"]
            result = monte_carlo_pi.run(iterations, progress_callback)
            
        elif job_type == "matrix_multiply":
            A_chunk = chunk_data["A_chunk"]
            B = extra_params["B"]
            result = matrix_multiply.run(A_chunk, B, progress_callback)
            
        elif job_type == "word_count":
            paragraphs = chunk_data["paragraphs"]
            result = word_count.run(paragraphs, progress_callback)

        elif job_type == "distributed_render":
            from demo_jobs import distributed_render
            # extra_params contains scene and total_frames
            result = distributed_render.run(
                chunk_index, total_chunks, progress_callback,
                extra_params=extra_params
            )

        elif job_type == "video_processing":
            from demo_jobs import video_processing
            # chunk_data contains an "extra_segment" dict with segment_b64, effect, segment_ext
            seg_info = chunk_data.get("extra_segment", {})
            # Merge into extra_params for the module
            vp_params = {**extra_params, **seg_info}
            result = video_processing.run(
                chunk_index, total_chunks, progress_callback,
                extra_params=vp_params
            )

        elif job_type == "blender_render":
            from demo_jobs import blender_render
            # chunk_data has frame_start, frame_end
            # extra_params has blend_b64, render_engine, samples
            blend_params = {
                **extra_params,
                "frame_start": chunk_data.get("frame_start", 1),
                "frame_end":   chunk_data.get("frame_end", 1),
            }
            result = blender_render.run(
                chunk_index, total_chunks, progress_callback,
                extra_params=blend_params
            )

        elif job_type == "llm_inference":
            from demo_jobs import llm_inference
            result = llm_inference.run(
                chunk_index, total_chunks, progress_callback,
                extra_params=extra_params
            )

        elif job_type in ["render_frames", "blur_images", "process_portrait", "federated_learning", "grid_search_rf", "process_logs"]:
            import importlib
            module_name = f"demo_jobs.{job_type}"
            if module_name in sys.modules:
                del sys.modules[module_name]
            module = importlib.import_module(module_name)
            result = module.run(chunk_index, total_chunks, progress_callback)
            
        elif job_type == "custom_script":
            code = chunk_data["code"]
            filename = chunk_data.get("filename", "custom_job.py")
            
            # Run code safety static analysis check before saving or executing
            is_safe, msg = verify_code_safety(code)
            if not is_safe:
                raise PermissionError(f"Security Policy Violation: {msg}")
                
            import multiprocessing
            
            result_queue = multiprocessing.Queue()
            progress_queue = multiprocessing.Queue()
            
            p = multiprocessing.Process(
                target=_run_isolated_script_process,
                args=(code, filename, chunk_index, total_chunks, result_queue, progress_queue),
                daemon=True
            )
            p.start()
            
            timeout = 30.0
            start_wait = time.time()
            success = False
            captured_stdout = ""
            run_result = None
            
            while p.is_alive():
                # Check if master requested abort
                if abort_flag:
                    p.terminate()
                    p.join()
                    raise InterruptedError("Job execution aborted by master")
                    
                # Flush progress queue
                while not progress_queue.empty():
                    try:
                        pct = progress_queue.get_nowait()
                        progress_callback(pct)
                    except Exception:
                        pass
                        
                # Check timeout limits
                elapsed = time.time() - start_wait
                if elapsed > timeout:
                    p.terminate()
                    p.join()
                    raise TimeoutError(f"Script execution timed out after exceeding {timeout} seconds limit.")
                    
                time.sleep(0.1)
                
            # Final progress queue drain
            while not progress_queue.empty():
                try:
                    pct = progress_queue.get_nowait()
                    progress_callback(pct)
                except Exception:
                    pass
                    
            # Retrieve queue payload
            if not result_queue.empty():
                queue_val = result_queue.get()
                success = queue_val[0]
                captured_stdout = queue_val[1]
                run_result = queue_val[2]
            else:
                raise RuntimeError("Process completed without returning validation outcomes.")
                
            if not success:
                raise Exception(f"Isolated Runtime Error: {run_result}\nStdout Console:\n{captured_stdout}")
                
            result = captured_stdout
            if run_result is not None:
                if result:
                    result += "\n--- Return Value ---\n"
                result += str(run_result)
                
            if not result:
                result = "Script executed successfully with empty output."
            
        else:
            raise ValueError(f"Unsupported job type: {job_type}")
            
        time_taken = time.time() - start_time
        print(f"\n[+] Chunk {chunk_index} complete! Time taken: {time_taken:.2f}s")
        
        # Send completed payload
        sio.emit("chunk_complete", {
            "job_id": job_id,
            "chunk_index": chunk_index,
            "result": result,
            "time_taken": time_taken
        })
        
    except InterruptedError:
        print(f"\n[!] Chunk {chunk_index} execution successfully aborted.")
        return  # Exit thread cleanly without sending chunk_failed
    except Exception as e:
        print(f"\n[!] Chunk execution error: {e}")
        try:
            sio.emit("chunk_failed", {
                "job_id": job_id,
                "chunk_index": chunk_index,
                "error": str(e)
            })
        except Exception:
            pass
            
    finally:
        running_chunk = None

@sio.on("assign_chunk")
def handle_assign_chunk(data):
    """Receives chunk assignment and spawns a background executor thread"""
    global active_job_thread, running_chunk
    if running_chunk is not None:
        print(f"[!] Worker is currently executing chunk {running_chunk}. Rejecting chunk assignment.")
        return
        
    active_job_thread = threading.Thread(target=execute_chunk_task, args=(data,), daemon=True)
    active_job_thread.start()

@sio.on("abort_job")
def handle_abort_job(data):
    global abort_flag, current_job_id, running_chunk
    job_id = data.get("job_id")
    if job_id == current_job_id:
        print(f"\n[!] Master requested abort for active Job {job_id}.")
        abort_flag = True
        running_chunk = None

@sio.on("run_benchmark")
def handle_run_benchmark(data):
    """Executes a NumPy-based float matrix multiplication test to calculate GFLOPS"""
    global worker_id
    print("\n[*] Initializing NumPy FLOPS performance benchmarking...")
    
    try:
        import numpy as np
        
        # Standard NumPy float benchmarking
        # Multiply two 1000x1000 float matrices
        size = 1000
        runs = 3
        durations = []
        
        for run in range(runs):
            A = np.random.rand(size, size).astype(np.float64)
            B = np.random.rand(size, size).astype(np.float64)
            
            start_time = time.time()
            C = np.dot(A, B)
            dur = time.time() - start_time
            durations.append(dur)
            time.sleep(0.05)
            
        avg_dur = sum(durations) / runs
        
        # 1000^3 multiplications and 1000^3 additions = 2 * 1000^3 operations
        flops = (2.0 * (size ** 3)) / avg_dur
        gflops = flops / 1e9
        
        print(f"[+] Benchmark complete. Speed: {gflops:.2f} GFLOPS")
        
        sio.emit("benchmark_result", {
            "worker_id": worker_id,
            "gflops": round(gflops, 2)
        })
    except Exception as e:
        print(f"[!] Benchmarking failed: {e}")
        sio.emit("benchmark_result", {
            "worker_id": worker_id,
            "gflops": 0.0
        })

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CampusGrid Lab Worker Agent")
    parser.add_argument("--master", type=str, default="localhost", help="Master LAN IP (e.g. 192.168.1.15)")
    parser.add_argument("--id", type=str, default="", help="Custom unique Worker ID (e.g. PC-LAB-05)")
    parser.add_argument("--cores", type=int, default=0, help="Number of CPU cores/threads to donate (0 = use all)")
    args = parser.parse_args()
    
    donated_cores = args.cores
    
    # Generate unique ID if none provided
    if args.id:
        worker_id = args.id
    else:
        # Generate ID based on hostname plus a random segment
        hostname = socket.gethostname().upper()
        # Clean hostname
        hostname = "".join(c for c in hostname if c.isalnum() or c in "-_")
        worker_id = f"{hostname}-{os.getpid() % 1000:03d}"
        
    # Setup master connection address
    # If the user specified a master parameter with a port, use it, otherwise assume 5000
    host = args.master
    if ":" in host:
        master_url = f"http://{host}"
    else:
        master_url = f"http://{host}:5000"
        
    print("="*60)
    print("CAMPUSGRID LAB WORKER AGENT")
    print(f"Worker Node ID:  {worker_id}")
    print(f"Connecting to:   {master_url}")
    print("="*60)
    
    # Establish persistent connection and block main thread with automatic retries
    connected = False
    while not connected:
        try:
            sio.connect(master_url)
            connected = True
            sio.wait()
        except Exception as e:
            print(f"[!] Connection to Master at {master_url} failed: {e}")
            print("[*] Retrying connection in 2 seconds...")
            time.sleep(2.0)
