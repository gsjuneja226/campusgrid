"""
CampusGrid: Distributed LLM Inference Worker
Calls the local Ollama REST API to run reasoning LLMs (DeepSeek-R1, Llama, Qwen, etc.)
on the worker machine's own RAM/GPU. Streams tokens back to master in real time.

Modes:
  - batch_qa  : Worker answers a slice of a question list
  - document  : Worker summarises a section of a large document
  - ensemble  : Worker answers the same prompt (all workers compare answers)
"""

import json
import sys
import time
import urllib.request
import urllib.error

DEFAULT_OLLAMA_URL  = "http://localhost:11434"
STREAM_TIMEOUT_SEC  = 600
CONNECT_TIMEOUT_SEC = 10



import subprocess
import shutil

def _ensure_ollama(ollama_url: str, model: str) -> tuple:
    """
    1. Check if Ollama is already running.
    2. If not, try to auto-start it (ollama serve) in the background.
    3. Wait up to 15 seconds for it to come up.
    4. Check if the model is pulled; if not, pull it automatically.
    Returns (ok: bool, message: str)
    """
    # First check: already running?
    try:
        urllib.request.urlopen(f"{ollama_url}/api/tags", timeout=5)
        print("[LLM] Ollama already running.")
    except Exception:
        # Not running — try to start it
        if not shutil.which("ollama"):
            return False, (
                "Ollama is not installed on this worker.\n"
                "Download and install from: https://ollama.com\n"
                "Then pull your model: ollama pull deepseek-r1:7b"
            )

        print("[LLM] Ollama not running — starting automatically with: ollama serve")
        try:
            # Start ollama serve as a detached background process
            if sys.platform == "win32":
                proc = subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW
                )
            else:
                proc = subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True
                )
        except Exception as e:
            return False, f"Failed to start Ollama: {e}"

        # Wait up to 15 seconds for Ollama to become ready
        print("[LLM] Waiting for Ollama to start...")
        deadline = time.time() + 15
        ready = False
        while time.time() < deadline:
            time.sleep(1.0)
            try:
                urllib.request.urlopen(f"{ollama_url}/api/tags", timeout=3)
                ready = True
                break
            except Exception:
                pass

        if not ready:
            return False, (
                "Ollama started but did not become ready within 15 seconds.\n"
                "Try running 'ollama serve' manually in a terminal on this worker."
            )
        print("[LLM] Ollama is now ready!")

    # Check if model is pulled; auto-pull if missing
    try:
        resp = urllib.request.urlopen(f"{ollama_url}/api/tags", timeout=10)
        tags_data = json.loads(resp.read().decode("utf-8"))
        pulled_models = [m["name"] for m in tags_data.get("models", [])]

        # Normalize: "deepseek-r1:7b" and "deepseek-r1" should match
        model_base = model.split(":")[0]
        model_available = any(
            m == model or m.startswith(model_base + ":")
            for m in pulled_models
        )

        if not model_available:
            print(f"[LLM] Model '{model}' not found locally. Pulling now (this may take a while)...")
            pull_payload = json.dumps({"name": model, "stream": False}).encode("utf-8")
            pull_req = urllib.request.Request(
                f"{ollama_url}/api/pull",
                data=pull_payload,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            # Pull can take minutes for large models — use a long timeout
            urllib.request.urlopen(pull_req, timeout=600)
            print(f"[LLM] Model '{model}' pulled successfully.")

    except Exception as e:
        # Non-fatal: model check failed, let generation attempt proceed
        print(f"[LLM] Warning: could not verify/pull model: {e}")

    return True, ""



def _stream_ollama(prompt: str, model: str, ollama_url: str,
                   system_prompt: str = "", progress_callback=None) -> dict:
    """Stream generation from Ollama; returns dict with full_text, think_text, answer_text."""
    payload = {
        "model":  model,
        "prompt": prompt,
        "stream": True,
        "options": {"temperature": 0.6, "top_p": 0.9},
    }
    if system_prompt:
        payload["system"] = system_prompt

    data = json.dumps(payload).encode("utf-8")
    req  = urllib.request.Request(
        f"{ollama_url}/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    full_text   = ""
    token_count = 0
    start_time  = time.time()
    last_report = time.time()

    try:
        with urllib.request.urlopen(req, timeout=STREAM_TIMEOUT_SEC) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue

                token = chunk.get("response", "")
                full_text   += token
                token_count += 1

                now = time.time()
                if progress_callback and now - last_report > 0.5:
                    last_report = now
                    approx_pct  = min(95, int(token_count / 8))
                    progress_callback(approx_pct)

                if chunk.get("done"):
                    break

    except urllib.error.URLError as e:
        raise ConnectionError(
            f"Ollama not reachable at {ollama_url}: {e}\n"
            "Make sure Ollama is running: ollama serve"
        )

    duration = time.time() - start_time

    # Split <think> reasoning block (DeepSeek-R1 / QwQ style)
    think_text  = ""
    answer_text = full_text

    if "<think>" in full_text:
        think_start = full_text.find("<think>") + len("<think>")
        think_end   = full_text.find("</think>")
        if think_end > think_start:
            think_text  = full_text[think_start:think_end].strip()
            answer_text = full_text[think_end + len("</think>"):].strip()
        else:
            think_text  = full_text[think_start:].strip()
            answer_text = "(Reasoning in progress...)"

    return {
        "full_text":    full_text,
        "think_text":   think_text,
        "answer_text":  answer_text,
        "tokens":       token_count,
        "duration_sec": round(duration, 2),
    }


def run(chunk_index: int, total_chunks: int, progress_callback, extra_params: dict = None) -> str:
    """
    Main entry point called by worker.py.

    extra_params:
        model         (str)  : e.g. "deepseek-r1:7b"
        ollama_url    (str)  : default http://localhost:11434
        mode          (str)  : "batch_qa" | "document" | "ensemble"
        questions     (list) : for batch_qa
        document      (str)  : for document mode
        prompt        (str)  : for ensemble mode
        system_prompt (str)  : optional
    """
    if extra_params is None:
        extra_params = {}

    model         = extra_params.get("model", "deepseek-r1:7b")
    ollama_url    = extra_params.get("ollama_url", DEFAULT_OLLAMA_URL).rstrip("/")
    mode          = extra_params.get("mode", "ensemble")
    system_prompt = extra_params.get(
        "system_prompt",
        "You are a helpful, thorough reasoning assistant. Think carefully before answering."
    )

    # Verify Ollama is running on this worker — auto-start if needed
    progress_callback(2)
    ok, err = _ensure_ollama(ollama_url, model)
    if not ok:
        raise ConnectionError(err)

    results_parts = []

    # ── BATCH Q&A ─────────────────────────────────────────────────────────────
    if mode == "batch_qa":
        all_questions = extra_params.get("questions", [])
        if not all_questions:
            raise ValueError("No questions provided for batch_qa mode.")

        chunk_size   = max(1, (len(all_questions) + total_chunks - 1) // total_chunks)
        start_idx    = chunk_index * chunk_size
        end_idx      = min(start_idx + chunk_size, len(all_questions))
        my_questions = all_questions[start_idx:end_idx]

        if not my_questions:
            return f"[Worker {chunk_index + 1}] No questions assigned to this worker."

        print(f"[LLM] Worker {chunk_index}: answering {len(my_questions)} question(s) with {model}")

        for q_idx, question in enumerate(my_questions):
            prompt = (
                f"Question: {question}\n\n"
                f"Please reason through this carefully and provide a thorough answer."
            )

            base_pct  = int((q_idx / len(my_questions)) * 90)
            share_pct = int(90 / len(my_questions))

            def _cb(pct, _qi=q_idx, _base=base_pct, _share=share_pct):
                progress_callback(min(95, _base + int(pct / 100 * _share)))

            gen = _stream_ollama(prompt, model, ollama_url, system_prompt, _cb)

            part = f"### Q{start_idx + q_idx + 1}: {question}\n\n"
            if gen["think_text"]:
                part += f"**Reasoning Trace:**\n{gen['think_text']}\n\n"
            part += f"**Answer:**\n{gen['answer_text']}\n"
            part += f"\n*({gen['tokens']} tokens · {gen['duration_sec']}s)*\n"
            part += "\n" + "-" * 60 + "\n\n"
            results_parts.append(part)

    # ── DOCUMENT ANALYSIS ─────────────────────────────────────────────────────
    elif mode == "document":
        full_doc = extra_params.get("document", "")
        if not full_doc.strip():
            raise ValueError("No document provided for document mode.")

        doc_len  = len(full_doc)
        seg_size = (doc_len + total_chunks - 1) // total_chunks
        start_ch = chunk_index * seg_size
        end_ch   = min(start_ch + seg_size, doc_len)
        section  = full_doc[start_ch:end_ch].strip()

        prompt = (
            f"Carefully read the following document section and provide:\n"
            f"1. A concise summary (3–5 sentences)\n"
            f"2. Key facts and important points\n"
            f"3. Notable entities, concepts, or conclusions\n\n"
            f"--- SECTION {chunk_index + 1} of {total_chunks} ---\n\n"
            f"{section}\n\n"
            f"--- END OF SECTION ---\n\n"
            f"Provide your structured analysis:"
        )

        print(f"[LLM] Worker {chunk_index}: analysing {len(section)} char document section with {model}")

        def _cb(pct):
            progress_callback(min(95, pct))

        gen = _stream_ollama(prompt, model, ollama_url, system_prompt, _cb)

        part = f"## Section {chunk_index + 1}/{total_chunks} Analysis\n\n"
        if gen["think_text"]:
            part += f"**Reasoning Process:**\n{gen['think_text']}\n\n"
        part += f"**Analysis:**\n{gen['answer_text']}\n"
        part += f"\n*({gen['tokens']} tokens · {gen['duration_sec']}s)*\n\n"
        results_parts.append(part)

    # ── ENSEMBLE REASONING ────────────────────────────────────────────────────
    elif mode == "ensemble":
        prompt_text = extra_params.get("prompt", "")
        if not prompt_text.strip():
            raise ValueError("No prompt provided for ensemble mode.")

        # Each worker uses a slightly different reasoning approach
        variations = [
            "Think step-by-step from first principles.",
            "Analyse systematically, listing all considerations.",
            "Consider multiple perspectives and counterarguments.",
            "Focus on evidence and reason from facts outward.",
            "Break the problem into core components, then synthesise.",
            "Identify assumptions, then reason from them carefully.",
            "Consider the problem from both broad and specific angles.",
            "Apply logical deduction and check each step rigorously.",
        ]
        variation = variations[chunk_index % len(variations)]
        prompt = f"{prompt_text}\n\n[Approach: {variation}]"

        print(f"[LLM] Worker {chunk_index}: ensemble path '{variation}' with {model}")

        def _cb(pct):
            progress_callback(min(95, pct))

        gen = _stream_ollama(prompt, model, ollama_url, system_prompt, _cb)

        part = f"## Worker {chunk_index + 1} — Reasoning Path: {variation}\n\n"
        if gen["think_text"]:
            part += f"**Internal Reasoning:**\n{gen['think_text']}\n\n"
        part += f"**Final Answer:**\n{gen['answer_text']}\n"
        part += f"\n*({gen['tokens']} tokens · {gen['duration_sec']}s)*\n\n"
        results_parts.append(part)

    else:
        raise ValueError(f"Unknown LLM inference mode: '{mode}'")

    progress_callback(100)
    return "\n".join(results_parts)
