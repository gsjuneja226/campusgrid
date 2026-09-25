import time

def run(chunk_index, total_chunks, progress_callback=None):
    # 1. Simulating a massive raw log file
    base_logs = [
        "IP:192.168.1.1 - GET /index.html - STATUS:200",
        "IP:192.168.1.5 - POST /login - STATUS:401",
        "IP:192.168.1.2 - GET /dashboard - STATUS:200",
        "IP:192.168.1.1 - GET /app.js - STATUS:200",
        "IP:192.168.1.9 - GET /api/v1/data - STATUS:500",
        "IP:192.168.1.5 - GET /logout - STATUS:200",
        "IP:192.168.1.4 - POST /upload - STATUS:403"
    ] * 6000  # Creates ~42,000 log lines
    
    # 2. Partition logs based on worker chunk context
    chunk_size = len(base_logs) // total_chunks
    remainder = len(base_logs) % total_chunks
    
    start_idx = chunk_index * chunk_size + min(chunk_index, remainder)
    end_idx = start_idx + chunk_size + (1 if chunk_index < remainder else 0)
    
    my_logs = base_logs[start_idx:end_idx]
    total_my_logs = len(my_logs)
    
    # 3. Map Phase: Count occurrences
    status_counts = {}
    for idx, log in enumerate(my_logs):
        parts = log.split(" - STATUS:")
        if len(parts) == 2:
            status = parts[1]
            status_counts[status] = status_counts.get(status, 0) + 1
            
        # Update progress every 1,000 lines
        if idx % 1000 == 0 and progress_callback:
            progress_callback(int((idx / total_my_logs) * 100))
            
    if progress_callback:
        progress_callback(100)
        
    return f"Processed logs {start_idx:,} to {end_idx:,}. Local status counts: {status_counts}"
