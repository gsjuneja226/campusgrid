import time
import numpy as np

def run(chunk_index, total_chunks, progress_callback=None):
    total_frames = 120  # Total frame budget for a 5-second animation
    
    # Partition frames for this worker
    chunk_size = total_frames // total_chunks
    remainder = total_frames % total_chunks
    
    start_frame = chunk_index * chunk_size + min(chunk_index, remainder) + 1
    end_frame = start_frame + chunk_size + (1 if chunk_index < remainder else 0) - 1
    
    frames_to_render = list(range(start_frame, end_frame + 1))
    
    print(f"Rendering frames {start_frame} to {end_frame}...")
    
    for idx, frame in enumerate(frames_to_render):
        # Simulate render workload by doing a matrix multiplication
        matrix_dim = 600
        A = np.random.rand(matrix_dim, matrix_dim)
        np.dot(A, A)
        
        time.sleep(0.1)  # Simulate file output latency
        
        if progress_callback:
            progress_callback(int(((idx + 1) / len(frames_to_render)) * 100))
            
    return f"Success: Rendered frames {start_frame:03d} to {end_frame:03d} ({len(frames_to_render)} total frames)."
