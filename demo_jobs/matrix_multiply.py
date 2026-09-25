"""
CampusGrid Matrix Multiplication Job
Multiplies a chunk of rows from Matrix A with the full Matrix B.
"""
import numpy as np
import time

def run(A_chunk, B, progress_callback=None):
    """
    Multiplies A_chunk (list of rows) by B (full matrix list of rows).
    Uses numpy for efficient multiplication but computes row-by-row
    to track and report progress accurately.
    """
    A_arr = np.array(A_chunk)
    B_arr = np.array(B)
    
    num_rows = len(A_arr)
    num_cols_B = B_arr.shape[1]
    
    # Initialize result matrix segment
    result_chunk = np.zeros((num_rows, num_cols_B))
    
    last_report_time = time.time()
    
    for i in range(num_rows):
        result_chunk[i] = np.dot(A_arr[i], B_arr)
        
        if progress_callback and (i % max(1, num_rows // 20) == 0 or i == num_rows - 1):
            current_time = time.time()
            if current_time - last_report_time >= 0.1 or i == num_rows - 1:
                pct = int((i + 1) / num_rows * 100)
                progress_callback(pct)
                last_report_time = current_time
                
    if progress_callback:
        progress_callback(100)
        
    return result_chunk.tolist()
