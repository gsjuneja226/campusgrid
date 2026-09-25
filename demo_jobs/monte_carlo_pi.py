"""
CampusGrid Monte Carlo Pi Job
Estimates Pi using the Monte Carlo method and reports progress.
"""
import random
import time

def run(iterations, progress_callback=None):
    """
    Runs a Monte Carlo simulation of 'iterations' steps to estimate Pi.
    """
    inside_circle = 0
    last_report_time = time.time()
    
    # We can batch random number generation for speed, but standard loops with progress callback are ideal for visual progress.
    # To optimize, we run in batches of 1% iterations per callback check.
    batch_size = max(1, iterations // 100)
    
    for i in range(iterations):
        x = random.random()
        y = random.random()
        if x*x + y*y <= 1.0:
            inside_circle += 1
            
        if progress_callback and (i % batch_size == 0 or i == iterations - 1):
            current_time = time.time()
            if current_time - last_report_time >= 0.1 or i == iterations - 1:
                pct = int((i + 1) / iterations * 100)
                progress_callback(pct)
                last_report_time = current_time
                
    if progress_callback:
        progress_callback(100)
        
    # Return estimate of Pi for this worker chunk
    return (inside_circle / iterations) * 4.0
