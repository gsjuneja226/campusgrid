"""
CampusGrid Chunker Engine
Handles the Map phase: splits job parameters into parallelizable chunks.
"""

def chunk_range(start, end, num_chunks):
    """
    Splits a number range [start, end] into equal-ish sub-ranges.
    """
    total = end - start + 1
    if total <= 0:
        raise ValueError("Invalid range: start must be less than or equal to end")
    
    if num_chunks <= 0:
        raise ValueError("Number of chunks must be greater than 0")
        
    num_chunks = min(num_chunks, total)
    base_chunk_size = total // num_chunks
    remainder = total % num_chunks
    
    chunks = []
    current_start = start
    
    for i in range(num_chunks):
        # Distribute remainder across the first few chunks
        size = base_chunk_size + (1 if i < remainder else 0)
        current_end = current_start + size - 1
        chunks.append({
            "start": int(current_start),
            "end": int(current_end)
        })
        current_start = current_end + 1
        
    return chunks

def chunk_data(data_list, num_chunks):
    """
    Splits a list of items into N slices.
    """
    if not isinstance(data_list, list):
        raise TypeError("Data must be a list for DATA jobs")
        
    total = len(data_list)
    if total == 0:
        return []
        
    if num_chunks <= 0:
        raise ValueError("Number of chunks must be greater than 0")
        
    num_chunks = min(num_chunks, total)
    base_chunk_size = total // num_chunks
    remainder = total % num_chunks
    
    chunks = []
    current_index = 0
    
    for i in range(num_chunks):
        size = base_chunk_size + (1 if i < remainder else 0)
        end_index = current_index + size
        chunks.append(data_list[current_index:end_index])
        current_index = end_index
        
    return chunks

def chunk_monte_carlo(total_iterations, num_chunks):
    """
    Splits total iterations for a simulation into N equal-ish iterations.
    """
    if total_iterations <= 0:
        raise ValueError("Iterations must be greater than 0")
        
    if num_chunks <= 0:
        raise ValueError("Number of chunks must be greater than 0")
        
    num_chunks = min(num_chunks, total_iterations)
    base_chunk_size = total_iterations // num_chunks
    remainder = total_iterations % num_chunks
    
    chunks = []
    for i in range(num_chunks):
        size = base_chunk_size + (1 if i < remainder else 0)
        chunks.append({
            "iterations": int(size)
        })
        
    return chunks
