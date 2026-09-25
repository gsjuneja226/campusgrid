"""
CampusGrid Reducer Engine
Handles the Reduce phase: combines partial worker results into the final output.
"""

def sum_reduce(results):
    """
    Adds all partial numerical results.
    """
    if not results:
        return 0
    return sum(results)

def concat_reduce(results):
    """
    Joins multiple lists of items (e.g. lists of primes) into a single list.
    """
    final_list = []
    for r in results:
        if isinstance(r, list):
            final_list.extend(r)
        else:
            final_list.append(r)
    return final_list

def average_reduce(results):
    """
    Averages multiple numerical results (e.g. Monte Carlo Pi estimates).
    """
    if not results:
        return 0.0
    return sum(results) / len(results)

def dict_merge_reduce(results):
    """
    Merges multiple word frequency dictionaries by summing frequencies of identical keys.
    """
    merged = {}
    for r in results:
        if isinstance(r, dict):
            for word, count in r.items():
                merged[word] = merged.get(word, 0) + count
    # Return sorted by frequency descending
    return dict(sorted(merged.items(), key=lambda item: item[1], reverse=True))

def matrix_stack_reduce(results):
    """
    Stacks segments of a matrix vertically in order of their chunks.
    Results should be sorted by chunk index or supplied in correct order.
    """
    final_matrix = []
    for chunk_res in results:
        if isinstance(chunk_res, list):
            final_matrix.extend(chunk_res)
    return final_matrix
