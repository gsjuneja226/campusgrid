"""
CampusGrid Prime Finder Job
Checks a range for prime numbers and reports progress.
"""
import time

def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    # Check odd numbers up to sqrt(n)
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

def run(start, end, progress_callback=None):
    """
    Finds all prime numbers in the range [start, end].
    Invokes progress_callback with percentage complete.
    """
    primes = []
    total_numbers = end - start + 1
    
    if total_numbers <= 0:
        return []
        
    last_report_time = time.time()
    
    for idx, num in enumerate(range(start, end + 1)):
        if is_prime(num):
            primes.append(num)
            
        # Limit progress callback rate to prevent flooding (approx every 100ms or 1% intervals)
        if progress_callback and (idx % max(1, total_numbers // 100) == 0 or idx == total_numbers - 1):
            current_time = time.time()
            if current_time - last_report_time >= 0.1 or idx == total_numbers - 1:
                pct = int((idx + 1) / total_numbers * 100)
                progress_callback(pct)
                last_report_time = current_time
                
    if progress_callback:
        progress_callback(100)
        
    return primes
