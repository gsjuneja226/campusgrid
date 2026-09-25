"""
CampusGrid Word Count Job
Counts occurrences of words in a chunk of text paragraphs.
"""
import re
import time

def run(paragraphs, progress_callback=None):
    """
    Counts word frequencies in a list of text paragraphs.
    """
    counts = {}
    total_paragraphs = len(paragraphs)
    
    if total_paragraphs == 0:
        return {}
        
    last_report_time = time.time()
    
    for idx, paragraph in enumerate(paragraphs):
        # Find all alphanumeric words, convert to lowercase
        words = re.findall(r'\b\w+\b', paragraph.lower())
        for word in words:
            counts[word] = counts.get(word, 0) + 1
            
        if progress_callback and (idx % max(1, total_paragraphs // 20) == 0 or idx == total_paragraphs - 1):
            current_time = time.time()
            if current_time - last_report_time >= 0.1 or idx == total_paragraphs - 1:
                pct = int((idx + 1) / total_paragraphs * 100)
                progress_callback(pct)
                last_report_time = current_time
                
    if progress_callback:
        progress_callback(100)
        
    return counts
