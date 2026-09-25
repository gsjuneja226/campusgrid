import os
import time
import numpy as np
from PIL import Image

def run(chunk_index, total_chunks, progress_callback=None):
    # Batch of 24 frames of 150x150 pixel images
    total_images = 24
    
    # Calculate partition
    chunk_size = total_images // total_chunks
    remainder = total_images % total_chunks
    start_img = chunk_index * chunk_size + min(chunk_index, remainder)
    end_img = start_img + chunk_size + (1 if chunk_index < remainder else 0)
    
    # Locate results folder dynamically (shared on local machine runs)
    results_dir = os.path.join(os.getcwd(), "campusgrid", "results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)
        
    processed_log = []
    
    for idx, img_idx in enumerate(range(start_img, end_img)):
        # 1. Generate a synthetic pixel array (a circle pattern)
        x, y = np.ogrid[-75:75, -75:75]
        frame = (x**2 + y**2 < (img_idx * 5) ** 2).astype(np.float64) * 255.0
        
        # 2. Perform Box Blur convolution (5x5 kernel size)
        kernel_size = 5
        padded = np.pad(frame, pad_width=kernel_size//2, mode='edge')
        blurred = np.zeros_like(frame)
        
        for r in range(frame.shape[0]):
            for c in range(frame.shape[1]):
                blurred[r, c] = np.mean(padded[r:r+kernel_size, c:c+kernel_size])
                
        # 3. Use Pillow to save the matrix as a real PNG image file
        filename = f"render_frame_{img_idx:02d}.png"
        img_path = os.path.join(results_dir, filename)
        
        # Convert float matrix to uint8 and save image
        img_uint8 = blurred.astype(np.uint8)
        img = Image.fromarray(img_uint8)
        img.save(img_path)
        
        # Generate the access URL served by the Master Flask application
        url = f"http://localhost:5000/api/results/{filename}"
        processed_log.append(f"Frame {img_idx:02d} rendered -> [Download/View Frame]({url})")
        
        if progress_callback:
            progress_callback(int(((idx + 1) / (end_img - start_img)) * 100))
            
    return "\n".join(processed_log)
