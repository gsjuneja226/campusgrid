import os
from PIL import Image, ImageFilter

def run(chunk_index, total_chunks, progress_callback=None):
    # Locate files and directories
    input_path = os.path.join(os.getcwd(), "jobs", "portrait.png")
    results_dir = os.path.join(os.getcwd(), "campusgrid", "results")
    
    if not os.path.exists(input_path):
        return f"Error: Source image not found at {input_path}"
        
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)
        
    # 1. Load the real-life portrait image
    img = Image.open(input_path)
    
    # 2. Assign different blur levels to different chunks
    # Chunk 0 -> light blur (radius 2)
    # Chunk 1 -> medium blur (radius 6)
    # Chunk 2 -> heavy blur (radius 12)
    # Chunk 3 -> extreme blur (radius 20)
    blur_radii = [2, 6, 12, 20]
    radius = blur_radii[chunk_index % len(blur_radii)]
    
    print(f"Worker {chunk_index + 1}: Applying Gaussian Blur filter (Radius: {radius})...")
    
    if progress_callback:
        progress_callback(25)
        
    # 3. Apply Gaussian Blur filter
    processed_img = img.filter(ImageFilter.GaussianBlur(radius))
    
    if progress_callback:
        progress_callback(75)
        
    # 4. Save the output image
    filename = f"blurred_portrait_chunk_{chunk_index}_radius_{radius}.png"
    output_path = os.path.join(results_dir, filename)
    processed_img.save(output_path)
    
    if progress_callback:
        progress_callback(100)
        
    # Generate the access URL served by the Master Flask application
    url = f"http://localhost:5000/api/results/{filename}"
    return f"Chunk {chunk_index} (Radius {radius}) completed -> [Download/View Image]({url})"
