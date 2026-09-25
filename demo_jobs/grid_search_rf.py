import time
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

def run(chunk_index, total_chunks, progress_callback=None):
    # 1. Define hyperparameter search combinations
    n_estimators_options = [10, 20, 50, 100]
    max_depth_options = [5, 10, 15, None]
    
    grid = []
    for est in n_estimators_options:
        for depth in max_depth_options:
            grid.append({"n_estimators": est, "max_depth": depth})
            
    # 2. Slice the grid based on chunk_index and total_chunks
    grid_size = len(grid)
    chunk_size = grid_size // total_chunks
    remainder = grid_size % total_chunks
    
    start_idx = chunk_index * chunk_size + min(chunk_index, remainder)
    end_idx = start_idx + chunk_size + (1 if chunk_index < remainder else 0)
    
    my_configurations = grid[start_idx:end_idx]
    
    # 3. Create dummy dataset
    X, y = make_classification(n_samples=3000, n_features=25, random_state=42)
    
    results = []
    print(f"Worker {chunk_index + 1}: Testing combinations {start_idx} to {end_idx - 1}")
    
    for idx, config in enumerate(my_configurations):
        # Create and fit the model
        clf = RandomForestClassifier(
            n_estimators=config["n_estimators"], 
            max_depth=config["max_depth"], 
            random_state=42, 
            n_jobs=-1
        )
        clf.fit(X, y)
        score = clf.score(X, y)
        
        results.append({
            "config": config,
            "accuracy": round(score, 4)
        })
        
        # Report progress to dashboard
        if progress_callback:
            progress_callback(int(((idx + 1) / len(my_configurations)) * 100))
            
    # Find the best hyperparameter match in this worker's chunk
    best_match = max(results, key=lambda x: x["accuracy"])
    return f"Tested {len(my_configurations)} configs. Best chunk result: {best_match['config']} -> Accuracy: {best_match['accuracy']}"
