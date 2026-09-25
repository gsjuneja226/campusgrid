import time
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

def run(chunk_index, total_chunks, progress_callback=None):
    # 1. Generate a simulated global classification dataset
    # We use a fixed random state so all nodes can reference the same dataset structure,
    # but partition it so each worker only sees and trains on its own chunk.
    n_samples = 4000
    X, y = make_classification(n_samples=n_samples, n_features=10, random_state=42)
    
    # Calculate partition indices for this worker
    chunk_size = n_samples // total_chunks
    remainder = n_samples % total_chunks
    start_idx = chunk_index * chunk_size + min(chunk_index, remainder)
    end_idx = start_idx + chunk_size + (1 if chunk_index < remainder else 0)
    
    # Slice the training data for this specific worker
    X_train = X[start_idx:end_idx]
    y_train = y[start_idx:end_idx]
    
    # Hold out a test set to check how well the local model generalizes
    X_test = X[:400]
    y_test = y[:400]
    
    print(f"Worker {chunk_index + 1}: Loading private partition from sample {start_idx} to {end_idx - 1} ({len(X_train)} samples).")
    
    if progress_callback:
        progress_callback(20)
        
    # 2. Initialize the Linear Model (represents a classification layer)
    # Using SGDClassifier to perform mini-batch gradient descent
    model = SGDClassifier(loss="log_loss", max_iter=1000, random_state=42)
    
    # 3. Train the model locally on worker's CPU
    model.fit(X_train, y_train)
    
    if progress_callback:
        progress_callback(65)
        
    # 4. Evaluate local performance
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Extract learned weights (coefficients) and bias (intercept)
    coef = model.coef_[0]
    intercept = model.intercept_[0]
    
    if progress_callback:
        progress_callback(100)
        
    # Shorten weights array to make output readable in the UI
    weights_summary = ", ".join([f"{w:.4f}" for w in coef[:5]]) + "..."
    
    # Return metrics and weights back to Master for averaging/aggregation
    return (
        f"Worker Node {chunk_index + 1} Status:\n"
        f"  - Dataset slice: samples [{start_idx} to {end_idx - 1}] ({len(X_train)} samples)\n"
        f"  - Evaluated Local Accuracy: {accuracy * 100:.2f}%\n"
        f"  - Extracted Bias (Intercept): {intercept:.6f}\n"
        f"  - Extracted Weights (First 5): [{weights_summary}]"
    )
