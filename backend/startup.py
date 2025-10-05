"""
Startup script to initialize ML models in background
"""

import threading
import time
from ml_predictor import ml_predictor

_training_started = False

def train_models_background():
    """Train ML models in background thread"""
    global _training_started
    if _training_started:
        return
    
    _training_started = True
    try:
        print("Starting background ML model training...")
        X_train, y_train = ml_predictor.generate_synthetic_training_data(1000)  # Smaller dataset for faster training
        ml_predictor.train_models(X_train, y_train)
        print("Background ML model training completed!")
    except Exception as e:
        print(f"Background training failed: {e}")
        _training_started = False

def initialize_system():
    """Initialize system components"""
    global _training_started
    
    if not _training_started:
        # Start ML training in background
        training_thread = threading.Thread(target=train_models_background, daemon=True)
        training_thread.start()
        
        print("System initialization started...")
        print("ML models will be available shortly...")
    else:
        print("System already initialized...")

if __name__ == "__main__":
    initialize_system()