import os

if not os.path.exists('models/xgb_model.pkl'):
    print("Models not found, training now...")
    from src.model import train
    train()
    print("Training complete.")
else:
    print("Models found, skipping training.")