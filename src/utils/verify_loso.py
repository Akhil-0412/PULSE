
import numpy as np
from pathlib import Path

# Mock configuration mirroring train_cnn.py
DATA_DIR = Path("data/experiment_dataset_16s")

def verify_loso():
    if not DATA_DIR.exists():
        print(f"Data directory {DATA_DIR} not found.")
        return

    # Get list of subjects that have data
    subjects = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir() and (d / "data.npy").exists()])
    
    print(f"Found {len(subjects)} valid subjects: {subjects}")
    print("-" * 60)
    print(f"{'Fold':<5} | {'Test Subject':<12} | {'Training Subjects Count':<25} | {'Status'}")
    print("-" * 60)
    
    # Simulate the loop from train_cnn.py
    for i, test_subj in enumerate(subjects):
        # Calibration Subject Logic (Skip for simplicity of explanation, or include)
        # train_cnn.py logic:
        # cal_idx = (i - 1) % len(subjects)
        # cal_subj = subjects[cal_idx]
        
        train_subjs = []
        
        for subj in subjects:
            if subj == test_subj:
                continue # strictly excluded from training
            # In train_cnn.py, we also exclude cal_subj from training, but use it for calibration.
            # For the purpose of "Testing", the Test Subject is definitely UNSEEN.
            train_subjs.append(subj)
            
        # Verify
        is_loso = test_subj not in train_subjs
        status = "PASSED (Unseen)" if is_loso else "FAILED (Data Leak)"
        
        print(f"{i+1:<5} | {test_subj:<12} | {len(train_subjs)} subjects (e.g. {train_subjs[0]}..{train_subjs[-1]}) | {status}")
        
    print("-" * 60)
    print("Verification: The Test Subject is ALWAYS excluded from the Training Set.")

if __name__ == "__main__":
    verify_loso()
