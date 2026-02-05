
import numpy as np
from pathlib import Path

def main():
    results_path = Path("viz_results.npy")
    if not results_path.exists():
        print("Results file not found.")
        return

    results = np.load(results_path, allow_pickle=True).item()
    
    maes = []
    coverages = []
    
    print(f"{'Subject':<10} | {'MAE':<6} | {'Cov(90%)':<8}")
    print("-" * 30)
    
    for subj, data in results.items():
        preds = data['preds']
        actuals = data['actuals']
        lower = data['lower']
        upper = data['upper']
        
        # Calculate MAE (Smoothed is already in preds if train.py saved it? 
        # Wait, train.py saves 'preds' which are the outputs returned by evaluate.
        # In the modified train.py, evaluate returns 'preds_smooth'. So yes.)
        
        mae = np.mean(np.abs(preds - actuals))
        
        # Coverage
        covered = ((actuals >= lower) & (actuals <= upper)).mean()
        
        maes.append(mae)
        coverages.append(covered)
        
        print(f"{subj:<10} | {mae:.2f}   | {covered*100:.1f}%")
        
    print("-" * 30)
    print(f"Average MAE: {np.mean(maes):.2f} BPM")
    print(f"Average Coverage: {np.mean(coverages)*100:.1f}%")

if __name__ == "__main__":
    main()
