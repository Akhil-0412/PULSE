
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    results_path = Path("cnn_results.npy")
    if not results_path.exists():
        print("Results file cnn_results.npy not found. Run train_cnn.py first.")
        return

    # Load results (allow_pickle=True needed for dictionary)
    results = np.load(results_path, allow_pickle=True).item()
    
    # Plotting
    # Create output directory
    out_dir = Path("artifacts")
    out_dir.mkdir(exist_ok=True)
    
    for subj, data in results.items():
        preds = data['preds']
        actuals = data['actuals']
        lower = data['lower']
        upper = data['upper']
        
        # Limit frames for clarity (e.g., first 100 windows ~ 4 mins with overlap)
        n = min(100, len(preds))
        x = np.arange(n)
        
        plt.figure(figsize=(12, 6))
        
        # Ground Truth
        plt.plot(x, actuals[:n], 'k-', label='Ground Truth', linewidth=2)
        
        # Prediction
        plt.plot(x, preds[:n], 'r--', label='TransPPG Prediction', linewidth=2)
        
        # Confidence Interval
        plt.fill_between(x, lower[:n], upper[:n], color='red', alpha=0.2, label='90% Conformal Interval')
        
        plt.title(f"Heart Rate Estimation: {subj} (Real Data)", fontsize=16)
        plt.xlabel("Time Windows (Overlap)", fontsize=12)
        plt.ylabel("Heart Rate (BPM)", fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        out_path = out_dir / f"viz_{subj}.png"
        plt.savefig(out_path)
        print(f"Saved plot to {out_path.absolute()}")
        plt.close()

if __name__ == "__main__":
    main()
