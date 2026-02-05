
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def main():
    results_path = Path("cnn_results.npy")
    if not results_path.exists():
        print("Results file cnn_results.npy not found.")
        return

    results = np.load(results_path, allow_pickle=True).item()
    
    data = []
    
    for subj, res in results.items():
        preds = res['preds']
        actuals = res['actuals']
        lower = res['lower']
        upper = res['upper']
        width = res['upper'] - res['lower']
        
        mae = np.mean(np.abs(preds - actuals))
        coverage = ((actuals >= lower) & (actuals <= upper)).mean() * 100
        mean_width = np.mean(width)
        
        data.append({
            'Subject': subj,
            'MAE': mae,
            'Coverage': coverage,
            'Width': mean_width
        })
        
    df = pd.DataFrame(data)
    
    # Sort by MAE (Best to Worst)
    df_sorted = df.sort_values(by='MAE', ascending=True)
    
    print("\n--- Performance Ranking (Best to Worst) ---")
    print(df_sorted.to_string(index=False, float_format="%.2f"))
    
    # Visualization
    plt.figure(figsize=(14, 8))
    
    # Color coding: Green (<5), Yellow (5-10), Red (>10)
    colors = []
    for mae in df_sorted['MAE']:
        if mae < 5: colors.append('green')
        elif mae < 10: colors.append('orange')
        else: colors.append('red')
        
    bars = plt.bar(df_sorted['Subject'], df_sorted['MAE'], color=colors)
    
    plt.axhline(y=5, color='gray', linestyle='--', alpha=0.7, label='Target (5 BPM)')
    
    plt.title("ResNet-1D Performance by Subject (Ranked Best to Worst)", fontsize=16)
    plt.xlabel("Subject", fontsize=12)
    plt.ylabel("Mean Absolute Error (BPM)", fontsize=12)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Add values on top
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom')
                
    out_path = Path("artifacts/mae_ranking.png")
    out_path.parent.mkdir(exist_ok=True)
    plt.savefig(out_path)
    print(f"\nSaved ranking plot to {out_path.absolute()}")

if __name__ == "__main__":
    main()
