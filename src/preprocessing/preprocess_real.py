
import wfdb
import numpy as np
import os
from pathlib import Path
from scipy.signal import resample

DATA_DIR = Path("data/physionet")
OUTPUT_DIR = Path("data/real_dataset")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SEC = 8
STRIDE_SEC = 2 # 2s stride (overlapping)
FS = 500 # Sampling Rate
SAMPLES_PER_WINDOW = WINDOW_SEC * FS

def process_subject(subj_id):
    subj_dir = OUTPUT_DIR / f"S{subj_id}"
    subj_dir.mkdir(exist_ok=True)
    
    all_data = []
    all_labels = []
    
    activities = ["sit", "walk", "run"]
    
    for act in activities:
        record_name = f"s{subj_id}_{act}"
        record_path = DATA_DIR / record_name
        
        # Check if files exist
        if not (DATA_DIR / f"{record_name}.dat").exists():
            print(f"Skipping {record_name} (not found)")
            continue
            
        print(f"Processing {record_name}...")
        
        try:
            # Read Record
            record = wfdb.rdrecord(str(record_path))
            signals = record.p_signal
            sig_names = record.sig_name
            
            # Identify indices
            # Need: PPG (pleth_1..6) and Accel (a_x, a_y, a_z)
            # Strategy: Take best PPG (highest std dev? or just pleth_1 for now?)
            # Thesis said "Best PPG". We will just use pleth_1 for simplicity in V1.
            try:
                ppg_idx = sig_names.index('pleth_1')
                acc_x_idx = sig_names.index('a_x')
                acc_y_idx = sig_names.index('a_y')
                acc_z_idx = sig_names.index('a_z')
            except ValueError as e:
                print(f"Missing channel in {record_name}: {e}")
                continue

            # Read Annotations (R-peaks)
            ann = wfdb.rdann(str(record_path), 'atr')
            r_peaks = ann.sample # Indices of R-peaks
            
            # Segment
            num_samples = len(signals)
            step = int(STRIDE_SEC * FS)
            
            for start in range(0, num_samples - SAMPLES_PER_WINDOW, step):
                end = start + SAMPLES_PER_WINDOW
                
                # Extract Window
                ppg = signals[start:end, ppg_idx]
                acc_x = signals[start:end, acc_x_idx]
                acc_y = signals[start:end, acc_y_idx]
                acc_z = signals[start:end, acc_z_idx]
                
                # Check NaNs
                if np.isnan(ppg).any() or np.isnan(acc_x).any():
                    continue
                
                # Calculate Label (HR)
                # Find R-peaks in this window
                peaks_in_window = r_peaks[(r_peaks >= start) & (r_peaks < end)]
                if len(peaks_in_window) < 2:
                    continue # Cannot calc HR
                
                # HR = 60 / mean(RR interval in seconds)
                rr_intervals = np.diff(peaks_in_window) / FS
                hr = 60.0 / np.mean(rr_intervals)
                
                if hr < 40 or hr > 180:
                    continue # Outlier removal
                
                # Stack
                # dim: (4000, 4)
                window_data = np.stack([ppg, acc_x, acc_y, acc_z], axis=1)
                
                all_data.append(window_data)
                all_labels.append(hr)
                
        except Exception as e:
            print(f"Error processing {record_name}: {e}")
            
    if all_data:
        np.save(subj_dir / "data.npy", np.array(all_data).astype(np.float32))
        np.save(subj_dir / "labels.npy", np.array(all_labels).astype(np.float32))
        print(f"Saved Subject {subj_id}: {len(all_data)} windows")
    else:
        print(f"No valid data for Subject {subj_id}")

def main():
    # Process S1 to S22
    for s in range(1, 23):
        process_subject(s)

if __name__ == "__main__":
    main()
