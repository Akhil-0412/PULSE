
import wfdb
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import butter, filtfilt, resample

# --- CONFIGURATION ---
DATA_DIR = Path("data/physionet")
OUTPUT_DIR = Path("data/experiment_dataset_16s")
TARGET_FS = 100  # Downsample to 100Hz
WINDOW_SEC = 16  # 16 seconds
WINDOW_SIZE = TARGET_FS * WINDOW_SEC # 1600 samples
STEP_SEC = 4     # 4 seconds shift (75% overlap)
STEP_SIZE = TARGET_FS * STEP_SEC

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def process_subject(subj_id):
    """
    Process all records (sit, walk, run) for a single subject.
    """
    subj_dir = OUTPUT_DIR / f"S{subj_id}"
    subj_dir.mkdir(exist_ok=True, parents=True) # Ensure S1 exists
    
    activities = ["sit", "walk", "run"]
    all_windows = []
    all_labels = []
    
    for act in activities:
        record_name = f"s{subj_id}_{act}"
        record_path = DATA_DIR / record_name
        
        # Check if files exist
        if not (DATA_DIR / f"{record_name}.dat").exists():
            print(f"Skipping {record_name} (not found)")
            continue
            
        try:
            # Read header first to get fs
            # wfdb.rdrecord works with the base name (no extension)
            signals, fields = wfdb.rdsamp(str(record_path))
            fs = fields['fs']
            sig_names = fields['sig_name']
            
            # Identify indices
            try:
                ppg_idx = sig_names.index('pleth_1')
                # Try to find acceleration channels
                if 'a_x' in sig_names:
                     acc_indices = [sig_names.index('a_x'), sig_names.index('a_y'), sig_names.index('a_z')]
                elif 'acc_x' in sig_names:
                     acc_indices = [sig_names.index('acc_x'), sig_names.index('acc_y'), sig_names.index('acc_z')]
                else:
                    # Fallback: assume 1, 2, 3 are acc
                    acc_indices = [1, 2, 3] 
            except ValueError as e:
                # If pleth_1 not found, skip
                print(f"  Missing matching channels in {record_name}: {e}")
                continue
                
            # 1. Bandpass Filter PPG
            ppg = signals[:, ppg_idx]
            ppg_filtered = butter_bandpass_filter(ppg, 0.5, 8.0, fs, order=4)
            
            # 2. Resample to 100Hz
            num_samples = len(ppg)
            new_num_samples = int(num_samples * TARGET_FS / fs)
            
            # Resample specific channels (PPG + Accel)
            # Create a matrix of just the 4 channels we want
            raw_matrix = signals[:, [ppg_idx] + acc_indices]
            resampled_data = resample(raw_matrix, new_num_samples)
            
            # Replace the resampled PPG with the FILTERED + RESAMPLED PPG
            # (Filtering first is better)
            ppg_resampled = resample(ppg_filtered, new_num_samples)
            resampled_data[:, 0] = ppg_resampled
            
            # 3. Z-Score Normalize
            mean = np.mean(resampled_data, axis=0)
            std = np.std(resampled_data, axis=0)
            resampled_data = (resampled_data - mean) / (std + 1e-6)
            
            # 4. Load Labels (HR)
            ann = wfdb.rdann(str(record_path), 'atr')
            peaks = ann.sample
            peaks_resampled = (peaks * TARGET_FS / fs).astype(int)
            
            # 5. Windowing
            for start in range(0, new_num_samples - WINDOW_SIZE, STEP_SIZE):
                end = start + WINDOW_SIZE
                window_data = resampled_data[start:end, :] # (1600, 4)
                
                win_peaks = peaks_resampled[(peaks_resampled >= start) & (peaks_resampled < end)]
                
                if len(win_peaks) < 2:
                    continue
                    
                rr_intervals = np.diff(win_peaks) / TARGET_FS
                if np.any(rr_intervals == 0): continue
                hr_window = 60.0 / np.mean(rr_intervals)
                
                if hr_window < 40 or hr_window > 200:
                    continue
                
                all_windows.append(window_data)
                all_labels.append(hr_window)
                
        except Exception as e:
            print(f"  Error reading {record_name}: {e}")
            continue

    if all_windows:
        np.save(subj_dir / "data.npy", np.array(all_windows, dtype=np.float32))
        np.save(subj_dir / "labels.npy", np.array(all_labels, dtype=np.float32))
        print(f"Saved Subject {subj_id}: {len(all_windows)} windows")
    else:
        print(f"No valid data for Subject {subj_id}")

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing subjects in {DATA_DIR}")
    
    for s in range(1, 23):
        process_subject(s)

if __name__ == "__main__":
    main()
