
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from pathlib import Path
from model import TransPPG

# --- Data Loader ---
class PPGDataset(Dataset):
    def __init__(self, data_sources, label_sources, from_memory=False):
        self.data = []
        self.labels = []
        
        if from_memory:
             # Expect list of numpy arrays
            self.data = np.concatenate(data_sources, axis=0)
            self.labels = np.concatenate(label_sources, axis=0).astype(np.float32)
        else:
            # Expect list of paths
            for d_path, l_path in zip(data_sources, label_sources):
                d = np.load(d_path)
                l = np.load(l_path)
                self.data.append(d)
                self.labels.append(l)
            self.data = np.concatenate(self.data, axis=0) # (TotalSamples, 4000, 4)
            self.labels = np.concatenate(self.labels, axis=0).astype(np.float32)
        
        # Normalize data (Z-score)
        mean = np.mean(self.data, axis=(0, 1), keepdims=True)
        std = np.std(self.data, axis=(0, 1), keepdims=True)
        self.data = (self.data - mean) / (std + 1e-6)
        
        # Normalize labels (0-200 BPM range)
        self.labels = self.labels / 200.0

        # Transpose to (Channels, Length) for PyTorch Conv1d
        self.data = self.data.transpose(0, 2, 1).astype(np.float32) # (N, 4, 4000)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx]), torch.tensor(self.labels[idx])

def get_loso_splits(data_dir):
    data_dir = Path(data_dir)
    subjects = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    splits = []
    
    for test_subj in subjects:
        train_data = []
        train_labels = []
        test_data_path = []
        test_label_path = []
        
        for subj in subjects:
            d_path = data_dir / subj / "data.npy"
            l_path = data_dir / subj / "labels.npy"
            
            if subj == test_subj:
                test_data_path.append(d_path)
                test_label_path.append(l_path)
            else:
                train_data.append(d_path)
                train_labels.append(l_path)
        
        splits.append({
            'test_subj': test_subj,
            'train': (train_data, train_labels),
            'test': (test_data_path, test_label_path)
        })
    return splits

# --- Training Loop ---
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device).unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_error = 0
    predictions = []
    actuals = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs).view(-1)
             
            # Rescale back to BPM
            outputs = outputs * 200.0
            targets = targets * 200.0
            
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(targets.cpu().numpy())
            total_error += torch.abs(outputs - targets).sum().item()
    
    mae = total_error / len(loader.dataset)
    
    # --- Post-Processing: Moving Average Smoothing ---
    # HR cannot change instantly. Smoothing removes outlier jumps.
    window_size = 5
    preds_smooth = np.convolve(predictions, np.ones(window_size)/window_size, mode='same')
    
    # Recalculate MAE on smoothed
    mae_smooth = np.mean(np.abs(preds_smooth - actuals))
    
    return mae_smooth, preds_smooth, actuals

def get_conformal_quantile(model, loader, device, alpha=0.1):
    model.eval()
    scores = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs).view(-1) # Safer than squeeze()
            
            # Rescale
            outputs = outputs * 200.0
            targets = targets * 200.0
            
            # Score = |y - y_hat|
            residuals = torch.abs(targets - outputs)
            scores.extend(residuals.cpu().numpy())
    
    n = len(scores)
    q_val = np.quantile(scores, np.ceil((n + 1) * (1 - alpha)) / n)
    return q_val

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Reload splits manually to handle Calibration split
    data_dir = Path("data/real_dataset")
    if not data_dir.exists():
        print("Real dataset not found, falling back to mock.")
        data_dir = Path("data/mock_dataset")
        
    subjects = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    
    # Limit to first 2 subjects for Quick Viz demo
    # subjects = subjects[:2] 
    # print(f"Running Quick Demo on: {subjects}")
    
    aggregate_mae = []
    aggregate_coverage = []
    aggregate_width = []
    
    # Store results for visualization
    viz_results = {}
    
    print(f"{'Test Subj':<10} | {'MAE':<6} | {'Cov(90%)':<8} | {'Width':<6}")
    print("-" * 45)
    
    for i, test_subj in enumerate(subjects):
        # ... (Same Logic) ...
        # 1. Select Calibration Subject (simply the previous one in list, or next)
        # Strategy: Use subjects[i-1] as calibration (if i=0, use last)
        cal_idx = (i - 1) % len(subjects)
        if cal_idx == i: # Should not happen with >1 subjects
            # Hack for 1 subject case (if only S1 exists but split)
             cal_idx = i # Overfit if only 1 subj? No, we split S1.
             # If subjects=['S1_A', 'S1_B'], len=2.
             pass
        
        cal_subj = subjects[cal_idx]
        train_subjs = [s for s in subjects if s != test_subj and s != cal_subj]
        
        # Load Datasets
        train_data, train_labels = [], []
        cal_data, cal_labels = [], []
        test_data, test_labels = [], []
        
        for subj in subjects:
            d = np.load(data_dir / subj / "data.npy")
            l = np.load(data_dir / subj / "labels.npy")
            if subj == test_subj:
                test_data.append(d); test_labels.append(l)
            elif subj == cal_subj:
                cal_data.append(d); cal_labels.append(l)
            else:
                train_data.append(d); train_labels.append(l)
        
        # Helper to wrap Dataset
        def create_loader(d_list, l_list, shuffle=False):
            if not d_list: return None
            ds = PPGDataset(d_list, l_list, from_memory=True)
            return DataLoader(ds, batch_size=16, shuffle=shuffle)
            
        train_loader = create_loader(train_data, train_labels, shuffle=True)
        # Fallback if no train data (only 2 subjs total) -> Use Calibration as Train
        if not train_loader: 
            train_loader = create_loader(cal_data, cal_labels, shuffle=True)
            
        cal_loader = create_loader(cal_data, cal_labels, shuffle=False)
        test_loader = create_loader(test_data, test_labels, shuffle=False)
        
        # Model
        model = TransPPG(input_channels=4).to(device)
        criterion = nn.L1Loss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        
        # Train
        model.train()
        for epoch in range(30): # Full training epochs
            if not train_loader: break
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device).unsqueeze(1)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
        
        # Calibrate (Conformal)
        alpha = 0.1 # 90% Confidence
        q_hat = get_conformal_quantile(model, cal_loader, device, alpha=alpha)
        
        # Eval
        mae, preds, actuals = evaluate(model, test_loader, criterion, device)
        
        # Calculate Coverage
        preds = np.array(preds)
        actuals = np.array(actuals)
        lower = preds - q_hat
        upper = preds + q_hat
        covered = ((actuals >= lower) & (actuals <= upper)).mean()
        width = q_hat * 2
        
        aggregate_mae.append(mae)
        aggregate_coverage.append(covered)
        aggregate_width.append(width)
        
        print(f"{test_subj:<10} | {mae:.2f}   | {covered*100:.1f}%     | {width:.2f}")
        
        # Save for Viz
        viz_results[test_subj] = {
            'preds': preds,
            'actuals': actuals,
            'lower': lower,
            'upper': upper
        }

    print("-" * 45)
    print(f"Average MAE: {np.mean(aggregate_mae):.2f} BPM")
    print(f"Average Coverage (Target 90%): {np.mean(aggregate_coverage)*100:.1f}%")
    
    # Save Results
    np.save("viz_results.npy", viz_results)
    print("Verification Completed. Results saved to viz_results.npy")


if __name__ == "__main__":
    main()
