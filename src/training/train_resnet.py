
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import random
from model_cnn import ResNet1D

# --- CONFIG ---
DATA_DIR = Path("data/experiment_dataset_16s")
BATCH_SIZE = 128
EPOCHS = 30
LR = 1e-4

# --- DATASET ---
class PPGDataset(Dataset):
    def __init__(self, data_list, label_list, from_memory=False):
        self.data_list = data_list
        self.label_list = label_list
        self.from_memory = from_memory

    def __len__(self):
        if self.from_memory:
            return len(self.data_list)
        return len(self.data_list) 

    def __getitem__(self, idx):
        if self.from_memory:
            data = self.data_list[idx] # (1600, 4)
            label = self.label_list[idx]
        else:
            # Lazy loading logic if paths were passed (omitted for now as we fit in RAM)
            pass
            
        # Transpose to (Channels, Length) -> (4, 1600)
        data = torch.tensor(data, dtype=torch.float32).permute(1, 0)
        label = torch.tensor(label, dtype=torch.float32)
        return data, label

# --- EVALUATION ---
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_error = 0.0
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs).view(-1)
            
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(targets.cpu().numpy())
            total_error += torch.abs(outputs - targets).sum().item()
    
    mae = total_error / len(loader.dataset)
    return mae, predictions, actuals

def get_conformal_quantile(model, loader, device, alpha=0.1):
    model.eval()
    scores = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs).view(-1)
            residuals = torch.abs(targets - outputs)
            scores.extend(residuals.cpu().numpy())
            
    n = len(scores)
    q_val = np.ceil((n + 1) * (1 - alpha)) / n
    q_val = min(1.0, q_val)
    q_hat = np.quantile(scores, q_val, method='higher')
    return q_hat

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        
    if not DATA_DIR.exists():
        print(f"Data directory {DATA_DIR} not found. Run preprocess_experiment.py first.")
        return

    subjects = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir()])
    # Filter only those with data
    subjects = [s for s in subjects if (DATA_DIR / s / "data.npy").exists()]
    
    print(f"Found subjects: {subjects}")

    cnn_results = {}
    aggregate_mae = []
    aggregate_coverage = []
    
    # scaler for AMP (Mixed Precision)
    scaler = torch.amp.GradScaler('cuda')

    print(f"{'Test Subj':<10} | {'MAE':<6} | {'Cov(90%)':<8} | {'Width':<6}")
    print("-" * 45)

    for i, test_subj in enumerate(subjects):
        # Calibration Subject
        cal_idx = (i - 1) % len(subjects)
        if cal_idx == i: cal_idx = (i+1)%len(subjects) # Handle 1 subj case if needed
        cal_subj = subjects[cal_idx]
        
        # Load Data
        train_data = [] # List of numpy arrays
        train_labels = []
        cal_data, cal_labels = [], []
        test_data, test_labels = [], []
        
        # Load all into memory (should fit 8GB RAM easily for this dataset size)
        for subj in subjects:
            d = np.load(DATA_DIR / subj / "data.npy") # (N, 1600, 4)
            l = np.load(DATA_DIR / subj / "labels.npy")
            
            if subj == test_subj:
                test_data.extend(d); test_labels.extend(l)
            elif subj == cal_subj:
                cal_data.extend(d); cal_labels.extend(l)
            else:
                train_data.extend(d); train_labels.extend(l)
        
        train_loader = DataLoader(PPGDataset(train_data, train_labels, True), 
                                batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
        # Recycled train loader if empty (for single subj test)
        if len(train_loader) == 0: train_loader = DataLoader(PPGDataset(cal_data, cal_labels, True), batch_size=BATCH_SIZE, shuffle=True)
            
        cal_loader = DataLoader(PPGDataset(cal_data, cal_labels, True), 
                                batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        test_loader = DataLoader(PPGDataset(test_data, test_labels, True), 
                                batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        # Model
        model = ResNet1D(input_channels=4).to(device)
        criterion = nn.L1Loss()
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
        
        # Training Check
        model.train()
        for epoch in range(EPOCHS):
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                
                optimizer.zero_grad()
                
                # Mixed Precision Context
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs).view(-1)
                    loss = criterion(outputs, targets)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        
        # Conformal Calib
        q_hat = get_conformal_quantile(model, cal_loader, device)
        
        # Evaluate
        mae, preds, actuals = evaluate(model, test_loader, criterion, device)
        
        # Coverage
        preds = np.array(preds)
        actuals = np.array(actuals)
        lower = preds - q_hat
        upper = preds + q_hat
        covered = ((actuals >= lower) & (actuals <= upper)).mean()
        width = q_hat * 2
        
        aggregate_mae.append(mae)
        aggregate_coverage.append(covered)
        
        print(f"{test_subj:<10} | {mae:.2f}   | {covered*100:.1f}%     | {width:.2f}")
        
        cnn_results[test_subj] = {
            'preds': preds, 'actuals': actuals, 'lower': lower, 'upper': upper
        }
        
    print("-" * 45)
    print(f"Average MAE: {np.mean(aggregate_mae):.2f} BPM")
    
    np.save("cnn_results.npy", cnn_results)

if __name__ == "__main__":
    main()
