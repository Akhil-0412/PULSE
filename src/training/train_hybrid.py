
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from model_hybrid import HybridCNNLSTM, AttentionCNNLSTM

# --- CONFIG ---
DATA_DIR = Path("data/experiment_dataset_16s")
BATCH_SIZE = 64  # Smaller than CNN due to LSTM memory
EPOCHS = 50
LR = 1e-4
GRAD_CLIP = 1.0  # Prevent exploding gradients in RNN

# --- DATASET ---
class PPGDataset(Dataset):
    def __init__(self, data_list, label_list):
        self.data_list = data_list
        self.label_list = label_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]  # (1600, 4)
        label = self.label_list[idx]
        
        # Transpose to (Channels, Length) -> (4, 1600)
        data = torch.tensor(data, dtype=torch.float32).permute(1, 0)
        label = torch.tensor(label, dtype=torch.float32)
        return data, label


# --- EVALUATION ---
def evaluate(model, loader, device):
    model.eval()
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
    """Calculate conformal prediction quantile from calibration set."""
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
    subjects = [s for s in subjects if (DATA_DIR / s / "data.npy").exists()]
    
    print(f"Found {len(subjects)} subjects: {subjects}")
    print(f"Training Hybrid CNN-LSTM with {EPOCHS} epochs, batch size {BATCH_SIZE}")
    print("=" * 60)

    results = {}
    aggregate_mae = []
    aggregate_coverage = []
    
    # Use AMP for faster training on GPU
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    print(f"{'Test Subj':<10} | {'MAE':<6} | {'Cov(90%)':<8} | {'Width':<6}")
    print("-" * 50)

    for i, test_subj in enumerate(subjects):
        # Calibration Subject (previous one in the list)
        cal_idx = (i - 1) % len(subjects)
        if cal_idx == i:
            cal_idx = (i + 1) % len(subjects)
        cal_subj = subjects[cal_idx]
        
        # Load Data
        train_data, train_labels = [], []
        cal_data, cal_labels = [], []
        test_data, test_labels = [], []
        
        for subj in subjects:
            d = np.load(DATA_DIR / subj / "data.npy")  # (N, 1600, 4)
            l = np.load(DATA_DIR / subj / "labels.npy")
            
            if subj == test_subj:
                test_data.extend(d)
                test_labels.extend(l)
            elif subj == cal_subj:
                cal_data.extend(d)
                cal_labels.extend(l)
            else:
                train_data.extend(d)
                train_labels.extend(l)
        
        # Create DataLoaders
        train_loader = DataLoader(
            PPGDataset(train_data, train_labels), 
            batch_size=BATCH_SIZE, shuffle=True, 
            num_workers=0, pin_memory=True
        )
        cal_loader = DataLoader(
            PPGDataset(cal_data, cal_labels), 
            batch_size=BATCH_SIZE, shuffle=False, num_workers=0
        )
        test_loader = DataLoader(
            PPGDataset(test_data, test_labels), 
            batch_size=BATCH_SIZE, shuffle=False, num_workers=0
        )

        # Model - Using AttentionCNNLSTM for better performance
        model = AttentionCNNLSTM(input_channels=4).to(device)
        criterion = nn.L1Loss()
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
        
        # Learning Rate Scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        
        # Training Loop
        model.train()
        for epoch in range(EPOCHS):
            epoch_loss = 0.0
            for inputs, targets in train_loader:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                
                optimizer.zero_grad()
                
                if scaler is not None:
                    with torch.amp.autocast('cuda'):
                        outputs = model(inputs).view(-1)
                        loss = criterion(outputs, targets)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(inputs).view(-1)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    optimizer.step()
                    
                epoch_loss += loss.item()
            
            scheduler.step()
        
        # Conformal Calibration
        q_hat = get_conformal_quantile(model, cal_loader, device)
        
        # Evaluate
        mae, preds, actuals = evaluate(model, test_loader, device)
        
        # Coverage Calculation
        preds = np.array(preds)
        actuals = np.array(actuals)
        lower = preds - q_hat
        upper = preds + q_hat
        covered = ((actuals >= lower) & (actuals <= upper)).mean()
        width = q_hat * 2
        
        aggregate_mae.append(mae)
        aggregate_coverage.append(covered)
        
        print(f"{test_subj:<10} | {mae:.2f}   | {covered*100:.1f}%     | {width:.2f}")
        
        results[test_subj] = {
            'preds': preds, 
            'actuals': actuals, 
            'lower': lower, 
            'upper': upper,
            'mae': mae,
            'coverage': covered
        }
        
    print("-" * 50)
    print(f"Average MAE: {np.mean(aggregate_mae):.2f} BPM")
    print(f"Average Coverage: {np.mean(aggregate_coverage)*100:.1f}%")
    
    # Save Results
    np.save("hybrid_results.npy", results)
    print("\nResults saved to hybrid_results.npy")


if __name__ == "__main__":
    main()
