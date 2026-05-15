import numpy as np
import json
import os

# Paths
RESNET_PATH = 'results/resnet1d/cnn_results.npy'
HYBRID_PATH = 'results/hybrid/hybrid_results.npy'
OUTPUT_PATH = 'public/results.json'

# Target ~100 data points for clean visualization
TARGET_POINTS = 100

def load_data():
    results = {}
    
    if os.path.exists(RESNET_PATH):
        print(f"Loading {RESNET_PATH}...")
        results['resnet'] = np.load(RESNET_PATH, allow_pickle=True).item()
    
    if os.path.exists(HYBRID_PATH):
        print(f"Loading {HYBRID_PATH}...")
        results['hybrid'] = np.load(HYBRID_PATH, allow_pickle=True).item()
        
    return results

def subsample(arr, target_len):
    """Subsample array to target length by taking evenly spaced indices"""
    if len(arr) <= target_len:
        return arr
    indices = np.linspace(0, len(arr) - 1, target_len, dtype=int)
    return [arr[i] for i in indices]

def convert_to_json(data):
    subjects = set()
    if 'resnet' in data: subjects.update(data['resnet'].keys())
    if 'hybrid' in data: subjects.update(data['hybrid'].keys())

    def clean(arr):
        return [float(x) if isinstance(x, (np.float32, np.float64)) else x for x in arr]

    final_data = {
        'subjects': sorted(list(subjects)),
        'data': {}
    }

    for subj in final_data['subjects']:
        final_data['data'][subj] = {}
        
        for model in ['resnet', 'hybrid']:
            if model in data and subj in data[model]:
                d = data[model][subj]
                
                # Subsample all arrays to TARGET_POINTS
                actuals = subsample(list(d['actuals']), TARGET_POINTS)
                preds = subsample(list(d['preds']), TARGET_POINTS)
                lower = subsample(list(d.get('lower', d['preds'])), TARGET_POINTS)
                upper = subsample(list(d.get('upper', d['preds'])), TARGET_POINTS)
                
                final_data['data'][subj][model] = {
                    'actuals': clean(actuals),
                    'preds': clean(preds),
                    'lower': clean(lower),
                    'upper': clean(upper),
                    'mae': float(np.mean(np.abs(np.array(d['preds']) - np.array(d['actuals'])))),
                    'coverage': float(d.get('coverage', 0))
                }
    
    return final_data

if __name__ == "__main__":
    if not os.path.exists('public'):
        os.makedirs('public')
        
    raw_data = load_data()
    json_data = convert_to_json(raw_data)
    
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(json_data, f)
    
    print(f"Exported to {OUTPUT_PATH}")
    print(f"S1 hybrid actuals count: {len(json_data['data']['S1']['hybrid']['actuals'])}")
