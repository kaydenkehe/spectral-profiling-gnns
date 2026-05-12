import json
from pathlib import Path

def load_existing_slp(metrics_path, dataset_name):
    """Load SLP data for a specific dataset from metrics.json."""
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    if dataset_name not in metrics:
        return None
    
    data = metrics[dataset_name]
    eigenvalues = np.array(data['eigenvalues'])
    cdf = np.array(data['cdf'])
    
    # Convert to probability mass
    mass = np.diff(np.concatenate([[0], cdf]))
    mass = mass / mass.sum()  # Normalize
    
    return eigenvalues, mass, cdf

def get_available_datasets(metrics_path):
    """Get list of datasets with existing SLP data."""
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    return list(metrics.keys())

if __name__ == "__main__":
    metrics_path = Path(__file__).parent / "metrics.json"
    datasets = get_available_datasets(metrics_path)
    print("Available datasets with SLP data:")
    for ds in datasets:
        print(f"  - {ds}")
