import json
from pathlib import Path

def save_results(results_dict, filepath):
    """Save a results dict as readable JSON, creating parent folders if needed."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(results_dict, f, indent=2)

def load_results(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)