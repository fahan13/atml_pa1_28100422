import json
import numpy as np
from pathlib import Path

class NumpyEncoder(json.JSONEncoder):
    """Teaches json.dump how to handle NumPy types automatically,
    so we stop hunting for missing float()/tolist() calls one by one."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return super().default(obj)

def save_results(results_dict, filepath):
    """Save a results dict as readable JSON, creating parent folders if needed."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(results_dict, f, indent=2, cls=NumpyEncoder)

def load_results(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)