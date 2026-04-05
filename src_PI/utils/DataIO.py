import os
import json
from datetime import datetime

def save_sweep_data(L, dim, params, sweep_results):
    """
    Saves the results of an EFT resource estimation sweep to a timestamped JSON file.
    
    Args:
        L (int): Lattice side length.
        dim (int): Dimensionality.
        params (dict): The physical parameters used for the sweep.
        sweep_results (list): A list of dictionaries containing the data for each A.
    """
    # 1. Create Date-Based Directory structure (e.g., data/2026-04-04/)
    current_date = datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.now().strftime("%H%M%S")
    
    # Assuming this is run from the root of the project
    target_dir = os.path.join("data", current_date)
    os.makedirs(target_dir, exist_ok=True)
    
    # 2. Define Unique Filename
    filename = f"sweep_L{L}_{dim}D_{current_time}.json"
    filepath = os.path.join(target_dir, filename)
    
    # 3. Assemble Data Package
    data_package = {
        "metadata": {
            "L": L,
            "dim": dim,
            "params": params,
            "timestamp": current_time,
            "date": current_date
        },
        "results": sweep_results
    }
    
    # 4. Write to JSON
    with open(filepath, 'w') as f:
        json.dump(data_package, f, indent=4)
        
    print(f"\n[DataIO] Successfully saved sweep data to: {filepath}")
    return filepath