import os
import json
from datetime import datetime


def save_sweep_data(L, dim, params, sweep_results, config=None):
    """
    Saves the results of an EFT resource estimation sweep to a timestamped JSON file.

    Args:
        L (int): Lattice side length.
        dim (int): Dimensionality.
        params (dict): The physical parameters used for the sweep.
        sweep_results (list): A list of dictionaries containing the data for each A.
        config (Config | None): The pipeline config (basis, walk_mode, ...).
            Saved into metadata so the sweep file is self-describing about
            which design-axis choices were active. None tolerated for
            backward-compatibility with old call sites.
    """
    # 1. Create Date-Based Directory structure (e.g., data/2026-05-22/)
    current_date = datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.now().strftime("%H%M%S")

    target_dir = os.path.join("data", current_date)
    os.makedirs(target_dir, exist_ok=True)

    # 2. Define Unique Filename. Tag with basis if known to keep
    #    side-by-side comparison runs distinguishable on disk.
    basis_tag = ""
    if config is not None and getattr(config, 'pion_basis', None):
        basis_tag = f"_{config.pion_basis}"
        # For the amplitude basis, the cutoff method distinguishes otherwise
        # identically-tagged comparison runs (energy_bound vs NS).
        if config.pion_basis == 'amplitude' and getattr(config, 'cutoff_method', None):
            basis_tag += f"_{config.cutoff_method}"
    filename = f"sweep_L{L}_{dim}D{basis_tag}_{current_time}.json"
    filepath = os.path.join(target_dir, filename)

    # 3. Assemble Data Package
    metadata = {
        "L": L,
        "dim": dim,
        "params": params,
        "timestamp": current_time,
        "date": current_date,
    }
    if config is not None:
        metadata["config"] = config.to_dict()

    data_package = {
        "metadata": metadata,
        "results": sweep_results,
    }

    # 4. Write to JSON
    with open(filepath, 'w') as f:
        json.dump(data_package, f, indent=4)

    print(f"\n[DataIO] Successfully saved sweep data to: {filepath}")
    return filepath
