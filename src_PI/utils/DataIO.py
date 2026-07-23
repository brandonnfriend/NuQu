import os
import json
from datetime import datetime


def save_sweep_data(L, dim, params, sweep_results, config=None, label=None,
                    data_root=None):
    """
    Save an EFT resource-estimation sweep as a self-describing per-run folder.

    Mirrors the classical side (`classical/io.save_classical_run`): one run ==
    one folder under ``data/quantum/<date>/<run-id>/``, holding this sweep's
    outputs (and, when landed from the HPC, its condor logs) together — instead
    of loose files sharing a flat date directory. See ``data/README.md``.

    Args:
        L (int): Lattice side length.
        dim (int): Dimensionality.
        params (dict): The physical parameters used for the sweep.
        sweep_results (list): A list of dictionaries containing the data for each A.
        config (Config | None): The pipeline config (basis, walk_mode, ...).
            Saved into metadata so the sweep file is self-describing about
            which design-axis choices were active. None tolerated for
            backward-compatibility with old call sites.
        label (str | None): optional short human tag folded into the run-id, so
            an important run is easy to find later (e.g. "paperfig", "L4-highA").
        data_root (str | None): base directory for runs. Defaults to
            ``data/quantum``. Override (e.g. a tmp dir) for tests.

    Returns:
        Path to the written sweep JSON: ``<data_root>/<date>/<run-id>/sweep.json``.
    """
    current_date = datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.now().strftime("%H%M%S")
    if data_root is None:
        data_root = os.path.join("data", "quantum")

    # Tag with basis (and, for the amplitude basis, the cutoff method) so
    # side-by-side comparison runs stay distinguishable in the run-id.
    basis_tag = ""
    if config is not None and getattr(config, 'pion_basis', None):
        basis_tag = f"_{config.pion_basis}"
        if config.pion_basis == 'amplitude' and getattr(config, 'cutoff_method', None):
            basis_tag += f"_{config.cutoff_method}"

    # 1. Per-run subfolder: data/quantum/<date>/<run-id>/ ---------------------
    label_tag = f"_{label}" if label else ""
    run_id = f"sweep_L{L}_{dim}D{basis_tag}_{current_time}{label_tag}"
    run_dir = os.path.join(data_root, current_date, run_id)
    os.makedirs(run_dir, exist_ok=True)
    filepath = os.path.join(run_dir, "sweep.json")

    # 2. Assemble Data Package ----------------------------------------------
    metadata = {
        "L": L,
        "dim": dim,
        "params": params,
        "timestamp": current_time,
        "date": current_date,
        "label": label,
    }
    if config is not None:
        metadata["config"] = config.to_dict()

    data_package = {
        "metadata": metadata,
        "results": sweep_results,
    }

    # 3. Write to JSON -------------------------------------------------------
    with open(filepath, 'w') as f:
        json.dump(data_package, f, indent=4)

    print(f"\n[DataIO] Saved sweep run to: {filepath}")
    return filepath
