"""Pin the quantum save layout: one run -> one folder under data/quantum/.

`save_sweep_data` must write ``<data_root>/<date>/<run-id>/sweep.json`` (mirroring
the classical `save_classical_run` convention), fold an optional label into the
run-id, and round-trip the results. Run:

    python -m tests.test_dataio_layout
"""
import json
import os
import tempfile

from src_PI.utils.DataIO import save_sweep_data


class _Cfg:
    """Minimal stand-in for Config (only what save_sweep_data touches)."""
    pion_basis = "amplitude"
    cutoff_method = "ns"

    def to_dict(self):
        return {"pion_basis": self.pion_basis, "cutoff_method": self.cutoff_method}


def test_per_run_subfolder():
    root = tempfile.mkdtemp()
    results = [{"A": 1, "Total_T_Count": 123}, {"A": 2, "Total_T_Count": 456}]

    fp = save_sweep_data(2, 3, {"m_pi": 138.0}, results,
                         config=_Cfg(), label="unit-test", data_root=root)

    # Path shape: <root>/<date>/<run-id>/sweep.json
    assert os.path.basename(fp) == "sweep.json", fp
    run_dir = os.path.dirname(fp)
    date_dir = os.path.dirname(run_dir)
    assert os.path.dirname(date_dir) == root, f"expected <root>/<date>/<run-id>/, got {fp}"
    run_id = os.path.basename(run_dir)
    assert run_id.startswith("sweep_L2_3D"), run_id
    assert "amplitude_ns" in run_id, f"basis tag missing from run-id: {run_id}"
    assert run_id.endswith("_unit-test"), f"label missing from run-id: {run_id}"

    with open(fp) as f:
        pkg = json.load(f)
    assert pkg["metadata"]["label"] == "unit-test"
    assert pkg["metadata"]["config"] == {"pion_basis": "amplitude", "cutoff_method": "ns"}
    assert pkg["results"] == results

    print("PASS test_per_run_subfolder")
    print(f"  wrote:  {fp}")
    print(f"  run-id: {run_id}")


if __name__ == "__main__":
    test_per_run_subfolder()
