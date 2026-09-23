"""The shard contact-convention tag (2026-09-23) and the loaders that read it.

`misc.run_frame_shard` builds the Wick-ordered H since the 2026-09-14 fix, but the loaders
used to assume EVERY shard was run in the legacy convention and add +23.3725*A MeV. A new
shard would therefore have been shifted twice. Shards now carry `contact_convention`, and
`apply_to_rungs(..., native=shard_convention(j))` shifts from what the shard was run in.

Checks:
  * `build_from_eft` records the ordering it actually used (the Config default: wick);
  * an untagged shard is legacy; an unknown tag raises;
  * `rung_shift` is zero native->same, +23.3725*A legacy->wick, and its inverse;
  * `aggregate_classical_energies.load` puts a legacy shard and a Wick shard of the same
    physics on ONE energy (no double shift), filters by `A`, and refuses to pool two A
    values under one (n_b, L) key (the explicit-A sweep puts all A in one directory).
"""
import json
import os
import shutil
import sys
import tempfile

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from misc.apply_wick_correction import C_SELF, rung_shift, shard_convention  # noqa: E402


def _write(path, L, A, seed, E, convention=None):
    rungs = [{"core": 1000 * 2 ** i, "E_var": E + 40.0 / (i + 1), "dE_pt2": -10.0 / (i + 1)}
             for i in range(6)]
    j = {"kind": "frame_shard", "L": L, "dim": 3, "A": A, "filling": None, "frame": "bare",
         "seed": seed, "n_b": 3, "N_f": 8, "sites": L ** 3, "n_terms": 1777,
         "rungs": rungs, "done": True}
    if convention is not None:
        j["contact_convention"] = convention
    json.dump(j, open(path, "w"))


def test_build_from_eft_records_convention():
    from classical.trimci import build_from_eft
    H = build_from_eft(1, 1, 1, transform="bare")
    assert H.meta["contact_convention"] == "wick"


def test_shard_convention_and_shift():
    assert shard_convention({}) == "legacy"
    assert shard_convention({"contact_convention": "wick"}) == "wick"
    try:
        shard_convention({"contact_convention": "normal"})
    except ValueError:
        pass
    else:
        raise AssertionError("unknown convention tag accepted")
    A = 7
    assert rung_shift(A, "wick", native="wick") == 0.0
    assert rung_shift(A, "legacy", native="legacy") == 0.0
    assert abs(rung_shift(A, "wick", native="legacy") - (-C_SELF * A)) < 1e-12
    assert abs(rung_shift(A, "legacy", native="wick") - (C_SELF * A)) < 1e-12
    # the historical default (untagged = legacy) is unchanged
    assert rung_shift(A, "wick") == rung_shift(A, "wick", native="legacy")


def test_loader_no_double_shift_and_A_handling():
    from misc.aggregate_classical_energies import load
    tmp = tempfile.mkdtemp(prefix="convtag_")
    try:
        d_leg = os.path.join(tmp, "2026-09-06", "legacy_run")
        d_new = os.path.join(tmp, "2026-09-23", "wick_run")
        os.makedirs(d_leg)
        os.makedirs(d_new)
        A, E_leg = 8, 2000.0
        E_wick = E_leg - C_SELF * A                    # same physics, run Wick-ordered
        _write(os.path.join(d_leg, "bare_L2d3_f1.0_s0.json"), 2, A, 0, E_leg)
        _write(os.path.join(d_new, "bare_L2d3_A8_s1.json"), 2, A, 1, E_wick, "wick")
        _write(os.path.join(d_new, "bare_L2d3_A4_s0.json"), 2, 4, 0, 1500.0, "wick")

        for conv in ("wick", "legacy"):
            g, meta = load([d_leg, d_new], convention=conv, A=8)
            e0, e1 = (g[(3, 2)][s][-1]["E_var"] for s in (0, 1))
            assert abs(e0 - e1) < 1e-9, f"{conv}: legacy {e0} vs wick-run {e1} disagree"
            assert int(meta[(3, 2)]["A"]) == 8

        g, _ = load([d_new], convention="wick", A=4)
        assert abs(g[(3, 2)][0][-1]["E_var"] - (1500.0 + 40.0 / 6)) < 1e-9, "wick shard shifted"

        try:
            load([d_new], convention="wick")
        except ValueError:
            pass
        else:
            raise AssertionError("two A values pooled under one (n_b, L) key")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    test_build_from_eft_records_convention()
    test_shard_convention_and_shift()
    test_loader_no_double_shift_and_A_handling()
    print("test_contact_convention_tag: PASS  (build records wick; untagged=legacy; "
          "legacy + wick-run shards agree after loading; A filter + mixed-A refusal)")


if __name__ == "__main__":
    main()
