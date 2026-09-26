"""Guards for the seed control verdict logic (`misc.compare_seed_control`).

The control exists to answer whether the low-occupation prior is circular, so the
verdict function must not be able to launder an ambiguous result into a PASS. Three
behaviours are load-bearing:

  * UNMATCHED ARMS ARE REFUSED. If the two arms differ in anything but the init, the
    comparison confounds initialization with search budget and must not produce a
    verdict at all.
  * DIRECTION IS DISTINGUISHED. Uniform ABOVE prior is benign (it searched worse;
    E_var is still a valid upper bound). Uniform BELOW prior is the actual failure —
    the prior would be steering the search away from the ground state. These must not
    collapse into one "mismatch" outcome.
  * BOTH OBSERVABLES GATE THE PASS. Energy agreement alone is not enough; the original
    control named E_var AND <N>, and the occupation is the quantity the prior most
    directly biases.
"""
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from misc.compare_seed_control import compare, settings_match, verdict

CFG = {"L": 2, "dim": 3, "A": 1, "N_f_list": [2, 4, 8, 16], "core": 4000,
       "n_runs": 16, "seed": 0, "pt2": False}


def _arms(pairs, cfg_u=None):
    """pairs: {N_f: (E_prior, N_prior, E_uniform, N_uniform)}"""
    pr = {nf: dict(N_f=nf, n_b=nf.bit_length() - 1, E_var=v[0], N_per_mode=v[1])
          for nf, v in pairs.items()}
    un = {nf: dict(N_f=nf, n_b=nf.bit_length() - 1, E_var=v[2], N_per_mode=v[3])
          for nf, v in pairs.items()}
    return {"prior": {"rows": pr, "cfg": CFG}, "uniform": {"rows": un, "cfg": cfg_u or CFG}}


def test_unmatched_arms_are_refused():
    arms = _arms({4: (100.0, 0.04, 100.0, 0.04)}, cfg_u={**CFG, "n_runs": 3})
    ok, why = settings_match(arms)
    assert not ok and "not matched" in why


def test_missing_arm_is_refused():
    arms = _arms({4: (100.0, 0.04, 100.0, 0.04)})
    del arms["uniform"]
    assert settings_match(arms)[0] is False


def test_arm_without_a_config_is_refused():
    """A retired pre-fix file has no matched-config record — we cannot certify it."""
    arms = _arms({4: (100.0, 0.04, 100.0, 0.04)})
    arms["uniform"]["cfg"] = None
    ok, why = settings_match(arms)
    assert not ok and "matched-config" in why


def test_agreement_is_a_pass():
    arms = _arms({2: (200.0, 0.05, 200.2, 0.051), 4: (100.0, 0.045, 100.1, 0.0455)})
    v, _ = verdict(compare(arms))
    assert v == "PASS"


def test_uniform_above_prior_is_inconclusive_not_failure():
    """Benign direction: the uniform ensemble searched worse. Must NOT read as bias."""
    arms = _arms({4: (100.0, 0.045, 140.0, 0.30)})
    v, why = verdict(compare(arms))
    assert v.startswith("INCONCLUSIVE") and "searched worse" in why


def test_uniform_below_prior_is_the_real_failure():
    """The prior would be steering the search away from the ground state."""
    arms = _arms({4: (100.0, 0.045, 80.0, 0.044)})
    v, why = verdict(compare(arms))
    assert v.startswith("FAIL") and "steering" in why


def test_occupation_disagreement_alone_blocks_the_pass():
    """Energies agree but <N> does not -> not a PASS. The occupation is exactly what
    the prior would bias, so it cannot be waved through on energy agreement."""
    arms = _arms({4: (100.0, 0.045, 100.1, 0.60)})
    rows = compare(arms)
    assert rows[0]["E_ok"] and not rows[0]["N_ok"]
    assert verdict(rows)[0].startswith("INCONCLUSIVE")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


def test_unmatched_arm_is_reported_not_dropped():
    """A held counterpart arm must still surface the surviving arm's numbers.

    The L=4 shards of cluster 293942 are the motivating case: both uniform arms hit
    OOM at 96 GB while both prior arms finished. Silently skipping the group would
    erase the only L=4 occupation we have AND the infeasibility that caused it.
    """
    from misc.compare_seed_control import unmatched_groups
    arms = _arms({4: (100.0, 0.04, 100.0, 0.04)})
    del arms["uniform"]
    out = unmatched_groups({(4, 64): arms, (2, 64): _arms({4: (1.0, 0.04, 1.0, 0.04)})})
    assert len(out) == 1, "only the single-arm group is unmatched"
    assert out[0]["key"] == (4, 64)
    assert out[0]["arm"] == "prior" and out[0]["missing"] == "uniform"
    assert [r["N_f"] for r in out[0]["rows"]] == [4]


def test_unmatched_group_gets_no_verdict():
    """No comparison exists, so nothing may launder into a PASS."""
    from misc.compare_seed_control import unmatched_groups
    arms = _arms({4: (100.0, 0.04, 100.0, 0.04)})
    del arms["uniform"]
    out = unmatched_groups({(4, 64): arms})
    assert "verdict" not in out[0]
