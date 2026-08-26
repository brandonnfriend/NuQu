"""Realistic total GSEE cost assembler — the classically-informed warm-start payoff (total_costs §4 #3).

Combines the four logical pieces into one honest total and contrasts a COLD start (bare single
determinant) with a WARM start (the frame-optimized D-determinant compact core through U):

    total_qubits = walk_register + max(m_QPE + ae, a_prep)      (max, not sum — reuse; ae only binary)
    total_T      = R(p0) · (N_walk·walk_T + T_prep)             (per-window QPE + prep, × repetitions)

where N_walk = π·λ/ε_qpe (adopted headline), m = ⌈log₂(N_walk/2)⌉, `R(p0)` is the branch-aware
repetition factor (`overlap_repetition_factor`), and `a_prep`/`T_prep` come from `state_prep_cost`.

The warm-start SAVING is `total_T_cold / total_T_warm`. Because state prep is sub-dominant to the
walk (Berry 2024 §VII.C), the saving is dominated by the repetition ratio `R_cold/R_warm`, i.e. by
how much the frame raises the overlap p0. Honest-claim: p0 comes from `frame_qpe.warmstart_fidelity`
at ED-tractable L (genuine) or the self-referential proxy at larger L (labeled).
"""

from src_PI.estimation.qpe_cost import (
    WALK_QUERY_CONSTANT_HEISENBERG,
    overlap_repetition_factor,
    qpe_phase_register_qubits_from_nwalk,
    total_logical_qubits,
    walk_queries,
)
from src_PI.estimation.state_prep_cost import state_prep_cost


def total_gsee_cost(*, physical_lambda, walk_T, walk_register_qubits, eps_qpe,
                    p0_warm, p0_cold, D_warm, n_bos_modes, N_f, b=17, D_cold=1,
                    confidence=0.99, branch="sampling", lambda_qr=None, displace=False,
                    n_walk_constant=WALK_QUERY_CONSTANT_HEISENBERG, n_walk_override=None):
    """Realistic total GSEE cost, cold vs warm. Returns per-start `{p0,R,expected_shots,branch,
    total_T,total_qubits,T_prep,a_prep,ae_register_qubits}` plus `warmstart_saving_x` (fixed-
    confidence R ratio) and `warmstart_saving_expected_x` (mean-shots p0 ratio) and the shared
    per-window quantities.

    Cold start = a single bare determinant (`D_cold=1`): trivial prep, but low p0 → many reps.
    Warm start = the frame core (`D_warm` determinants through the Gaussian U): modest prep, high
    p0 → few reps. The frame does NOT change the walk (same λ, same walk_T) — only p0 and prep.
    `branch='sampling'` (exact Bernoulli) is the default; 'binary' is experimental (warmstart audit).

    `n_walk_override`: pass a precomputed N_walk to consume it as-is (e.g. the shard's stored
    `QPE_Walk_Queries` to reproduce the historical √2·π anchor). Left None, N_walk is derived with
    the adopted π constant from λ / ε_qpe. `total_gsee_cost_from_record` derives with π by default.
    """
    N_walk = (float(n_walk_override) if n_walk_override is not None
              else walk_queries(physical_lambda, eps_qpe, constant=n_walk_constant))
    coherent_query_T = N_walk * walk_T                       # one QPE window
    m = qpe_phase_register_qubits_from_nwalk(N_walk)

    def _one(p0, D):
        rep = overlap_repetition_factor(p0, confidence=confidence, branch=branch)
        if D <= 1:
            prep = {"T_prep": 0.0, "a_prep": 0}              # single bare determinant: trivial prep
        else:
            prep = state_prep_cost(D, n_bos_modes, N_f, b=b, lambda_qr=lambda_qr, displace=displace)
        # the amplitude-estimation register (binary branch) coexists with the phase register during
        # QPE; state prep (a_prep) runs before. Peak width = walk + max(m+ae, a_prep).
        width = total_logical_qubits(walk_register_qubits, m + rep["ae_register_qubits"], prep["a_prep"])
        total_T = rep["R"] * (coherent_query_T + prep["T_prep"])
        return {"p0": p0, "R": rep["R"], "expected_shots": rep["expected_shots"],
                "branch": rep["branch"], "total_T": total_T, "total_qubits": width,
                "T_prep": prep["T_prep"], "a_prep": prep["a_prep"],
                "ae_register_qubits": rep["ae_register_qubits"]}

    warm = _one(p0_warm, D_warm)
    cold = _one(p0_cold, D_cold)
    saving = (cold["total_T"] / warm["total_T"]) if warm["total_T"] > 0 else None
    return {
        "N_walk": N_walk, "coherent_query_T": coherent_query_T, "m_qpe": m,
        "n_walk_from_record": n_walk_override is not None,
        "prep_frac_of_window": (warm["T_prep"] / coherent_query_T) if coherent_query_T else None,
        "warm": warm, "cold": cold,
        "warmstart_saving_x": saving,                        # fixed-confidence integer R ratio
        "warmstart_saving_expected_x": p0_warm / p0_cold if p0_cold > 0 else None,  # mean-shots ratio
    }


def total_gsee_cost_from_record(record, *, p0_warm, p0_cold, D_warm, n_bos_modes, N_f, **kw):
    """Adapter: pull λ / walk_T / walk-register / ε_qpe from a quantum-shard result dict and derive
    N_walk with the ADOPTED π constant (headline convention) from λ / ε_qpe. To instead reproduce
    the historical √2·π anchor query count, pass `n_walk_override=record['QPE_Walk_Queries']` in kw
    (a deliberate versioned scenario)."""
    b = record.get("QPE_Budget") or {}
    return total_gsee_cost(
        physical_lambda=record["Physical_Lambda"], walk_T=record["Walk_T_Count"],
        walk_register_qubits=record["Logical_Qubits"], eps_qpe=b.get("eps_qpe"),
        p0_warm=p0_warm, p0_cold=p0_cold, D_warm=D_warm,
        n_bos_modes=n_bos_modes, N_f=N_f, **kw)
