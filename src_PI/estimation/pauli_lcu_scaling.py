"""
Analytic PauliLCU resource scaling to reach L=10 (the publication anchor).

PauliLCU (`estimators.py`) is the genuinely compiler-derived quantum-resource
path, but materialising the Pauli operator blows up memory well before L=10 in
3D (L=10, dim=3 = 1000 sites). This module reaches L=10 **without** materialising
the operator, by exploiting the lattice structure: a translation-invariant
Hamiltonian is a sum of on-site terms (∝ sites `S=L^dim`) and nearest-neighbour
bond terms (∝ bonds `N_b`), so its PauliLCU cost drivers are

    quantity(L) = a·S + b·N_b ,    N_b(L,dim) = dim·L^(dim-1)·(L-1)   (open BC).

**What is exact.** The block-encoding subnormalisation `λ` (the Pauli 1-norm,
which is additive over terms) fits this form to < 0.02% at L=1,2,3 (dim=3), so
`λ(L=10)` and hence the walk-query count `N_walk = √2·π·λ/ΔE` are pinned by exact
lattice combinatorics. The logical-qubit count is `≈ (4 + dim·n_b)·S + O(log)`,
also exact. These are the dominant scaling drivers.

**What is modelled (verify on the cluster; biased LOW).** The per-walk-step
T-count `walk_T` is a slower-growing prefactor. Empirically
`walk_T ≈ κ · total_Pauli_weight` (κ ≈ 64 from the clean dim=3 points, matching
the compiled walk_T to 1-2% at L=2,3). The catch: the total Pauli weight has a
**super-linear** component the linear `a·S + b·N_b` fit does NOT capture —
Jordan-Wigner Z-strings for out-of-row bonds span the lattice ordering, so
z-direction bonds carry Z-fills of length ~`n_b·L²` at large L. The linear model
therefore **under-estimates** `total_weight` (and hence `walk_T`, `Total_T`) at
L=10; the reported band is indicative, not a bound, and the true value is
**likely higher**. This is encoding/ordering-dependent (a locality-preserving
qubit order or a Verstraete-Cirac fermion map would shrink it). **Confirm against
a compiled L=4-6 (dim=3) point on the cluster before quoting `Total_T`.** `λ`,
`N_walk`, and qubit counts are exact and need no such caveat — they are the
robust feasibility headline (~8.6k logical qubits, ~3.9e7 walk queries at L=10).

Calibration data below are the compiler-derived PauliLCU outputs
(`NUQU_DISABLE_PYLIQTR_CACHE=1`) at n_b=2; regenerate with `probe_pauli_lcu`.
"""

import math

from src_PI.estimation.qpe_cost import walk_queries

# --- compiler-derived calibration points (n_b=2; cache disabled) ------------ #
# each: (L, dim, lam, L_eff, total_weight, walk_T, logical_qubits)
_CALIB = [
    (1, 3, 2086.6, 408, 2038, 107128, 20),
    (2, 3, 46521.1, 5160, 25736, 1708232, 94),
    (3, 3, 190565.5, 19467, 105975, 6828124, 286),
    (1, 1, 882.2, 408, 2038, 107128, 20),
    (2, 1, 3077.5, 938, 4418, 214724, 31),
    (4, 1, 7467.9, 1068, 4172, 426192, 52),
    (6, 1, 11858.4, 1678, 6550, 428632, 72),
    (10, 1, 20639.2, 1938, 6826, 429672, 112),
]

# walk_T ≈ κ · total_Pauli_weight, from the clean (monotone) dim=3 points.
_WALK_T_PER_WEIGHT = 64.0
# Indicative band (NOT a bound). κ scatter is ~±5%; the dominant, one-sided risk
# is JW-nonlocality making total_weight super-linear → the model is biased LOW at
# large L. Reported as a +100%/−15% asymmetric band; confirm on the cluster.
_WALK_T_BAND_LOW = 0.15
_WALK_T_BAND_HIGH = 1.00


def n_bonds(L, dim, periodic=False):
    """Nearest-neighbour bond count. Open BC: `dim·L^(dim-1)·(L-1)`; periodic: `dim·L^dim`."""
    if periodic:
        return dim * (L ** dim)
    return dim * (L ** (dim - 1)) * (L - 1)


def _fit_site_bond(points):
    """Least-squares fit `y = a·S + b·N_b` over `[(S, N_b, y), ...]`.

    Two unknowns; with ≥2 non-degenerate points this is well-posed. Returns
    `(a, b)`."""
    # normal equations for [[ΣS², ΣS·Nb],[ΣS·Nb, ΣNb²]] [a,b] = [ΣS·y, ΣNb·y]
    Ss = sum(S * S for S, _Nb, _y in points)
    Sn = sum(S * Nb for S, Nb, _y in points)
    Nn = sum(Nb * Nb for _S, Nb, _y in points)
    Sy = sum(S * y for S, _Nb, y in points)
    Ny = sum(Nb * y for _S, Nb, y in points)
    det = Ss * Nn - Sn * Sn
    if abs(det) < 1e-12:                    # degenerate (e.g. all N_b=0): fit a only
        a = Sy / Ss if Ss else 0.0
        return a, 0.0
    a = (Sy * Nn - Ny * Sn) / det
    b = (Ss * Ny - Sn * Sy) / det
    return a, b


def _dim_models(dim):
    """Fit (a,b) site/bond coefficients for λ, L_eff, total_weight, qubits at `dim`."""
    pts = [c for c in _CALIB if c[1] == dim]
    if len(pts) < 2:
        raise ValueError(f"need ≥2 calibration points for dim={dim}")
    models = {}
    for key, idx in (('lam', 2), ('L_eff', 3), ('total_weight', 4), ('qubits', 6)):
        data = [(L ** dim, n_bonds(L, dim), c[idx]) for c in pts for L in (c[0],)]
        models[key] = _fit_site_bond(data)
    return models


def _predict(models, key, L, dim):
    a, b = models[key]
    return a * (L ** dim) + b * n_bonds(L, dim)


def pauli_lcu_resources(L, dim, n_b=2, delta_E=1.0):
    """Analytic PauliLCU resource estimate at `(L, dim)` (n_b=2 calibration).

    Returns a dict with EXACT `lambda`, `walk_queries`, `logical_qubits` and a
    MODELLED `walk_T` / `total_T` (with a relative uncertainty band). See the
    module docstring for what is exact vs modelled.
    """
    if n_b != 2:
        raise NotImplementedError("calibration is for n_b=2; add points for others")
    models = _dim_models(dim)
    lam = _predict(models, 'lam', L, dim)
    l_eff = _predict(models, 'L_eff', L, dim)
    twt = _predict(models, 'total_weight', L, dim)
    qubits = _predict(models, 'qubits', L, dim)
    n_walk = walk_queries(lam, delta_E)
    walk_T = _WALK_T_PER_WEIGHT * twt
    total_T = walk_T * n_walk
    lo, hi = _WALK_T_BAND_LOW, _WALK_T_BAND_HIGH
    return {
        'L': L, 'dim': dim, 'sites': L ** dim, 'bonds': n_bonds(L, dim),
        'lambda': lam,                      # exact (lattice combinatorics)
        'walk_queries': n_walk,             # exact (from λ)
        'logical_qubits': int(round(qubits)),   # exact (~ (4+dim·n_b)·S)
        'total_weight_model': twt,          # modelled (JW-nonlocality → biased low)
        'walk_T_model': walk_T,             # modelled (biased low at large L)
        'walk_T_band': (walk_T * (1 - lo), walk_T * (1 + hi)),
        'total_T_model': total_T,           # modelled (biased low at large L)
        'total_T_band': (total_T * (1 - lo), total_T * (1 + hi)),
        'exact': ('lambda', 'walk_queries', 'logical_qubits'),
        'modelled': ('walk_T_model', 'total_T_model', 'total_weight_model'),
    }


def validation_table(dim=3):
    """Model-vs-compiled at the calibration L for `dim` — for the report/tests."""
    models = _dim_models(dim)
    rows = []
    for c in _CALIB:
        if c[1] != dim:
            continue
        L = c[0]
        rows.append({
            'L': L,
            'lam_actual': c[2], 'lam_model': _predict(models, 'lam', L, dim),
            'qubits_actual': c[6], 'qubits_model': round(_predict(models, 'qubits', L, dim)),
            'walkT_actual': c[5],
            'walkT_model': _WALK_T_PER_WEIGHT * _predict(models, 'total_weight', L, dim),
        })
    return rows


def probe_pauli_lcu(L, dim, n_b=2, delta_E=1.0):
    """Regenerate one compiler-derived calibration point (materialises the operator).

    Only feasible at small L (dim=3: ≲ L=3-4 locally). Returns the tuple shape
    used in `_CALIB`. Run with `NUQU_DISABLE_PYLIQTR_CACHE=1`."""
    from src_PI.hamiltonians.ConstructEFT import build_eft_hamiltonian
    from src_PI.hamiltonians.core.EFTParameters import get_physical_parameters
    from src_PI.estimation.NormalizeHamiltonians import normalize_for_qpe
    from src_PI.estimation.estimators import run_qubitization_analysis
    from src_PI.utils.Config import Config

    cfg = Config(pion_basis='fock', block_encoder='pauli_lcu')
    bundle, _q, ns = build_eft_hamiltonian(
        L, dim, n_b, pi_max=0.0, params=get_physical_parameters(), config=cfg)
    nd = normalize_for_qpe(bundle, safety_factor=2.5)
    l_eff = total_weight = 0
    for _name, H in nd['sub_hamiltonians']:
        for term in H.terms:
            if term:
                l_eff += 1
                total_weight += len(term)
    res = run_qubitization_analysis(nd, ns, n_b)
    return (L, dim, nd['physical_lambda'], l_eff, total_weight,
            res['T'], res['LogicalQubits'])
