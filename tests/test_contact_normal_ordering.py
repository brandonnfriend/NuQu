"""Contact-term Wick-ordering oracle (the 2026-09-14 fix).

WHY THIS FILE EXISTS
--------------------
`StaticTerms.HC`/`HCI2` used to call OpenFermion's `normal_ordered`, which is an
ALGEBRA-PRESERVING REWRITE (`normal_ordered(X) == X` as operators — it applies the
anticommutation relations and keeps every contraction), NOT the `: :` prescription
Watson 2026 Eqs. (54)/(55) call for, which DISCARDS the contractions. The result was a
spurious one-body nucleon self-interaction `c·N̂`, `c = C/2 + 3·C_I²/2 = −23.3725` MeV:
a lone nucleon on an empty lattice felt the contact interaction with itself.

Hermiticity, fermion-number conservation and every energy DIFFERENCE were unaffected,
which is exactly why it survived — so the decisive tests here are the two Wick
identities against explicitly constructed oracles, plus the exact λ closed form.

Run: .venv/bin/python -m pytest -q tests/test_contact_normal_ordering.py
"""

import os
import sys

import pytest
from openfermion import FermionOperator, QubitOperator, jordan_wigner, normal_ordered

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src_PI.hamiltonians.ConstructEFT import build_eft_hamiltonian
from src_PI.hamiltonians.core.EFTParameters import get_physical_parameters
from src_PI.hamiltonians.core.StaticTerms import (
    HC,
    HCI2,
    Static_Nucleon_Hamiltonian,
    _wick,
    contact_self_energy_shift,
    rho,
    rho_I,
)
from src_PI.utils.Config import Config
from src_PI.utils.LatticeGeometry import get_total_sites

PARAMS = get_physical_parameters()
C, CI, H_HOP = PARAMS['C'], PARAMS['CI'], PARAMS['h']
C_SELF = C / 2.0 + 3.0 * CI / 2.0          # −23.3725 MeV


def _zero(op):
    return normal_ordered(op) == FermionOperator()


# --------------------------------------------------------------------------
# 1. The two Wick identities — independent oracles, owing nothing to the code
# --------------------------------------------------------------------------

def test_rho_squared_wick_identity():
    """ρ² = :ρ²: + ρ, with :ρ²: built as the explicit Σ_{σ≠σ'} N_σ N_σ' oracle."""
    r = rho(0, n_b=0)
    oracle = FermionOperator()
    for a in range(4):
        for b in range(4):
            if a != b:
                oracle += FermionOperator(f'{a}^ {a} {b}^ {b}')
    assert _zero(_wick(r * r) - oracle), "_wick(ρ²) != Σ_{σ≠σ'} N_σ N_σ'"
    assert _zero(r * r - oracle - r), "ρ² != :ρ²: + ρ"


def test_rho_I_squared_wick_identity():
    """Σ_I ρ_I² = :Σ_I ρ_I²: + 3ρ, and :·: matches Watson's Eq. (60) bracket."""
    r = rho(0, n_b=0)
    S = FermionOperator()
    for I in (1, 2, 3):
        rI = rho_I(I, 0, n_b=0)
        S += rI * rI
    assert _zero(S - _wick(S) - 3 * r), "Σ_I ρ_I² != :Σ_I ρ_I²: + 3ρ"

    # Watson Eq. (60), in our on-site order (0,1,2,3) = (↑p, ↑n, ↓p, ↓n).
    up_p, up_n, dn_p, dn_n = 0, 1, 2, 3
    N = lambda i: FermionOperator(f'{i}^ {i}')
    watson60 = (N(up_p) * N(up_p) + N(dn_p) * N(dn_p)
                + N(up_n) * N(up_n) + N(dn_n) * N(dn_n)
                - 6 * N(up_p) * N(up_n) + 2 * N(up_p) * N(dn_p)
                - 2 * N(up_p) * N(dn_n) - 2 * N(dn_p) * N(up_n)
                + 2 * N(up_n) * N(dn_n) - 6 * N(dn_p) * N(dn_n)
                - 4 * (FermionOperator(f'{up_p}^ {dn_p} {dn_n}^ {up_n}')
                       + FermionOperator(f'{up_n}^ {dn_n} {dn_p}^ {up_p}')))
    assert _zero(_wick(watson60) - _wick(S)), "Watson Eq. (60) inconsistent with Eq. (55)"


# --------------------------------------------------------------------------
# 2. The defect the fix removes: a one-nucleon self-interaction
# --------------------------------------------------------------------------

def test_single_nucleon_has_zero_contact_energy():
    """One nucleon on an empty lattice must feel NO contact interaction.

    The legacy (un-prescribed) Hamiltonian gives it `c = −23.37` MeV of contact
    self-energy; the Wick-ordered one gives exactly zero.
    """
    import numpy as np
    from openfermion import get_sparse_operator

    L, dim, n_b = 1, 3, 0
    idx = 0                                    # |↑p⟩ on the single site
    for wick, expected in ((True, 0.0), (False, C_SELF)):
        H = HC(C, L, dim, n_b, wick_ordered=wick) + HCI2(CI, L, dim, n_b, wick_ordered=wick)
        M = get_sparse_operator(jordan_wigner(H), n_qubits=4).toarray()
        ket = np.zeros(16, dtype=complex)
        ket[1 << (3 - idx)] = 1.0              # OpenFermion: qubit 0 is most significant
        got = float(np.real(ket.conj() @ M @ ket))
        assert abs(got - expected) < 1e-9, (
            f"wick={wick}: single-nucleon contact energy {got}, expected {expected}")


@pytest.mark.parametrize("A", [0, 1, 2, 5, 32])
def test_offset_is_exactly_c_times_A(A):
    """`H_legacy − H_wick = c·N̂` as an operator ⇒ the sector offset is c·A."""
    assert abs(contact_self_energy_shift(A, PARAMS) - C_SELF * A) < 1e-12


@pytest.mark.parametrize("L,dim,n_b", [(2, 1, 1), (2, 2, 2), (2, 3, 2), (3, 3, 2)])
def test_difference_operator_is_the_number_operator(L, dim, n_b):
    """The legacy-minus-Wick difference is c·N̂ EXACTLY — no L, dim or n_b dependence."""
    legacy = Static_Nucleon_Hamiltonian(H_HOP, C, CI, L, dim, n_b, wick_ordered=False)
    wick = Static_Nucleon_Hamiltonian(H_HOP, C, CI, L, dim, n_b, wick_ordered=True)
    n_op = FermionOperator()
    from src_PI.utils.LatticeGeometry import site_to_nucleon_qubit
    for s in range(get_total_sites(L, dim)):
        for mode in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            i = site_to_nucleon_qubit(s, mode, n_b)
            n_op += FermionOperator(f'{i}^ {i}')
    assert _zero(legacy - wick - C_SELF * n_op)


# --------------------------------------------------------------------------
# 3. The exact quantum-resource consequence
# --------------------------------------------------------------------------

DLAMBDA_PER_SITE = abs(C + 3.0 * CI)        # = 2|c| = 46.745 MeV


@pytest.mark.parametrize("L,dim,n_b", [(1, 3, 2), (2, 3, 1), (2, 3, 3), (3, 3, 2), (2, 2, 2)])
def test_lambda_shift_closed_form(L, dim, n_b):
    """Δλ = |C + 3C_I²|·L^dim exactly — independent of n_b, with the term count
    unchanged (no Pauli string is created or destroyed)."""
    lam, nterms = {}, {}
    for wick in (False, True):
        H = jordan_wigner(Static_Nucleon_Hamiltonian(
            H_HOP, C, CI, L, dim, n_b, wick_ordered=wick))
        lam[wick] = sum(abs(v) for k, v in H.terms.items() if k)
        nterms[wick] = len([k for k in H.terms if k])
    assert nterms[False] == nterms[True], "Pauli term count changed"
    assert abs((lam[False] - lam[True]) - DLAMBDA_PER_SITE * L ** dim) < 1e-6


def test_full_bundle_lambda_shift_matches_static_sector():
    """No pion-sector collision: the full-H λ moves by exactly the static-sector amount.

    (H_AV and H_WT carry traceless boson factors, so they contribute no pure
    single-Z nucleon string that could interfere with the correction.)
    """
    from src_PI.estimation.NormalizeHamiltonians import normalize_for_qpe
    L, dim, n_b = 2, 3, 2
    lam = {}
    for wick in (False, True):
        cfg = Config(pion_basis='fock', block_encoder='pauli_lcu',
                     wick_ordered_contacts=wick)
        bundle, _, _ = build_eft_hamiltonian(L, dim, n_b, 0.0, PARAMS, cfg)
        lam[wick] = normalize_for_qpe(bundle)['physical_lambda']
    assert abs((lam[False] - lam[True]) - DLAMBDA_PER_SITE * L ** dim) < 1e-6


# --------------------------------------------------------------------------
# 4. The legacy switch must stay bit-reproducible
# --------------------------------------------------------------------------

def test_legacy_switch_reproduces_published_lambda():
    """`wick_ordered_contacts=False` reproduces the published r3 anchor λ."""
    from src_PI.estimation.NormalizeHamiltonians import normalize_for_qpe
    cfg = Config(pion_basis='fock', block_encoder='pauli_lcu',
                 wick_ordered_contacts=False)
    bundle, _, _ = build_eft_hamiltonian(2, 3, 2, 0.0, PARAMS, cfg)
    assert abs(normalize_for_qpe(bundle)['physical_lambda'] - 46521.1) < 0.2


def test_config_records_the_axis():
    """The bundle metadata must record which convention built it."""
    for wick in (False, True):
        cfg = Config(pion_basis='fock', block_encoder='pauli_lcu',
                     wick_ordered_contacts=wick)
        bundle, _, _ = build_eft_hamiltonian(1, 3, 2, 0.0, PARAMS, cfg)
        assert bundle.metadata['wick_ordered_contacts'] is wick


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-q']))
