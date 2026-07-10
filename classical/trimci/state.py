"""
Mixed fermion-boson Fock basis states.

A basis state of the dynamical-pion EFT is a tensor product

    |Phi> = |fermion determinant> (x) |boson occupations>

(see `claude/research/trimci/01_hamiltonian_form.md` Sec. 0). We store:

  * `ferm`: an integer bitmask over the compact fermion-mode indices
            (bit p set  <=>  nucleon spin-orbital p occupied). Modes are
            ordered ascending; the fermionic sign convention matches
            OpenFermion (parity = number of occupied modes with index < p).
  * `bos`:  a tuple of non-negative ints, one per pion mode, each in
            [0, N_f-1] (the Fock cutoff).

States are immutable and hashable so they can be graph nodes / dict keys
in the TrimCI core (`graph.py`).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from math import comb

# Default cap on how many basis states enumerate_basis will materialize. Each
# MixedState is a small Python object (~150-250 B incl. tuple), so 1e5 states
# is tens of MB. This guard prevents the OOM crashes a full-sector enumeration
# can trigger (size = C(n_ferm,A) * N_f**n_bos grows explosively in N_f/n_bos).
# Callers that genuinely need more — and have the RAM — pass max_states.
MAX_BASIS = 100_000


@dataclass(frozen=True)
class MixedState:
    ferm: int            # bitmask over fermion modes
    bos: tuple           # tuple[int], length = n_boson_modes

    def occupied_fermions(self):
        """Sorted list of occupied fermion-mode indices."""
        return [p for p in range(self.ferm.bit_length()) if (self.ferm >> p) & 1]

    def n_fermions(self):
        return bin(self.ferm).count("1")

    def total_bosons(self):
        return sum(self.bos)

    def __repr__(self):
        occ = "".join("1" if (self.ferm >> p) & 1 else "0"
                      for p in range(max(1, self.ferm.bit_length())))
        return f"MixedState(f={occ}, b={self.bos})"


def fermion_determinants(n_modes, n_elec=None):
    """Enumerate fermion bitmasks.

    n_elec=None -> the full 2**n_modes Fock space (all particle numbers).
    n_elec=k    -> only determinants with exactly k occupied modes
                   (the nucleon-number sector A=k, which every term in H
                   conserves).
    """
    if n_elec is None:
        for occ in range(1 << n_modes):
            yield occ
    else:
        for combo in itertools.combinations(range(n_modes), n_elec):
            occ = 0
            for p in combo:
                occ |= (1 << p)
            yield occ


def boson_occupations(n_modes, N_f):
    """Enumerate every boson occupation vector in the cutoff box [0,N_f)^n_modes.

    WARNING: size N_f**n_modes — toy/ED use only. Boson number is *not*
    conserved by H (H_AV is linear, H_WT/gradient bilinear in ladders), so
    there is no boson-number sector to restrict to.
    """
    if n_modes == 0:
        yield ()
        return
    for combo in itertools.product(range(N_f), repeat=n_modes):
        yield combo


def basis_size(n_ferm_modes, n_bos_modes, N_f, n_elec=None):
    """Size of the full mixed basis without materializing it (pure int math)."""
    n_ferm = comb(n_ferm_modes, n_elec) if n_elec is not None else (1 << n_ferm_modes)
    return n_ferm * (N_f ** n_bos_modes)


def enumerate_basis(n_ferm_modes, n_bos_modes, N_f, n_elec=None,
                    max_states=MAX_BASIS):
    """Full mixed basis (fermion sector x boson box). Toy/ED use only.

    Returns a list of MixedState. Size = |fermion sector| * N_f**n_bos_modes.

    Guarded against OOM: the size is computed up front (cheap int arithmetic)
    and the call refuses to allocate more than `max_states` states. Keep n_elec
    set and N_f small — this is the exact-diagonalization / full-space-Lanczos
    enumerator, not the TrimCI selected-CI path.
    """
    size = basis_size(n_ferm_modes, n_bos_modes, N_f, n_elec)
    if size > max_states:
        raise MemoryError(
            f"enumerate_basis would build {size:,} states (cap {max_states:,}; "
            f"~{size * 200 / 1e9:.2f} GB of MixedState objects). Reduce "
            f"N_f/n_bos_modes/n_elec, or pass a larger max_states if you have "
            f"the RAM."
        )
    states = []
    for occ in fermion_determinants(n_ferm_modes, n_elec):
        for bos in boson_occupations(n_bos_modes, N_f):
            states.append(MixedState(occ, bos))
    return states
