"""TrimCI-style classical solver over the mixed fermion-boson EFT.

Modules:
  * state        — MixedState (fermion determinant (x) boson occupations)
  * hamiltonian  — build the EFT H as a flat ladder-operator term list (MixedH)
  * hij          — mixed-state H_ij evaluator (matrix-free connections)
  * dump         — generalized-FCIDUMP writer/reader
  * graph        — TrimCI core (expansion / local + global trim / ground_state)
  * pt2          — Epstein-Nesbet PT2 correction over the mixed selected-CI state
  * extrapolation— honest E_infinity fit (power-law + SHCI/PT2) with uncertainty
                   and the 3-number energy report (variational / +PT2 / extrapolated)

See TODO.md for the integration roadmap toward the released TrimCI package.
"""

from .hamiltonian import MixedH, OperatorTerm, build_from_eft, from_mixed_hamiltonian
from .hij import build_dense, connections, h_ij
from .state import MixedState, enumerate_basis
from .graph import (ground_state, ground_state_ensemble, exact_ground_state,
                    boson_occupation_weights)
from .lanczos import lanczos_ground_state
from .backend import (backend_diagonalize, backend_available, davidson_lowest,
                      backend_diagonalize_sparse, davidson_lowest_sparse,
                      has_sparse_davidson, cpp_available, cpp_diagonalize,
                      cpp_expand, cpp_ground_state, cpp_ground_state_ensemble)
from .pt2 import epstein_nesbet_pt2, pt2_from_result
from .extrapolation import (fit_einf_power, fit_einf_pt2, report_energies)
from .observables import (mean_occupation, occupation_tail, occupation_histogram,
                          occupation_from_coeffs)
from .tong_bound import (mean_occupation_scs, squeeze_r_star, squeezed_tail,
                         cutoff_predictions)
from .dump import write_dump, read_dump, summarize

__all__ = [
    "MixedH", "OperatorTerm", "build_from_eft", "from_mixed_hamiltonian",
    "build_dense", "connections", "h_ij",
    "MixedState", "enumerate_basis",
    "ground_state", "ground_state_ensemble", "exact_ground_state",
    "boson_occupation_weights", "lanczos_ground_state",
    "backend_diagonalize", "backend_available", "davidson_lowest",
    "backend_diagonalize_sparse", "davidson_lowest_sparse", "has_sparse_davidson",
    "cpp_available", "cpp_diagonalize", "cpp_expand",
    "cpp_ground_state", "cpp_ground_state_ensemble",
    "epstein_nesbet_pt2", "pt2_from_result",
    "fit_einf_power", "fit_einf_pt2", "report_energies",
    "mean_occupation", "occupation_tail", "occupation_histogram",
    "occupation_from_coeffs",
    "mean_occupation_scs", "squeeze_r_star", "squeezed_tail", "cutoff_predictions",
    "write_dump", "read_dump", "summarize",
]
