"""
Build the dynamical-pion EFT Hamiltonian as a flat list of elementary
ladder-operator terms, ready for the mixed-state H_ij evaluator (`hij.py`)
and the generalized dump writer (`dump.py`).

We do NOT re-derive the Hamiltonian here. We reuse the existing, tested
native-algebra builder in the quantum pipeline
(`src_PI.hamiltonians.ConstructEFT.build_eft_hamiltonian` with
pion_basis='fock', block_encoder='sparse'), which already assembles the
*complete* mixed operator as a `MixedHamiltonian`:

    * fermion_part : FermionOperator  = H_free + H_C + H_CI2  (static nucleon)
    * boson_part   : BosonOperator    = H_pion_free (m_pi n + zero-point + grad)
    * mixed_terms  : list[MixedTerm]  = H_AV + H_WT  (unmultiplied F (x) B)

and convert it into our `MixedH` (a list of `OperatorTerm`s with compact,
contiguous fermion-mode indices). This keeps the classical path bit-for-bit
consistent with the quantum path's Hamiltonian.

Term/index conventions match OpenFermion:
  * Each ladder op is (index, action) with action 1 = creation (dagger),
    0 = annihilation.
  * A term applies its ops right-to-left to a ket.
  * Fermion modes are compacted to 0..n_ferm_modes-1; boson modes are the
    global pion-mode indices (already contiguous 0..n_bos_modes-1).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class OperatorTerm:
    """coeff * (product of ferm_ops) * (product of bos_ops).

    ferm_ops / bos_ops are tuples of (mode_index, action) with
    action in {0: annihilate, 1: create}, ordered as written (applied
    right-to-left to a ket). Empty ferm_ops and bos_ops => constant term.
    """
    coeff: complex
    ferm_ops: tuple = ()
    bos_ops: tuple = ()


@dataclass
class MixedH:
    terms: list                      # list[OperatorTerm]
    n_ferm_modes: int
    n_bos_modes: int
    N_f: int                         # per-mode boson Fock cutoff (default 2**n_b;
                                     # may be any int classically — see build_from_eft)
    meta: dict = field(default_factory=dict)

    def constant(self):
        """Sum of pure-constant (identity) term coefficients."""
        return sum(t.coeff for t in self.terms if not t.ferm_ops and not t.bos_ops)

    def fermion_only_terms(self):
        return [t for t in self.terms if not t.bos_ops]

    def boson_only_terms(self):
        return [t for t in self.terms if not t.ferm_ops]

    def mixed_only_terms(self):
        return [t for t in self.terms if t.ferm_ops and t.bos_ops]


def _remap(ops, index_map):
    """Remap OpenFermion ladder-op index tuple to compact indices."""
    return tuple((index_map[i], a) for (i, a) in ops)


def from_mixed_hamiltonian(mh, n_b, N_f=None):
    """Convert a quantum-pipeline `MixedHamiltonian` into a `MixedH`.

    Args:
        mh: MixedHamiltonian (from build_eft_hamiltonian, fock+sparse path).
        n_b: qubits per pion mode. Sets the DEFAULT per-mode Fock cutoff
             N_f = 2**n_b (the quantum-side convention: whole qubits).
        N_f: optional explicit per-mode Fock cutoff. Classically the boson
             register is an occupation-number integer (see `state.MixedState`
             / `hij._apply_boson_ops`: `a|n>=√n|n-1>`, a 1-sparse map), so the
             cutoff need NOT be a power of two — that is a quantum-hardware
             constraint, not a classical one. Passing e.g. N_f=6 keeps the same
             ladder operators and truncates the box at n=5. The Hamiltonian
             TERM LIST (coeffs/ops, after the fermion-index compaction below) is
             cutoff-independent; N_f only enters at apply-time, so overriding it
             is exact — no rebuild needed. Defaults to 2**n_b.
    """
    if N_f is None:
        N_f = 2 ** n_b
    # --- compact fermion-mode index map ------------------------------------
    ferm_indices = set()
    for term in mh.fermion_part.terms:
        for (i, _a) in term:
            ferm_indices.add(i)
    for mt in mh.mixed_terms:
        for term in mt.fermion_factor.terms:
            for (i, _a) in term:
                ferm_indices.add(i)
    index_map = {orig: k for k, orig in enumerate(sorted(ferm_indices))}
    n_ferm_modes = len(index_map)

    n_bos_modes = len(mh.mode_to_qubits)

    terms = []

    # --- pure fermion (H_free + contact) -----------------------------------
    for ops, coeff in mh.fermion_part.terms.items():
        terms.append(OperatorTerm(complex(coeff), _remap(ops, index_map), ()))

    # --- pure boson (H_pion_free: number + zero-point + gradient) ----------
    for ops, coeff in mh.boson_part.terms.items():
        terms.append(OperatorTerm(complex(coeff), (), tuple(ops)))

    # --- mixed (H_AV, H_WT): distribute coeff * sum(F) * sum(B) -------------
    for mt in mh.mixed_terms:
        for f_ops, f_c in mt.fermion_factor.terms.items():
            for b_ops, b_c in mt.boson_factor.terms.items():
                terms.append(OperatorTerm(
                    complex(mt.coeff) * complex(f_c) * complex(b_c),
                    _remap(f_ops, index_map),
                    tuple(b_ops),
                ))

    meta = {
        "n_ferm_modes": n_ferm_modes,
        "n_bos_modes": n_bos_modes,
        "n_b": n_b,
        "N_f": N_f,
        "fermion_index_map": index_map,
        "n_terms": len(terms),
    }
    return MixedH(terms=terms, n_ferm_modes=n_ferm_modes,
                  n_bos_modes=n_bos_modes, N_f=N_f, meta=meta)


def build_from_eft(L, dim, n_b, params=None, transform="bare", frame_params=None,
                   N_f=None):
    """Build the full mixed EFT Hamiltonian as a `MixedH`, optionally in a compacting
    FRAME (task 33). `transform="bare"` is the native basis (default); a frame returns
    `Ū = U†HU` as a new term list the solver consumes unchanged.

    Args:
        L, dim: lattice side and spatial dimension.
        n_b: qubits per pion mode (default cutoff N_f = 2**n_b). Pass a SMALL value
             (e.g. 2 -> N_f=4) for ED-tractable toy systems; the physical
             cutoff from estimate_boson_cutoff() is larger.
        params: physical-parameter dict; defaults to get_physical_parameters().
        transform: frame axis — "bare" | "gaussian" (squeeze) | "LF" |
            "gaussian+LF" | "projector-LF" | "COO".
        frame_params: kwargs for the frame (e.g. {"r": 0.4} for "gaussian").
        N_f: optional explicit per-mode Fock cutoff (need NOT be a power of two —
             the classical boson register is an occupation-number integer, so the
             power-of-two box is a quantum-hardware constraint we can drop here).
             Defaults to 2**n_b. Useful for finer convergence resolution and to
             push the exact Lanczos reference further (sector ~ N_f**n_bos, so a
             non-power-of-two N_f between 2**k and 2**(k+1) is reachable where the
             next power of two is not). See `from_mixed_hamiltonian`.

    Returns:
        MixedH (bare or framed).
    """
    # Imported lazily so `classical/` has no import-time dependency on the
    # quantum pipeline unless a Hamiltonian is actually built.
    from src_PI.hamiltonians.ConstructEFT import build_eft_hamiltonian
    from src_PI.hamiltonians.core.EFTParameters import get_physical_parameters
    from src_PI.utils.Config import Config

    if params is None:
        params = get_physical_parameters()
    cfg = Config(pion_basis="fock", block_encoder="sparse")
    bundle, q_count, num_sites = build_eft_hamiltonian(
        L, dim, n_b, 0.0, params, cfg
    )
    mh = bundle.sub_hamiltonians[0].operator
    H = from_mixed_hamiltonian(mh, n_b, N_f=N_f)
    H.meta.update({"L": L, "dim": dim, "num_sites": num_sites,
                   "q_count": q_count})

    if transform in (None, "bare"):
        return H
    fp = dict(frame_params or {})
    from . import frame                     # deferred: avoids circular import
    if transform == "gaussian":
        # frame_params={"bogoliubov": True} -> full multi-mode Bogoliubov (STEP 2b,
        # also kills cross-mode pairs); else the per-mode squeeze (STEP 1/2).
        if fp.pop("bogoliubov", False):
            return frame.multimode_squeeze_terms(H, **fp)
        return frame.squeeze_terms(H, **fp)
    if transform == "LF":
        # Lang-Firsov polaron frame (STEP 3). frame_params: lambdas (required; scalar
        # or per-boson-mode), plus optional gen / fc_dress / order. EXACT for a density
        # vertex; leading-order for our transition vertex (-> projector-conditioned STEP 4).
        if "lambdas" not in fp:
            raise ValueError("transform='LF' needs frame_params={'lambdas': ...}")
        return frame.displace_terms(H, **fp)
    if transform == "gaussian+LF":
        # layered squeeze ∘ displace (STEP 4). frame_params: r/phi (squeeze), lambdas
        # (displace), optional gen/fc_dress/order/squeeze_first.
        return frame.combined_frame_terms(H, **fp)
    if transform == "projector-LF":
        # Variational projector-conditioned Lang-Firsov (STEP 4). The commuting n̂_p
        # projectors keep the displacement exact/finite even for our transition vertex.
        # frame_params:
        #   {"n_elec": A}                 -> analytic polaron seed + TrimCI-optimized
        #                                    global scale (the variational amplitude).
        #   {"combined": True} (default)  -> squeeze (analytic) FIRST, then projector-LF
        #                                    seeded from the squeezed coupling; else pure.
        #   {"entries":[...], "scale": s} -> explicit amplitude, skip the optimize scan.
        #   optional: fc_dress, order, core (optimizer fixed-core size).
        combined = fp.pop("combined", True)
        Hbase = H
        if combined:
            r, phi = frame.analytic_squeeze(H)
            Hbase = frame.squeeze_terms(H, -r, phi)
        if "entries" in fp and "scale" in fp:
            gen = frame.projector_generator(fp["entries"])
            out = frame.displace_terms(Hbase, lambdas=fp["scale"], gen=gen,
                                       fc_dress=fp.get("fc_dress"), order=fp.get("order", 4))
        else:
            if "n_elec" not in fp:
                raise ValueError("transform='projector-LF' needs frame_params={'n_elec': A}"
                                 " (or explicit {'entries','scale'})")
            best = frame.optimize_displacement(Hbase, fp["n_elec"],
                                               fc_dress=fp.get("fc_dress"),
                                               order=fp.get("order", 4),
                                               core=fp.get("core", 2000))
            out = frame.displace_terms(Hbase, lambdas=best["scale"], gen=best["gen"],
                                       fc_dress=fp.get("fc_dress"), order=fp.get("order", 4))
            out.meta["frame"] = {"transform": "projector-LF", "combined": bool(combined),
                                 "scale": best["scale"], "opt_energy": best["energy"],
                                 "seed_info": best.get("info")}
        return out
    if transform == "COO":
        # Core-Optimized Orbitals = fermion orbital rotation (STEP 5). frame_params:
        # {"R": ...} or {"kappa": ...} for an explicit rotation; {"natural": True,
        # "n_elec": A} for the ED 1-RDM natural-orbital basis; {"natural_smallcutoff":
        # True, "n_elec": A, "seed_n_b": 1} to seed R from a cheap small-cutoff ED slice;
        # {"natural_core": True, "coeffs","ferm_arr","bos_arr"} to seed R from a TrimCI
        # core 1-RDM (ED-free, scalable to large A — pass a pre-solved bare result).
        if fp.pop("natural", False):
            return frame.natural_orbital_terms(H, fp["n_elec"])
        if fp.pop("natural_smallcutoff", False):
            R, _ = frame.natural_orbitals_smallcutoff(
                L, dim, fp["n_elec"], seed_n_b=fp.get("seed_n_b", 1), params=params)
            return frame.rotate_orbitals_terms(H, R=R)
        if fp.pop("natural_core", False):
            R, _ = frame.natural_orbitals_from_core(
                H, fp["coeffs"], fp["ferm_arr"], fp["bos_arr"])
            return frame.rotate_orbitals_terms(H, R=R)
        return frame.rotate_orbitals_terms(H, **fp)
    raise ValueError(f"unknown transform {transform!r} "
                     "(bare|gaussian|LF|gaussian+LF|projector-LF|COO)")
