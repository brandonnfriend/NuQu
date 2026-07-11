"""
D-DMRG (part 1): the rigorous, exact bond-dimension lower bound.

A DMRG/MPS ground state needs, across every bond of its 1D ordering, a bond
dimension chi >= the Schmidt rank of the exact state across that cut; to reach
truncation error eps you need chi >= chi*(eps) = #(Schmidt values holding weight
1-eps). The von Neumann entropy S of the cut sets chi ~ e^S. This is an
INFORMATION-THEORETIC LOWER BOUND: DMRG cannot beat it (finite sweeps / local
minima only make it worse). So if chi*(eps) across a lattice bisection grows with
the cut area, the DMRG cost is provably walled off -- no cleverness in the DMRG
code escapes it.

We compute chi*(eps) and S EXACTLY from the project's own Lanczos ground state,
across a planar spatial bisection of the L^dim lattice (interface area L^(D-1) =
1, 2, 4 for D=1,2,3 at L=2). Reachable exactly for small sectors (L=2 d=1,2;
L=3 d=1). The area law S ~ (cut area) then PROJECTS the 3D blow-up chi ~ e^{a L^2}
that block2 confirms at L=2^3 (see dmrg_block2.py) and that is out of exact reach
at L>=3 3D -- which is exactly the wall.

Fermionic correctness: splitting a Slater determinant across a spatial cut incurs
a Jordan-Wigner reordering sign that depends jointly on the A- and B-occupations;
we apply it (see `_reorder_sign`) so the singular values are the true Schmidt
coefficients. Bosons commute -> no sign.
"""

from __future__ import annotations

import numpy as np


def ground_state(L, dim, A, N_f=2, n_b=1, max_states=800000):
    """Exact Lanczos ground state as {MixedState: amplitude} + the MixedH."""
    from classical.trimci import build_from_eft
    from classical.trimci.lanczos import lanczos_ground_state
    H = build_from_eft(L=L, dim=dim, n_b=n_b, N_f=N_f)
    E0, vec, info = lanczos_ground_state(H, A, return_vec=True, max_states=max_states)
    return E0, vec, H


def mode_site_maps(H, L, dim):
    """(ferm_mode->site, bos_mode->site). Fermion modes: 4/site, contiguous after
    the builder's compaction (offset 0..3 < stride) => site = m//4. Boson (pion)
    modes: 3 species/site, contiguous => site = m//3."""
    num_sites = L ** dim
    assert H.n_ferm_modes == 4 * num_sites, \
        f"expected 4*{num_sites} fermion modes, got {H.n_ferm_modes}"
    assert H.n_bos_modes == 3 * num_sites, \
        f"expected 3*{num_sites} boson modes, got {H.n_bos_modes}"
    fmap = np.arange(H.n_ferm_modes) // 4
    bmap = np.arange(H.n_bos_modes) // 3
    return fmap, bmap


def bipartition(L, dim):
    """Planar bisection along axis 0: region A = sites with coord[0] < L//2 (or the
    single low site if L is odd). Returns (A_sites set, cut_area = # crossing bonds)."""
    from src_PI.utils.LatticeGeometry import index_to_coord, get_total_sites
    num_sites = get_total_sites(L, dim)
    cut = L // 2 if L > 1 else 1
    A = set(x for x in range(num_sites) if index_to_coord(x, L, dim)[0] < cut)
    # crossing bonds: nearest-neighbor pairs straddling coord0 == cut-1 | cut
    cut_area = sum(
        1 for x in range(num_sites)
        if index_to_coord(x, L, dim)[0] == cut - 1
        and (x + L ** 0) < num_sites
        and index_to_coord(x + 1, L, dim)[0] == cut
    ) if dim >= 1 else 1
    # for a clean planar cut the area is L^(dim-1); use that (robust)
    cut_area = L ** (dim - 1)
    return A, cut_area


def _reorder_sign(ferm_bitmask, A_ferm_set):
    """JW sign to reorder occupied fermion modes into (A-occupied | B-occupied),
    from global increasing-index order. = (-1)^(# pairs a in A_occ, b in B_occ, b<a)."""
    inv = 0
    b_seen = 0
    m = 0
    bm = ferm_bitmask
    while bm:
        if bm & 1:
            if m in A_ferm_set:
                inv += b_seen           # B-occupied already passed with lower index
            else:
                b_seen += 1
        bm >>= 1
        m += 1
    return -1.0 if (inv & 1) else 1.0


def schmidt_values(vec, H, L, dim):
    """Exact Schmidt spectrum of the ground state across the planar bisection."""
    fmap, bmap = mode_site_maps(H, L, dim)
    A_sites, cut_area = bipartition(L, dim)
    A_ferm = set(int(m) for m in range(H.n_ferm_modes) if fmap[m] in A_sites)
    A_bos = [m for m in range(H.n_bos_modes) if bmap[m] in A_sites]
    B_bos = [m for m in range(H.n_bos_modes) if bmap[m] not in A_sites]

    rows, cols = {}, {}
    entries = []
    for st, amp in vec.items():
        f = st.ferm
        # split fermion occupation into A / B by mode
        fA = f & _mask(A_ferm)
        fB = f & ~_mask(A_ferm)
        bos = st.bos
        bA = tuple(bos[m] for m in A_bos)
        bB = tuple(bos[m] for m in B_bos)
        keyA = (fA, bA)
        keyB = (fB, bB)
        ia = rows.setdefault(keyA, len(rows))
        ib = cols.setdefault(keyB, len(cols))
        sign = _reorder_sign(f, A_ferm)
        entries.append((ia, ib, amp * sign))

    M = np.zeros((len(rows), len(cols)), dtype=complex)
    for ia, ib, v in entries:
        M[ia, ib] += v
    sv = np.linalg.svd(M, compute_uv=False)
    sv = sv[sv > 1e-14]
    sv = sv / np.sqrt((sv ** 2).sum())      # normalize Schmidt weights
    return sv, cut_area


def _mask(mode_set):
    m = 0
    for i in mode_set:
        m |= (1 << i)
    return m


def entanglement_report(sv, eps_list=(1e-1, 1e-2, 1e-3, 1e-6)):
    """von Neumann entropy + chi*(eps) = #Schmidt values to hold weight 1-eps."""
    p = sv ** 2
    p = p[p > 1e-15]
    S = float(-(p * np.log(p)).sum())
    tail = np.cumsum(p[::-1])[::-1]          # discarded weight if we keep i..end
    chi = {}
    ps = np.sort(p)[::-1]                    # descending Schmidt weights
    csum = np.cumsum(ps)
    for eps in eps_list:
        # smallest k such that the kept top-k hold weight >= 1 - eps
        k = int(np.searchsorted(csum, 1 - eps) + 1)
        chi[eps] = min(k, len(ps))
    return {'S_vonNeumann': S, 'schmidt_rank': int(len(sv)), 'chi_star': chi}


def run(points=((2, 1, 2), (2, 2, 2), (3, 1, 2)), N_f=2, save=True):
    """Compute exact bisection entanglement across geometries; area-law scaling."""
    print("D-DMRG (exact entanglement lower bound on chi):\n")
    print(f"  {'L':>2} {'dim':>3} {'A':>2} {'cut_area':>8} {'S':>7} {'S/area':>7}  "
          f"chi*(1e-1/1e-2/1e-3/1e-6)")
    rows = []
    for (L, dim, A) in points:
        try:
            E0, vec, H = ground_state(L, dim, A, N_f=N_f)
        except MemoryError as e:
            print(f"  {L:>2} {dim:>3} {A:>2}   (skipped: {str(e)[:50]})")
            continue
        sv, area = schmidt_values(vec, H, L, dim)
        rep = entanglement_report(sv)
        chi = rep['chi_star']
        rows.append({'L': L, 'dim': dim, 'A': A, 'cut_area': area, 'E0': E0, **rep})
        print(f"  {L:>2} {dim:>3} {A:>2} {area:>8} {rep['S_vonNeumann']:7.3f} "
              f"{rep['S_vonNeumann']/area:7.3f}  "
              f"{chi[1e-1]:>3}/{chi[1e-2]:>3}/{chi[1e-3]:>3}/{chi[1e-6]:>3}")
    if save and rows:
        _save(rows, N_f)
    return rows


def _save(rows, N_f):
    import json, os, datetime as _dt
    date = _dt.date.today().isoformat()
    outdir = f"data/classical/baselines/{date}"
    os.makedirs(outdir, exist_ok=True)
    with open(f"{outdir}/dmrg_entanglement.json", 'w') as f:
        json.dump({'N_f': N_f, 'rows': rows}, f, indent=2, default=str)
    print(f"\nSaved: {outdir}/dmrg_entanglement.json")


if __name__ == '__main__':
    run()
