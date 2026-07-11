"""
D-NQS: neural quantum states vs selected CI on the mixed fermion-boson H (NetKet).

NQS is the one surveyed method that could, in principle, overtake selected CI at
3D scale (see `claude/research/classical_baselines/other_methods/`): variational,
mixed-Fock, no boson truncation, handles both on-site (Holstein/WT-like) and bond
(SSH/AV-like) couplings. The open question is whether a NuQu-specific mixed-Fock
NQS actually MATCHES TrimCI's energies at fewer wall-clock-seconds and scales
better -- if so, we switch the baseline; if it needs more parameters or stalls in
the strong-SSH regime our H_AV inhabits, that confirms TrimCI.

We wire our EXACT mixed H into NetKet by wrapping the project's validated
`connections` oracle (`classical.trimci.hij`) as a custom NetKet DiscreteOperator
-- so the NQS optimizes the *same* Hamiltonian block2/TrimCI/Lanczos use, with no
re-derivation. Hilbert space: SpinOrbitalFermions(n_ferm, A) (x) Fock(N_f-1, n_bos);
a config is [fermion occ (0/1) x n_ferm | boson occ (0..N_f-1) x n_bos].

Correctness gate: NetKet exact diagonalization of the wrapped operator must equal
the project's Lanczos ground energy.
"""

from __future__ import annotations

import numpy as np


def make_hilbert(n_ferm, n_bos, N_f, A):
    import netket as nk
    fh = nk.hilbert.SpinOrbitalFermions(n_orbitals=n_ferm, n_fermions=A)
    bh = nk.hilbert.Fock(n_max=N_f - 1, N=n_bos)
    return fh * bh


def _build_operator_class():
    """Deferred so importing this module doesn't require netket."""
    import netket as nk
    from classical.trimci.hij import connections_nocache
    from classical.trimci.state import MixedState

    class MixedHOperator(nk.operator.DiscreteOperator):
        """Custom NetKet operator wrapping the project's mixed-H connections oracle."""

        def __init__(self, hilbert, H, n_ferm, n_bos, max_conn):
            super().__init__(hilbert)
            self._H = H
            self._n_ferm = n_ferm
            self._n_bos = n_bos
            self._max_conn = int(max_conn)

        @property
        def dtype(self):
            return complex

        @property
        def is_hermitian(self):
            return True          # build_from_eft H is Hermitian (real Lanczos spectrum)

        @property
        def max_conn_size(self):
            return self._max_conn

        def _to_state(self, xv):
            ferm = 0
            for m in range(self._n_ferm):
                if xv[m] > 0.5:
                    ferm |= (1 << m)
            bos = tuple(int(round(float(xv[self._n_ferm + m])))
                        for m in range(self._n_bos))
            return MixedState(ferm, bos)

        def _from_state(self, st, out):
            f = st.ferm
            for m in range(self._n_ferm):
                out[m] = 1.0 if (f >> m) & 1 else 0.0
            for m in range(self._n_bos):
                out[self._n_ferm + m] = st.bos[m]

        def get_conn_padded(self, x):
            xa = np.asarray(x)
            lead = xa.shape[:-1]
            size = self.hilbert.size
            xr = xa.reshape(-1, size)
            B = xr.shape[0]
            per = [connections_nocache(self._H, self._to_state(xr[b])) for b in range(B)]
            width = max(self._max_conn, max((len(c) for c in per), default=1))
            xp = np.zeros((B, width, size), dtype=xa.dtype)
            mels = np.zeros((B, width), dtype=complex)
            for b in range(B):
                xp[b, :] = xr[b]                            # pad with self (0 mel)
                for i, (si, amp) in enumerate(per[b].items()):
                    self._from_state(si, xp[b, i])
                    mels[b, i] = np.conj(amp)               # row: <x|H|x'>=conj(<x'|H|x>)
            return (xp.reshape(lead + (width, size)),
                    mels.reshape(lead + (width,)))

        def get_conn_flattened(self, x, sections):
            xr = np.asarray(x).reshape(-1, self.hilbert.size)
            B = xr.shape[0]
            xps, mls = [], []
            total = 0
            for b in range(B):
                st = self._to_state(xr[b])
                conns = connections_nocache(self._H, st)
                for si, amp in conns.items():
                    row = np.zeros(self.hilbert.size, dtype=xr.dtype)
                    self._from_state(si, row)
                    xps.append(row)
                    mls.append(np.conj(amp))
                    total += 1
                sections[b] = total
            if total == 0:
                return (np.zeros((0, self.hilbert.size), dtype=xr.dtype),
                        np.zeros((0,), dtype=complex))
            return np.array(xps), np.array(mls, dtype=complex)

    return MixedHOperator


def estimate_max_conn(H, hilbert, n_ferm, n_bos, n_samples=300, seed=0):
    """Max number of connected states over a random sample (+ margin)."""
    import netket as nk
    from classical.trimci.hij import connections_nocache
    from classical.trimci.state import MixedState
    xs = np.asarray(hilbert.random_state(nk.jax.PRNGKey(seed), (n_samples,)))
    mx = 1
    for xv in xs:
        ferm = 0
        for m in range(n_ferm):
            if xv[m] > 0.5:
                ferm |= (1 << m)
        bos = tuple(int(round(float(xv[n_ferm + m]))) for m in range(n_bos))
        mx = max(mx, len(connections_nocache(H, MixedState(ferm, bos))))
    return mx + 4


def build(L, dim, A, N_f=2, n_b=1, scales=None):
    """(hilbert, operator, H) for the mixed EFT H at (L,dim,A,N_f).

    `scales` (dict) optionally toggles term groups via sign_structure.scaled_terms
    (e.g. {'wt': 0.0} to drop Weinberg-Tomozawa) -- used to test whether the
    sign/phase structure is what stalls the NQS.
    """
    from classical.trimci import build_from_eft
    H = build_from_eft(L=L, dim=dim, n_b=n_b, N_f=N_f)
    if scales:
        from classical.baselines.sign_structure import scaled_terms
        H = scaled_terms(H, **scales)
    n_ferm, n_bos = H.n_ferm_modes, H.n_bos_modes
    hi = make_hilbert(n_ferm, n_bos, N_f, A)
    max_conn = estimate_max_conn(H, hi, n_ferm, n_bos)
    Op = _build_operator_class()
    op = Op(hi, H, n_ferm, n_bos, max_conn)
    return hi, op, H


def validate_vs_lanczos(L=2, dim=1, A=2, N_f=2, tol=1e-5):
    """Correctness gate: NetKet exact-diag of the wrapped operator == Lanczos."""
    import netket as nk
    from classical.trimci import build_from_eft
    from classical.trimci.lanczos import lanczos_ground_state
    hi, op, H = build(L, dim, A, N_f=N_f)
    E_lanc, _ = lanczos_ground_state(H, A)
    E_nk = nk.exact.lanczos_ed(op, k=1)[0]
    ok = abs(E_nk - E_lanc) < tol
    print(f"  L={L} d={dim} A={A} N_f={N_f}: Lanczos={E_lanc:.6f}  "
          f"NetKet-ED={E_nk:.6f}  diff={abs(E_nk-E_lanc):.2e}  "
          f"{'OK' if ok else 'MISMATCH'}")
    return ok


def run_vmc(L, dim, A, N_f=2, alpha=2, n_samples=4096, n_iter=300, lr=0.05,
            diag_shift=0.01, seed=0, verbose=True, scales=None):
    """Optimize an RBM NQS on the mixed H via VMC (SR).

    Indexable Hilbert (small) -> FullSumState (deterministic exact expectation,
    the clean capability/validation regime). Otherwise -> MCState with a local
    Metropolis sampler (the oracle runs host-side; this is where the non-JAX
    connections oracle makes NQS slow -- itself a finding).
    """
    import time
    import netket as nk
    hi, op, H = build(L, dim, A, N_f=N_f, scales=scales)

    ma = nk.models.RBM(alpha=alpha, param_dtype=complex)
    exact = hi.is_indexable
    if exact:
        vs = nk.vqs.FullSumState(hi, ma, seed=seed)
    else:
        sa = nk.sampler.MetropolisLocal(hi, n_chains=16)
        vs = nk.vqs.MCState(sa, ma, n_samples=n_samples, seed=seed)
    n_params = vs.n_parameters
    opt = nk.optimizer.Sgd(learning_rate=lr)
    sr = nk.optimizer.SR(diag_shift=diag_shift)
    gs = nk.VMC(op, opt, variational_state=vs, preconditioner=sr)

    hist = []
    t0 = time.time()
    for it in range(n_iter):
        gs.advance()
        e = complex(gs.energy.mean)
        hist.append(e.real)
        if verbose and (it % max(1, n_iter // 10) == 0 or it == n_iter - 1):
            sig = getattr(gs.energy, 'error_of_mean', 0.0)
            print(f"    it={it:4d}  E={e.real:.4f}  (sigma={sig})", flush=True)
    wall = time.time() - t0
    e_best = float(np.min(hist[-max(10, n_iter // 10):]))
    return {'L': L, 'dim': dim, 'A': A, 'N_f': N_f, 'alpha': alpha,
            'mode': 'fullsum' if exact else 'mcmc',
            'n_params': int(n_params), 'E_final_mean': float(np.mean(hist[-10:])),
            'E_best': e_best, 'wall_s': wall, 'history': hist}


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--validate', action='store_true')
    p.add_argument('--vmc', action='store_true')
    p.add_argument('--alpha', type=int, default=2)
    p.add_argument('--n_iter', type=int, default=300)
    p.add_argument('--L', type=int, default=2)
    p.add_argument('--dim', type=int, default=1)
    p.add_argument('--A', type=int, default=2)
    p.add_argument('--N_f', type=int, default=2)
    args = p.parse_args()
    if args.validate:
        validate_vs_lanczos(args.L, args.dim, args.A, args.N_f)
    elif args.vmc:
        r = run_vmc(args.L, args.dim, args.A, N_f=args.N_f, alpha=args.alpha,
                    n_iter=args.n_iter)
        print(f"  NQS: E_best={r['E_best']:.4f}  n_params={r['n_params']}  "
              f"wall={r['wall_s']:.1f}s")
