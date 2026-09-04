"""Analytic padding-law projection of the n_b=3 walk_T to L=8..10 (re-audit L=7 follow-up).

WHY this replaces the naive per-step ratio: pyLIQTR pads the PauliLCU PREPARE oracle to a power of
two, so the walk_T is a STEP function of L, quantized in the padded LCU size
    P(L) = 2**ceil(log2(n_terms(L))) ,   rotations = 2*P .
The compiled anchor shows this exactly: L=6 (1.15M terms) and L=7 (1.85M terms) BOTH fall in the
2^20..2^21 bin, so they share an identical padded PREPARE -> identical walk_T. A per-step total-T
RATIO to the n_b=2 anchor therefore JITTERS at bin boundaries (L=7 ratio 16.3 vs the smooth ~33 at
L=3..6) whenever the two step functions momentarily mis-align. The lambda ratio (x3.75) is smooth and
physical; only walk_T needs the quantization-aware model.

The model (fit on the compiled L=4..7, then back-tested on those same L before projecting):
    n_terms(L)/L^3 = c0 + c1/L                 (surface-corrected extensive term count)
    b   = (b/P) * P,     b/P constant           (rotation-synthesis slope; rotations = 2*P)
    a   = (a/P) * P,     a/P = m*log2(P) + k     (SELECT/QROM base; ~log in the index size)
    walk_T, total_T = optimize_qpe_fraction(a, b, lambda3, dE)   -- the SAME optimizer the shards use
with lambda3(L) = <lambda3/lambda2 over L=4..6> * lambda2(L)  (lambda2 is COMPILED at L=8..10).

Band: the only real projection risk is the bin assignment near a power-of-2 boundary, so the band is
walk_T evaluated at the bins spanned by n_terms * (1 +/- TERM_UNC). L=9 sits just under the 2^22 edge,
so it carries a genuinely wide band; L=10 is comfortably inside 2^23.

    python -m misc.nb3_padding_model          # back-test + projection table
"""
import glob
import json
import math

from src_PI.estimation.qpe_cost import walk_queries, WALK_QUERY_CONSTANT_HEISENBERG as PI
from src_PI.estimation.total_t_optimizer import optimize_qpe_fraction

TERM_UNC = 0.10   # +/-10% extrapolation uncertainty on n_terms(L) -> spans the neighbouring bin


def _load(d, patt):
    o = {}
    for f in glob.glob(f"{d}/{patt}"):
        j = json.load(open(f))
        if not (j.get("done") and j.get("results")) or "rep2" in f:
            continue
        r = j["results"][0]
        b = r.get("QPE_Budget") or {}
        o[r["L"]] = dict(lam=r["Physical_Lambda"], walkT=r["Walk_T_Count"], q=r["Logical_Qubits"],
                         terms=r.get("Pauli_Term_Count"), eps=b.get("eps_qpe"),
                         a=(b.get("walk_T_fit") or {}).get("a"), bb=(b.get("walk_T_fit") or {}).get("b"),
                         dE=b.get("delta_E", 1.0))
    return o


def _bin(n_terms):
    return 2 ** math.ceil(math.log2(n_terms))


def fit_model(nb3, fit_ls=(4, 5, 6, 7)):
    """Fit n_terms(L), b/P, a/P(log2 P) on the compiled asymptotic L. Returns a dict of params."""
    fit_ls = [L for L in fit_ls if L in nb3]
    # n_terms/L^3 = c0 + c1/L  (2-param LS over the fit L)
    xs = [1.0 / L for L in fit_ls]
    ys = [nb3[L]["terms"] / L ** 3 for L in fit_ls]
    n = len(xs); sx = sum(xs); sy = sum(ys); sxx = sum(x * x for x in xs); sxy = sum(x * y for x, y in zip(xs, ys))
    c1 = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    c0 = (sy - c1 * sx) / n
    # b/P constant; a/P = m*log2(P) + k  (LS over the fit L)
    bps = [nb3[L]["bb"] / _bin(nb3[L]["terms"]) for L in fit_ls]
    bP = sum(bps) / len(bps)
    lp = [math.log2(_bin(nb3[L]["terms"])) for L in fit_ls]
    ap = [nb3[L]["a"] / _bin(nb3[L]["terms"]) for L in fit_ls]
    n = len(lp); sx = sum(lp); sy = sum(ap); sxx = sum(x * x for x in lp); sxy = sum(x * y for x, y in zip(lp, ap))
    m = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    k = (sy - m * sx) / n
    return dict(c0=c0, c1=c1, bP=bP, m=m, k=k, fit_ls=fit_ls)


def n_terms_of(mp, L):
    return (mp["c0"] + mp["c1"] / L) * L ** 3


def walk_T_at_bin(mp, P, lam, dE):
    """(a,b) from the padding model at padded size P, then the SAME optimizer the shards run."""
    b = mp["bP"] * P
    a = (mp["m"] * math.log2(P) + mp["k"]) * P
    opt = optimize_qpe_fraction(a, b, lam, dE)
    return opt["walk_T"], opt["total_T"], a, b


def project(nb2, nb3, lam_ratio=None, dE=1.0):
    """Project walk_T / total_T / qubits for every L in nb2, using compiled n_b=3 where present and the
    padding model (with a bin-uncertainty band) where not. lam3 = lam_ratio * lam2 (compiled lam2)."""
    mp = fit_model(nb3)
    ls46 = [L for L in (4, 5, 6) if L in nb3 and L in nb2]
    if lam_ratio is None:
        lam_ratio = sum(nb3[L]["lam"] / nb2[L]["lam"] for L in ls46) / len(ls46)
    qr = sum(nb3[L]["q"] / nb2[L]["q"] for L in ls46) / len(ls46)
    rows = {}
    for L in sorted(nb2):
        if L in nb3:                                         # compiled
            eps = nb3[L]["eps"]
            nw = walk_queries(nb3[L]["lam"], eps, PI) if eps else None
            T = nw * nb3[L]["walkT"] if nw else None
            rows[L] = dict(L=L, exact=True, lam=nb3[L]["lam"], walkT=nb3[L]["walkT"], q=nb3[L]["q"],
                           T=T, Tlo=T, Thi=T, terms=nb3[L]["terms"], P=_bin(nb3[L]["terms"]))
        else:                                                # padding-model projection
            nt = n_terms_of(mp, L)
            lam3 = lam_ratio * nb2[L]["lam"]
            P = _bin(nt)
            wT, T, _a, _b = walk_T_at_bin(mp, P, lam3, dE)
            # band: bins spanned by n_terms*(1 +/- TERM_UNC)
            Plo, Phi = _bin(nt * (1 - TERM_UNC)), _bin(nt * (1 + TERM_UNC))
            Ts = [walk_T_at_bin(mp, PP, lam3, dE)[1] for PP in sorted({Plo, P, Phi})]
            rows[L] = dict(L=L, exact=False, lam=lam3, walkT=wT, q=nb2[L]["q"] * qr, T=T,
                           Tlo=min(Ts), Thi=max(Ts), terms=nt, P=P)
    return rows, dict(mp=mp, lam_ratio=lam_ratio, qr=qr, ls46=ls46)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--nb2", default="data/quantum/2026-08-21/vertexfix_r3_290826")
    ap.add_argument("--nb3", default="data/quantum/nb3_anchor")
    args = ap.parse_args()
    nb2 = _load(args.nb2, "*fock_pauli*nb2*.json")
    nb3 = _load(args.nb3, "*fock_pauli_nb3*.json")
    mp = fit_model(nb3)
    print(f"model: n_terms/L^3 = {mp['c0']:.1f} {mp['c1']:+.1f}/L ; b/P = {mp['bP']:.4f} ; "
          f"a/P = {mp['m']:.3f}*log2(P) {mp['k']:+.2f}  (fit L={mp['fit_ls']})\n")

    # BACK-TEST: predict walk_T for the compiled L from the model (using each L's ACTUAL lambda), compare.
    print("BACK-TEST (model vs compiled, using compiled lambda):")
    print(f"{'L':>2} {'n_terms(fit)':>12} {'n_terms(true)':>13} {'bin':>5} "
          f"{'walkT_pred':>11} {'walkT_true':>11} {'err%':>6}")
    for L in sorted(nb3):
        nt_fit = n_terms_of(mp, L); nt_true = nb3[L]["terms"]
        P = _bin(nt_true)                                    # use TRUE bin to isolate the (a,b) model
        wT, _T, _a, _b = walk_T_at_bin(mp, P, nb3[L]["lam"], nb3[L]["dE"])
        err = 100 * (wT - nb3[L]["walkT"]) / nb3[L]["walkT"]
        print(f"{L:>2} {nt_fit:>12.0f} {nt_true:>13} {int(math.log2(P)):>5} "
              f"{wT:>11.3e} {nb3[L]['walkT']:>11.3e} {err:>+6.1f}")

    rows, sc = project(nb2, nb3)
    print("\nPROJECTION (padding model, L>7):")
    print(f"{'L':>2} {'src':>9} {'n_terms':>10} {'bin':>4} {'lambda':>10} {'T':>10} {'band':>23}")
    for L in sorted(rows):
        r = rows[L]
        band = "-" if r["exact"] else f"[{r['Tlo']:.2e}, {r['Thi']:.2e}]"
        print(f"{L:>2} {'compiled' if r['exact'] else 'padding':>9} {r['terms']:>10.0f} "
              f"{int(math.log2(r['P'])):>4} {r['lam']:>10.2e} {r['T']:>10.2e} {band:>23}")
    r10 = rows[10]
    print(f"\nL=10 headline: T = {r10['T']:.2e}  band [{r10['Tlo']:.1e}, {r10['Thi']:.1e}]  "
          f"qubits {r10['q']:.0f}")


if __name__ == "__main__":
    main()
