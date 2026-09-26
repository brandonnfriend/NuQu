"""Seed control (F-009) verdict: is the low-occupation prior a circular argument?

Reads the two MATCHED arms produced by `misc.run_nb_convergence {Dprior,Duniform}` and
compares them on the two observables the original study_D named — E_var and the mean
per-mode occupation N_per_mode — at every cutoff N_f.

WHAT A PASS MEANS, precisely. The arms differ ONLY in initialization: `Dprior` seeds the
core near vacuum, `Duniform` samples occupation uniformly over [0, N_f) with no vacuum
anchor, so it must FIND the near-vacuum ground state. If the uniform arm reproduces the
prior arm on both observables, the prior is a convergence ACCELERATOR, not an imposed
answer, and the occupation measurements that rest on it are not circular.

WHAT A FAIL DOES NOT MEAN. E_var is a valid Ritz upper bound under either
initialization, so a mismatch is a SEARCH-BASIN result, not a broken bound: it would say
the uniform ensemble failed to find the basin the prior starts in, which is a statement
about search budget as much as about bias. Because a too-narrow ensemble makes that
ambiguous, both arms run at n_runs=16 (the breadth the block2 cross-check showed is
needed to escape basin traps). The script REFUSES to issue a verdict if the arms were
not run at matched settings.

DIRECTION MATTERS. The uniform arm landing ABOVE the prior arm in energy is the benign
direction (it searched worse). The uniform arm landing BELOW would mean the prior was
costing us energy — the prior would be steering the search away from the true ground
state, which is the failure the control exists to detect.

    python -m misc.compare_seed_control --data data/classical/<date>/seed_control_<cid>
"""
import argparse
import glob
import json
import os

E_TOL_MEV = 1.0          # energy agreement target: the project's GSEE accuracy
N_REL_TOL = 0.10         # occupation agreement: 10% relative on <N>/mode


def load_groups(data_dirs):
    """{(L, n_runs): {arm: record}} — one matched comparison per (L, ensemble breadth)."""
    groups = {}
    for d in data_dirs:
        for f in sorted(glob.glob(os.path.join(d, "studyD*init.json"))):
            j = json.load(open(f))
            arm = j.get("seed_control_arm")
            if arm is None:                       # tolerate the retired pre-fix naming
                arm = "uniform" if "uniforminit" in f else "prior"
            cfg = j.get("seed_control_cfg") or {}
            key = (cfg.get("L", j.get("L")), cfg.get("n_runs", j.get("n_runs")))
            groups.setdefault(key, {})[arm] = {
                "rows": {r["N_f"]: r for r in j["rows"]},
                "cfg": cfg or None, "path": f}
    return groups


def load_arms(data_dirs):
    """Back-compat single-group loader (first group found)."""
    g = load_groups(data_dirs)
    return next(iter(g.values())) if g else {}


# --- the check that caught the n_b=3 under-convergence -----------------------

def subspace_violations(rows_by_nf):
    """Rows where E(N_f) EXCEEDS the best bound proved at a SMALLER N_f.

    The N_f=k Fock space is a SUBSPACE of N_f=k' for k<k' (truncate each mode to k vs
    k' levels), and the term list is N_f-independent (the cutoff only enters at
    apply-time). So any N_f=k trial state is a valid N_f=k' trial state with the SAME
    Rayleigh quotient, giving E_0(N_f=k') <= E_var(N_f=k). A larger-N_f solve that lands
    ABOVE a smaller-N_f one is therefore UNDER-CONVERGED by at least that much -- a
    rigorous, self-contained detector that needs no reference calculation.

    This is what exposed the L=2 n_b=3 result: E(N_f=8)=2437.74 sat 9.27 MeV above the
    E(N_f=4)=2428.47 bound, so its 0.001 MeV agreement with N_f=16 was correlated search
    error, not convergence.
    """
    out, best = [], None
    for nf in sorted(rows_by_nf):
        e = rows_by_nf[nf]["E_var"]
        if best is not None and e > best[1] + 1e-9:
            out.append(dict(N_f=nf, E=e, bound_from_N_f=best[0], bound=best[1],
                            excess=e - best[1]))
        if best is None or e < best[1]:
            best = (nf, e)
    return out


def settings_match(arms):
    """Both arms must differ ONLY in the init, or the comparison is confounded."""
    p, u = arms.get("prior"), arms.get("uniform")
    if not (p and u):
        return False, "missing an arm (need both prior and uniform)"
    cp, cu = p.get("cfg"), u.get("cfg")
    if cp is None or cu is None:
        return False, ("one arm predates the matched-config record (likely retired "
                       "pre-vertex-fix data) — cannot certify the arms are matched")
    diffs = {k: (cp.get(k), cu.get(k)) for k in set(cp) | set(cu) if cp.get(k) != cu.get(k)}
    if diffs:
        return False, f"arms not matched: {diffs}"
    return True, f"matched at {cp}"


def compare(arms):
    rows = []
    shared = sorted(set(arms["prior"]["rows"]) & set(arms["uniform"]["rows"]))
    for nf in shared:
        p, u = arms["prior"]["rows"][nf], arms["uniform"]["rows"][nf]
        dE = u["E_var"] - p["E_var"]                       # >0 = uniform searched worse
        np_, nu = p.get("N_per_mode"), u.get("N_per_mode")
        dN_rel = (abs(nu - np_) / np_) if (np_ and nu is not None and np_ > 0) else None
        rows.append(dict(N_f=nf, n_b=p.get("n_b"), E_prior=p["E_var"], E_uniform=u["E_var"],
                         dE=dE, N_prior=np_, N_uniform=nu, dN_rel=dN_rel,
                         E_ok=abs(dE) <= E_TOL_MEV,
                         N_ok=(dN_rel is not None and dN_rel <= N_REL_TOL)))
    return rows


def verdict(rows):
    if not rows:
        return "NO OVERLAP", "the two arms share no N_f"
    below = [r for r in rows if r["dE"] < -E_TOL_MEV]
    if below:
        return ("FAIL — PRIOR IS STEERING",
                f"the uniform arm reached LOWER energy at N_f={[r['N_f'] for r in below]}; "
                "the prior was costing energy, i.e. steering the search away from the "
                "ground state. This is the failure the control exists to detect.")
    bad = [r for r in rows if not (r["E_ok"] and r["N_ok"])]
    if not bad:
        return ("PASS", f"both arms agree within {E_TOL_MEV} MeV and "
                        f"{N_REL_TOL:.0%} on <N>/mode at every N_f — the prior "
                        "accelerates convergence, it does not impose the answer")
    return ("INCONCLUSIVE — SEARCH-LIMITED",
            f"the uniform arm sits ABOVE the prior arm at N_f={[r['N_f'] for r in bad]} "
            "(the benign direction: it searched worse, and E_var stays a valid upper "
            "bound). Raise n_runs / core before reading this as seeding bias.")


def pooled_best(arms):
    """Best (lowest) E_var per N_f across BOTH arms — they bound the SAME operator, so the
    tightest bound at each N_f is whichever arm found it. Pooling is what exposes a
    CROSS-ARM subspace violation: at L=2 the prior arm's N_f=8 sat 9.27 MeV above the
    uniform arm's N_f=4 bound, which no within-arm check can see."""
    out = {}
    for a in arms.values():
        for nf, r in a["rows"].items():
            if nf not in out or r["E_var"] < out[nf]["E_var"]:
                out[nf] = r
    return out


def unmatched_groups(groups):
    """Groups where only ONE arm has data, with that arm's rows.

    A missing arm is not a null result. In cluster 293942 both L=4 UNIFORM shards
    exceeded 96 GB and were held while both L=4 PRIOR shards finished, so the
    asymmetry is itself the finding: a uniform draw across 3*L^3 boson modes seeds
    high-occupation states whose expansion graph is far denser. Dropping these
    groups from the report would hide both the surviving arm's numbers and the
    infeasibility of its counterpart, so they are reported separately and WITHOUT a
    verdict (no comparison is possible).
    """
    out = []
    for key in sorted(groups, key=lambda k: (k[0] or 0, k[1] or 0)):
        arms = groups[key]
        if len(arms) >= 2:
            continue
        arm = next(iter(arms))
        rows = arms[arm]["rows"]
        out.append(dict(key=key, arm=arm, missing="uniform" if arm == "prior" else "prior",
                        rows=[rows[nf] for nf in sorted(rows)],
                        per_arm=subspace_violations(rows)))
    return out


def report_group(key, arms, fails):
    L, nr = key
    hdr = f"=== L={L}, n_runs={nr} " + "=" * 30
    print(f"\n{hdr}")
    ok, why = settings_match(arms)
    print(f"[match] {'OK' if ok else 'REFUSED'} — {why}")
    if not ok:
        fails.append((key, "unmatched", why))
        return None
    rows = compare(arms)
    print(f"{'N_f':>4} {'n_b':>4} {'E_prior':>12} {'E_uniform':>12} {'dE':>10} "
          f"{'<N>_pri':>9} {'<N>_uni':>9} {'dN_rel':>9}")
    print("-" * 78)
    for r in rows:
        dn = "—" if r["dN_rel"] is None else f"{r['dN_rel']:8.3%}"
        print(f"{r['N_f']:>4} {r['n_b']:>4} {r['E_prior']:12.4f} {r['E_uniform']:12.4f} "
              f"{r['dE']:+10.4f} {r['N_prior']:9.5f} {r['N_uniform']:9.5f} {dn:>9}")
    for arm in ("prior", "uniform"):
        v = subspace_violations(arms[arm]["rows"])
        if v:
            print(f"  [under-converged] {arm}: " + "; ".join(
                f"N_f={x['N_f']} is {x['excess']:.3f} MeV above the N_f={x['bound_from_N_f']} bound"
                for x in v))
    xv = subspace_violations(pooled_best(arms))
    if xv:
        print("  [CROSS-ARM under-convergence] " + "; ".join(
            f"N_f={x['N_f']} is {x['excess']:.3f} MeV above the N_f={x['bound_from_N_f']} bound "
            "proved by the other arm" for x in xv))
    v, why2 = verdict(rows)
    print(f"  VERDICT: {v} — {why2}")
    return dict(key=key, rows=rows, verdict=v, why=why2,
                cross_arm=xv,
                per_arm={a: subspace_violations(arms[a]["rows"]) for a in arms})


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", default=None, help="optional markdown output path")
    args = ap.parse_args()

    groups = load_groups(args.data)
    assert groups, "no studyD*init.json shards found"
    print(f"[groups] {len(groups)}: {sorted(groups)}")
    fails, results = [], []
    for key in sorted(groups, key=lambda k: (k[0] or 0, k[1] or 0)):
        arms = groups[key]
        if len(arms) < 2:
            print(f"\n=== L={key[0]}, n_runs={key[1]} === only {sorted(arms)} — "
                  "no comparison possible, reported as an unmatched arm")
            continue
        r = report_group(key, arms, fails)
        if r:
            results.append(r)

    unmatched = unmatched_groups(groups)
    for u in unmatched:
        L, nr = u["key"]
        print(f"\n=== L={L}, n_runs={nr} (UNMATCHED: {u['arm']} only, "
              f"{u['missing']} arm absent) ===")
        for x in u["rows"]:
            n = x.get("N_per_mode")
            print(f"{x['N_f']:>4} {x.get('n_b'):>4} {x['E_var']:12.4f} "
                  f"{'—' if n is None else f'{n:9.5f}'}")

    if args.out and (results or unmatched):
        md = ["# Seed control (F-009) — uniform vs near-vacuum initialization\n",
              "_Matched arms per (L, n_runs): identical settings except `boson_init_mean`. "
              "`E_var` is a valid Ritz upper bound under EITHER init, so a mismatch is a "
              "search-basin result, not a broken bound._\n",
              "_**Subspace check.** The N_f=k space is a SUBSPACE of N_f=k'>k and the term "
              "list is N_f-independent, so E_0(N_f=k') <= E_var(N_f=k). A larger-N_f solve "
              "sitting ABOVE a smaller-N_f bound is under-converged by at least that much — "
              "pooled across arms, since both bound the same operator._\n"]
        for r in results:
            L, nr = r["key"]
            md += [f"\n## L={L}, n_runs={nr}\n",
                   "| $N_f$ | $n_b$ | $E$ prior | $E$ uniform | ΔE | ⟨N⟩ prior | ⟨N⟩ uniform | Δ⟨N⟩ |",
                   "|--:|--:|--:|--:|--:|--:|--:|--:|"]
            for x in r["rows"]:
                dn = "—" if x["dN_rel"] is None else f"{x['dN_rel']:.2%}"
                md.append(f"| {x['N_f']} | {x['n_b']} | {x['E_prior']:.4f} | "
                          f"{x['E_uniform']:.4f} | {x['dE']:+.4f} | {x['N_prior']:.5f} | "
                          f"{x['N_uniform']:.5f} | {dn} |")
            if r["cross_arm"]:
                md.append("\n**Cross-arm under-convergence:** " + "; ".join(
                    f"`N_f={x['N_f']}` is {x['excess']:.3f} MeV above the "
                    f"`N_f={x['bound_from_N_f']}` bound proved by the other arm"
                    for x in r["cross_arm"]) + ".\n")
            md.append(f"\n**VERDICT: {r['verdict']}** — {r['why']}\n")
        for u in unmatched:
            L, nr = u["key"]
            md += [f"\n## L={L}, n_runs={nr} — UNMATCHED ({u['arm']} arm only)\n",
                   f"_The `{u['missing']}` arm has no data at this (L, n_runs), so no "
                   "comparison is possible and no verdict is issued. The absence is "
                   "reported rather than dropped: in cluster 293942 the missing arm is "
                   "`uniform` at L=4, held after exceeding 96 GB while both prior arms "
                   "completed, which is itself evidence about the two initializations._\n",
                   f"| $N_f$ | $n_b$ | $E$ {u['arm']} | ⟨N⟩ {u['arm']} |",
                   "|--:|--:|--:|--:|"]
            for x in u["rows"]:
                n = x.get("N_per_mode")
                md.append(f"| {x['N_f']} | {x.get('n_b')} | {x['E_var']:.4f} | "
                          f"{'—' if n is None else f'{n:.5f}'} |")
            if u["per_arm"]:
                md.append("\n**Under-convergence (within arm):** " + "; ".join(
                    f"`N_f={x['N_f']}` is {x['excess']:.3f} MeV above the "
                    f"`N_f={x['bound_from_N_f']}` bound" for x in u["per_arm"]) + ".\n")
        open(args.out, "w").write("\n".join(md) + "\n")
        print(f"\n[tbl] wrote {args.out}")


if __name__ == "__main__":
    main()
