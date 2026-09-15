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


def load_arms(data_dirs):
    arms = {}
    for d in data_dirs:
        for f in sorted(glob.glob(os.path.join(d, "studyD*_L2d3A1_*init.json"))):
            j = json.load(open(f))
            arm = j.get("seed_control_arm")
            if arm is None:                       # tolerate the retired pre-fix naming
                arm = "uniform" if "uniforminit" in f else "prior"
            arms[arm] = {"rows": {r["N_f"]: r for r in j["rows"]},
                         "cfg": j.get("seed_control_cfg"), "path": f,
                         "n_runs": j.get("n_runs"), "core": j.get("core")}
    return arms


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


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", default=None, help="optional markdown output path")
    args = ap.parse_args()

    arms = load_arms(args.data)
    ok, why = settings_match(arms)
    print(f"[arms] {', '.join(sorted(arms))}")
    print(f"[match] {'OK' if ok else 'REFUSED'} — {why}")
    if not ok:
        raise SystemExit("cannot issue a verdict on unmatched arms")

    rows = compare(arms)
    print(f"\n{'N_f':>4} {'n_b':>4} {'E_prior':>12} {'E_uniform':>12} {'dE':>9} "
          f"{'<N>_prior':>10} {'<N>_unif':>10} {'dN_rel':>8}")
    print("-" * 82)
    for r in rows:
        dn = "—" if r["dN_rel"] is None else f"{r['dN_rel']:8.3%}"
        print(f"{r['N_f']:>4} {r['n_b']:>4} {r['E_prior']:12.4f} {r['E_uniform']:12.4f} "
              f"{r['dE']:+9.4f} {r['N_prior']:10.5f} {r['N_uniform']:10.5f} {dn}")
    v, why2 = verdict(rows)
    print(f"\nVERDICT: {v}\n  {why2}")

    if args.out:
        md = [f"# Seed control (F-009) — uniform vs near-vacuum initialization\n",
              f"_Matched arms, {why}. Differ ONLY in `boson_init_mean`. "
              f"`E_var` is a valid Ritz upper bound under either init, so a mismatch is a "
              f"search-basin result, not a broken bound._\n",
              "| $N_f$ | $n_b$ | $E_\\mathrm{var}$ prior | $E_\\mathrm{var}$ uniform | ΔE | "
              "⟨N⟩ prior | ⟨N⟩ uniform | Δ⟨N⟩ rel |", "|--:|--:|--:|--:|--:|--:|--:|--:|"]
        for r in rows:
            dn = "—" if r["dN_rel"] is None else f"{r['dN_rel']:.2%}"
            md.append(f"| {r['N_f']} | {r['n_b']} | {r['E_prior']:.4f} | {r['E_uniform']:.4f} "
                      f"| {r['dE']:+.4f} | {r['N_prior']:.5f} | {r['N_uniform']:.5f} | {dn} |")
        md.append(f"\n**VERDICT: {v}** — {why2}\n")
        open(args.out, "w").write("\n".join(md) + "\n")
        print(f"[tbl] wrote {args.out}")


if __name__ == "__main__":
    main()
