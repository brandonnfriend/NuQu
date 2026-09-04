"""Volume scaling of the boson-cutoff shift (re-audit P0-4): can we BOUND the n_b=3 cutoff error in the
TOTAL energy at large L? Measure the PAIRED shift Δ34(L) = E(n_b=4) - E(n_b=3) at COMMON cores for
L=2,3,4 (the shift cancels most core-incompleteness), track it DOWN THE CORE LADDER, and read the
per-site shift at the deepest cores.

HONEST FINDING (do not overclaim): the paired Δ34 at a fixed core is **selected-CI-noise-limited** at
L≥3 — two INDEPENDENT selected-CI solves (n_b=3 vs n_b=4) pick slightly different spatial determinants,
so their energy difference carries selection noise of ~0.02-0.2 MeV total (~0.001-0.003 MeV/site), the
same size as (or larger than) the true shift. Consequences:
  * The shift OSCILLATES in sign across cores (e.g. L=3: +0.064@256k → -0.040@512k) and the
    deepest-common-core estimate is unstable (L=4: +0.179@128k[OOM] → +0.017@256k once mem was fixed).
  * A clean per-site slope CANNOT be fit, so there is NO reliable linear-in-sites L=10 projection.
    (An earlier +1.78 MeV "projection" was a shallow-core noise spike; it is withdrawn.)
What we CAN say: at the deepest cores the shift is SMALL — |Δ34/site| ≤ ~0.0015 MeV/site at L=2,3,4,
and at the largest lattice (L=4) it is stable and below the 0.001 MeV/site target-equivalent (1 MeV /
1000 sites). This corroborates the L=2 gate (n_b=3 ≈ n_b=4) but does not DISCHARGE the conditional:
the differencing precision at L≥3 is coarser than the 1 MeV/L=10 target, so n_b=3's large-volume
adequacy is CONDITIONAL (retained), not proven.

    python -m misc.make_nb_volscaling
"""
import argparse
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, GREEN, CRIT, MUTED = "#2a78d6", "#eb6834", "#3a9b6a", "#d03b3b", "#898781"
INK, INK2, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#fcfcfb"
LCOLOR = {2: BLUE, 3: ORANGE, 4: GREEN}
DEEP_CORE = 32000            # cores at/below this are too shallow to be even approximately converged
TARGET_PER_SITE = 0.001      # 1 MeV GSEE target / 1000 sites (L=10) = the per-site line that matters
DEFAULT_SRC = [
    "data/classical/nb_energy_gate",       # L=2 (n_b=3,4)
    "data/classical/nb_energy_gate_L3",    # L=3 n_b=3
    "data/classical/nb_volscaling",        # L=3 n_b=4 + L=4 n_b={3,4} @256k (primary)
    "data/classical/nb_volscaling_deep",   # L=3,4 n_b={3,4} @512k, A=1 (deep-core probe)
]


def load(srcs):
    """{(L, n_b, A): {core: E_min_over_seeds}} + sites[L]."""
    lad, sites = {}, {}
    for d in srcs:
        for f in glob.glob(f"{d}/nb*/bare_L*.json"):
            nb = int(re.search(r"/nb(\d+)/", f).group(1))
            m = re.search(r"bare_L(\d+)d\d+_A(\d+)_s(\d+)", os.path.basename(f))
            if not m:
                continue
            L, A = int(m.group(1)), int(m.group(2))
            j = json.load(open(f)); sites[L] = j["sites"]
            for r in j["rungs"]:
                if r.get("E_var") is None:
                    continue
                c = int(r["core"]); dd = lad.setdefault((L, nb, A), {})
                dd[c] = min(dd.get(c, r["E_var"]), r["E_var"])
    return lad, sites


def ladder(lad, L, A):
    """[(core, Δ34_total)] over the common cores of (n_b=3, n_b=4) at (L, A); None if the pair is absent."""
    a, b = lad.get((L, 3, A)), lad.get((L, 4, A))
    if not a or not b:
        return None
    return [(c, b[c] - a[c]) for c in sorted(set(a) & set(b))]


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS); ax.spines[s].set_linewidth(1.0)
    ax.tick_params(colors=MUTED, labelcolor=INK2, length=3, width=0.8)
    ax.grid(True, color=GRID, lw=0.8, alpha=1.0); ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", default=DEFAULT_SRC)
    ap.add_argument("--A", type=int, default=1)
    ap.add_argument("--dim", type=int, default=3)
    ap.add_argument("--out-dir", default="data/classical/nb_volscaling")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    lad, sites = load(args.src)

    rows = []
    for L in (2, 3, 4):
        lad34 = ladder(lad, L, args.A) or ladder(lad, L, 0)   # fall back to A=0 if A=1 pair absent (L=2)
        if not lad34:
            continue
        n = L ** args.dim
        deep = [(c, d) for c, d in lad34 if c >= DEEP_CORE]
        deep_ps = [d / n for _, d in deep] or [lad34[-1][1] / n]
        rows.append(dict(L=L, sites=n, lad=lad34, top_core=lad34[-1][0], top_site=lad34[-1][1] / n,
                         env_lo=min(deep_ps), env_hi=max(deep_ps),
                         env_absmax=max(abs(x) for x in deep_ps)))
    assert rows, "no paired (n_b=3, n_b=4) shifts found — is the volume-scaling data pulled?"

    # conservative envelope over all sampled L, at the deepest cores
    absmax = max(r["env_absmax"] for r in rows)
    verdict = "CONDITIONAL (retained)"   # noise-limited at L≥3 → never a clean discharge from this test

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.0, 4.6))
    fig.patch.set_facecolor(SURFACE)
    # (a) Δ34/site DOWN THE CORE LADDER — shows it oscillates around 0 within the noise envelope
    axA.axhline(0, color=MUTED, lw=1.0, ls="-", zorder=1)
    for r in rows:
        cs = [c for c, _ in r["lad"] if c >= 1000]
        ys = [d / r["sites"] for c, d in r["lad"] if c >= 1000]
        axA.semilogx(cs, ys, "-o", color=LCOLOR[r["L"]], lw=1.6, ms=5, mec=SURFACE, mew=1.0,
                     zorder=4, label=f"$L$={r['L']} ({r['sites']} sites)")
    axA.axhspan(-TARGET_PER_SITE, TARGET_PER_SITE, color=GREEN, alpha=0.10, zorder=0)
    axA.set_xlabel("selected-CI core (determinants)", color=INK2, fontsize=9.5)
    axA.set_ylabel("Δ$_{34}$ per site  (MeV/site)", color=INK2, fontsize=9.5)
    axA.set_title("a  Cutoff shift vs core — noise-limited (oscillates about 0)", color=INK,
                  fontsize=10.0, loc="left", weight="bold")
    axA.legend(frameon=False, fontsize=8, labelcolor=INK2)
    _style(axA)
    # (b) deepest-core Δ34/site vs L, with the deep-core noise envelope + the target-equivalent band
    axB.axhspan(-TARGET_PER_SITE, TARGET_PER_SITE, color=GREEN, alpha=0.12, zorder=0,
                label="±(1 MeV / 1000 sites)")
    axB.axhline(0, color=MUTED, lw=1.0, zorder=1)
    xs = [r["L"] for r in rows]
    ys = [r["top_site"] for r in rows]
    yerr = [[r["top_site"] - r["env_lo"] for r in rows], [r["env_hi"] - r["top_site"] for r in rows]]
    axB.errorbar(xs, ys, yerr=yerr, fmt="o", color=CRIT, ms=8, mec=SURFACE, mew=1.2, capsize=5,
                 elinewidth=1.4, zorder=4, label="deepest core (bars = deep-core noise envelope)")
    axB.set_xticks(xs); axB.set_xlabel("lattice size $L$", color=INK2, fontsize=9.5)
    axB.set_ylabel("Δ$_{34}$ per site at deepest core  (MeV/site)", color=INK2, fontsize=9.5)
    axB.set_title("b  Shift ≤ noise; envelope straddles the target line", color=INK, fontsize=10.0,
                  loc="left", weight="bold")
    axB.legend(frameon=False, fontsize=7.6, labelcolor=INK2, loc="upper left")
    _style(axB)
    fig.suptitle("Volume scaling of the n_b=3 cutoff shift — %s: the paired shift is below selected-CI "
                 "differencing precision at L≥3" % verdict, fontsize=10.4, color=INK, y=1.02, x=0.01,
                 ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for e in ("pdf", "png"):
        fig.savefig(f"{args.out_dir}/nb_volscaling.{e}", dpi=200, bbox_inches="tight", facecolor=SURFACE)
    print(f"[fig] wrote {args.out_dir}/nb_volscaling.pdf / .png")

    md = ["# Volume scaling of the n_b=3 boson-cutoff shift (re-audit P0-4)\n",
          f"_Paired shift Δ34(L)=E(n_b=4)−E(n_b=3) at COMMON cores, A={args.A}, dim={args.dim}, tracked "
          f"down the core ladder. **The shift is selected-CI-noise-limited at L≥3** (two independent "
          f"solves → ~0.001–0.003 MeV/site selection noise), so it oscillates in sign across cores and "
          f"NO reliable per-site slope / L=10 projection can be extracted. We report the deepest-core "
          f"value and the deep-core (≥{DEEP_CORE//1000}k) noise envelope._\n",
          "| L | sites | deepest core | Δ34/site @deepest (MeV) | deep-core envelope (MeV/site) |",
          "|--:|--:|--:|--:|--:|"]
    for r in rows:
        md.append(f"| {r['L']} | {r['sites']} | {r['top_core']:,} | {r['top_site']:+.5f} | "
                  f"[{r['env_lo']:+.5f}, {r['env_hi']:+.5f}] |")
    md.append(f"\n**Reading (no overclaim):** at the deepest cores the n_b=3→4 shift is **small — "
              f"|Δ34/site| ≤ {absmax:.4f} MeV/site** across L=2,3,4, and at the largest lattice (L=4) it "
              f"is stable at {[r['top_site'] for r in rows if r['L']==4][0]:+.5f}/site, **below** the "
              f"0.001 MeV/site target-equivalent (1 MeV / 1000 sites). This CORROBORATES the L=2 gate "
              f"(n_b=3 ≈ n_b=4) — the shift shows no clean systematic accumulation with volume. BUT the "
              f"deep-core noise envelope (±{absmax:.4f}/site) is as large as the signal and STRADDLES "
              f"the target line (L=3 even flips sign 256k→512k), so the differencing precision at L≥3 is "
              f"coarser than the 1 MeV/L=10 target. **VERDICT: {verdict}** — the volume-scaling test "
              f"finds the cutoff shift SMALL and consistent with n_b=3 adequacy, but cannot bound it "
              f"below the target at L=10, so C1's large-volume conditional is **retained, not "
              f"discharged**. An earlier +1.78 MeV linear projection was a shallow-core (OOM-limited) "
              f"noise artifact and is withdrawn.\n")
    open(f"{args.out_dir}/nb_volscaling_table.md", "w").write("\n".join(md) + "\n")
    print(f"[tbl] wrote {args.out_dir}/nb_volscaling_table.md")
    print(f"[done] deepest-core |Δ34/site| ≤ {absmax:.5f} across L=2,3,4; L=4 stable "
          f"{[r['top_site'] for r in rows if r['L']==4][0]:+.5f}/site -> {verdict} (noise-limited, no projection)")


if __name__ == "__main__":
    main()
