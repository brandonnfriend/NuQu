"""DMRG area-law figure + table — the evidence for the DMRG rule-out claim.

THE CLAIM
    "DMRG is limited by the area law, since the bond dimension needed for a fixed
     truncation error grows with the area of the cut, which is L^2 in three dimensions,
     and the explicit pion modes make the local dimension larger on top of that."

WHAT IS MEASURED (one observable per clause)
  (1) chi(dw) -- the bond dimension at which the block2 DISCARDED WEIGHT first falls
      below a fixed target -- plotted against CUT AREA = L^(dim-1). This is the literal
      "bond dimension needed for a fixed truncation error".
  (2) The 3D rungs carry area L^2. The decisive control is the pair
      (L=4, dim=2) and (L=2, dim=3): SAME cut area 4, volumes 16 vs 8 sites. If chi
      tracks area they agree; if it tracks volume they do not. Panel (a) marks them.
  (3) chi(dw) vs N_f at fixed geometry, with N_f=1 the PION-FREE control (boson space =
      vacuum only). Per-site local dimension = 2^4 x N_f^3.

S_max_bond (max bipartite entanglement entropy) is carried alongside: it is the physical
driver -- the area law is a statement about S, and chi ~ e^S.

HONEST SCOPE. chi(dw) is read off a DISCRETE chi schedule, so it is a bracketing
estimate (the first rung that clears the target), not an interpolated exact value; the
table reports the bracket. A shard that hit the per-chi wall cap before clearing the
target is reported as a LOWER BOUND (chi > deepest rung), never silently dropped.

Contact-term convention: chi, S_max and the discarded weight are all INVARIANT under the
2026-09-14 Wick fix (a constant energy shift does not change eigenvectors); these runs are
post-fix by construction.

    python -m misc.make_dmrg_arealaw --data data/classical/<date>/dmrg_arealaw_<cid> \
        --out-dir results/06_classical_rule_out
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from misc.apply_wick_correction import annotate as _wick_annotate

BLUE, ORANGE, CRIT, GREEN, MUTED = "#2a78d6", "#eb6834", "#d03b3b", "#3a9b6a", "#898781"
INK, INK2, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#fcfcfb"

DW_TARGET = 1e-6          # the "fixed truncation error"


def cut_area(L, dim):
    """Bisecting cut through an L^dim lattice has area L^(dim-1) -- L^2 in 3D."""
    return L ** (dim - 1)


def load(data_dirs):
    recs = []
    for d in data_dirs:
        for f in sorted(glob.glob(f"{d}/dmrg_L*.json")):
            j = json.load(open(f))
            rungs = [r for r in j.get("results", []) if r.get("chi")]
            if not rungs:
                continue
            recs.append(dict(L=j["L"], dim=j["dim"], A=j["A"], N_f=j["N_f"],
                             sites=j["sites"], area=cut_area(j["L"], j["dim"]),
                             rungs=rungs, done=j.get("done", False),
                             local_dim=16 * j["N_f"] ** 3, path=f))
    return recs


def chi_at_dw(rec, target=DW_TARGET):
    """First chi whose discarded weight <= target. Returns (chi, is_lower_bound).

    `is_lower_bound=True` means no rung cleared the target -- the shard stopped (wall cap
    or schedule end) before reaching it, so the true chi is LARGER than the deepest rung.
    """
    for r in rec["rungs"]:
        dw = r.get("discarded_weight")
        if dw is not None and dw <= target:
            return r["chi"], False
    return rec["rungs"][-1]["chi"], True


def s_max(rec):
    vals = [r["S_max_bond"] for r in rec["rungs"] if r.get("S_max_bond") is not None]
    return max(vals) if vals else None


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS); ax.spines[s].set_linewidth(1.0)
    ax.tick_params(colors=MUTED, labelcolor=INK2, length=3, width=0.8)
    ax.grid(True, color=GRID, lw=0.8, alpha=1.0); ax.set_axisbelow(True)


def make_figure(recs, out_base):
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.8, 4.5))
    fig.patch.set_facecolor(SURFACE)

    # (a) chi vs CUT AREA, at the reference local dimension N_f=2
    geo = sorted([r for r in recs if r["N_f"] == 2], key=lambda r: (r["area"], r["sites"]))
    for r in geo:
        chi, lb = chi_at_dw(r)
        col = {1: BLUE, 2: GREEN, 3: CRIT}[r["dim"]]
        axA.scatter([r["area"]], [chi], s=70, color=col, zorder=5,
                    marker="^" if lb else "o",
                    edgecolor=INK if r["area"] == 4 else "none", linewidth=1.4)
        axA.annotate(f"$L$={r['L']}, {r['dim']}D", (r["area"], chi), fontsize=7.5,
                     color=INK2, textcoords="offset points", xytext=(7, -3))
    axA.set_xlabel("cut area  $L^{D-1}$   ($L^2$ in 3D)", color=INK2, fontsize=9.5)
    axA.set_ylabel(f"$\\chi$ for discarded weight $\\leq$ {DW_TARGET:g}", color=INK2, fontsize=9.5)
    axA.set_yscale("log")
    axA.set_title("a  Bond dimension vs cut area  ($N_f$=2)", color=INK, fontsize=10.5,
                  loc="left", weight="bold")
    axA.plot([], [], "o", color=MUTED, label="reached the target")
    axA.plot([], [], "^", color=MUTED, label="lower bound (capped)")
    axA.scatter([], [], s=70, facecolor="none", edgecolor=INK, linewidth=1.4,
                label="area 4: $L$=4 2D vs $L$=2 3D\n(same area, 16 vs 8 sites)")
    axA.legend(frameon=False, fontsize=7.4, loc="upper left", labelcolor=INK2)
    _style(axA)

    # (b) chi vs LOCAL DIMENSION (the explicit pion modes), at fixed geometry
    for dim, col, mk in ((2, GREEN, "o"), (3, CRIT, "s")):
        pts = sorted([r for r in recs if r["dim"] == dim and r["L"] == 2],
                     key=lambda r: r["N_f"])
        if not pts:
            continue
        xs = [p["local_dim"] for p in pts]
        ys = [chi_at_dw(p)[0] for p in pts]
        axB.plot(xs, ys, f"--{mk}", color=col, lw=1.6, ms=7,
                 label=f"$L$=2, {dim}D")
        for p, x, y in zip(pts, xs, ys):
            tag = "$N_f$=1\n(pion-free)" if p["N_f"] == 1 else f"$N_f$={p['N_f']}"
            axB.annotate(tag, (x, y), fontsize=7.2, color=INK2,
                         textcoords="offset points", xytext=(6, -4))
    axB.set_xscale("log"); axB.set_yscale("log")
    axB.set_xlabel("per-site local dimension  $2^4 \\times N_f^{\\,3}$", color=INK2, fontsize=9.5)
    axB.set_ylabel(f"$\\chi$ for discarded weight $\\leq$ {DW_TARGET:g}", color=INK2, fontsize=9.5)
    axB.set_title("b  The explicit pion modes, on top", color=INK, fontsize=10.5,
                  loc="left", weight="bold")
    axB.legend(frameon=False, fontsize=7.8, loc="upper left", labelcolor=INK2)
    _style(axB)

    fig.suptitle("DMRG is area-law limited: $\\chi$ for a fixed truncation error grows with the "
                 "cut area ($L^2$ in 3D), and the pion modes enlarge the local dimension",
                 fontsize=11.0, color=INK, y=1.02, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _wick_annotate(fig, kind="invariant")
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_base}.{ext}", dpi=200, bbox_inches="tight", facecolor=SURFACE)
    print(f"[fig] wrote {out_base}.pdf / .png")


def make_table(recs, out_path):
    md = ["# DMRG area-law wall — $\\chi$ for a fixed truncation error\n",
          f"_block2 DMRG on the post-vertex-fix mixed EFT Hamiltonian, A=2 (dilute). "
          f"$\\chi$ is the FIRST rung of the bond-dimension schedule whose discarded weight "
          f"falls to $\\leq$ {DW_TARGET:g} — a bracketing estimate on a discrete schedule, not "
          f"an interpolated value. Rows marked **LB** never reached the target (the shard hit "
          f"its per-$\\chi$ wall cap), so the true $\\chi$ is LARGER; they are lower bounds and "
          f"are drawn as triangles. Cut area = $L^{{D-1}}$._\n",
          "| L | D | sites | cut area | $N_f$ | local dim | $\\chi$(dw$\\leq$tgt) | $S_\\max$ | deepest $\\chi$ | done |",
          "|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--|"]
    for r in sorted(recs, key=lambda r: (r["N_f"], r["area"], r["sites"])):
        chi, lb = chi_at_dw(r)
        sm = s_max(r)
        md.append(f"| {r['L']} | {r['dim']} | {r['sites']} | {r['area']} | {r['N_f']} | "
                  f"{r['local_dim']:,} | {'**&gt;'+str(chi)+'** (LB)' if lb else chi} | "
                  f"{'—' if sm is None else f'{sm:.3f}'} | {r['rungs'][-1]['chi']} | "
                  f"{'yes' if r['done'] else 'CAPPED'} |")
    md.append("\n**The area-vs-volume control.** `L=4, D=2` and `L=2, D=3` have the same cut "
              "area (4) but different volumes (16 vs 8 sites). Agreement between their $\\chi$ "
              "is what distinguishes an AREA law from a volume law — read those two rows "
              "together before quoting the figure.\n")
    md.append("**The pion-mode clause.** `N_f=1` is the pion-free control: a single boson level "
              "is the vacuum, so the boson sector contributes nothing to the local dimension. "
              "The rise in $\\chi$ from $N_f$=1 to the physical $N_f$=8 at FIXED geometry is the "
              "cost the explicit pion modes add on top of the area law.\n")
    open(out_path, "w").write("\n".join(md) + "\n")
    print(f"[tbl] wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", nargs="+", required=True, help="shard dir(s)")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    recs = load(args.data)
    assert recs, "no dmrg_L*.json shards found"
    make_figure(recs, f"{args.out_dir}/dmrg_arealaw")
    make_table(recs, f"{args.out_dir}/dmrg_arealaw_table.md")
    print("[done] " + " | ".join(
        f"L{r['L']}d{r['dim']}Nf{r['N_f']}: area {r['area']}, chi {chi_at_dw(r)[0]}"
        f"{'(LB)' if chi_at_dw(r)[1] else ''}" for r in recs))


if __name__ == "__main__":
    main()
