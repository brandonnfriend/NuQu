"""Headline runtime figure (task 30/34/11): wall-clock cost vs L for A=100 GSEE.

Assembles the resource hierarchy on ONE axis:
  * Trotterization (Watson high n_b, and low n_b)  -- throughput only (product-formula
    depth model not built; T-count is so large the reaction floor stays astronomical).
  * Qubitization (high n_b = sparse_heuristic; best = frame+occ) -- throughput CEILING
    (dashed) AND reaction-limited FLOOR (solid + log<->QROAM band), the architectural
    speedup the qubitized walk structure enables (task 30 reaction model).
  * Classical (TrimCI) -- HELD (no error guarantee to sit beside FT curves; awaiting the
    error-matched determinant-count) -> shown as an annotation, not a curve.

Sources (same superconducting_sota profile, cross-checked consistent):
  * Trotter throughput: data/quantum/2026-08-05/mainfig_runtime.json (task 11 trotter_exact).
  * Qubitization throughput+reaction: data/quantum/2026-08-07/dwalk_290451/reaction_band_A100.json
    (cluster 290451).
"""
import json, os, sys
ROOT = "/Users/brandonfriend/Desktop/Projects/NuQu"
sys.path.insert(0, ROOT)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

YR = 3600 * 24 * 365.25
OUTDIR = os.path.join(ROOT, "data/quantum/2026-08-13")
os.makedirs(OUTDIR, exist_ok=True)

rt = json.load(open(os.path.join(ROOT, "data/quantum/2026-08-05/mainfig_runtime.json")))["runtime_s_by_L"]
rb = json.load(open(os.path.join(ROOT, "data/quantum/2026-08-07/dwalk_290451/reaction_band_A100.json")))["by_variant"]

Ls_tr = [L for L in range(2, 11)]
trot_hi = [rt[str(L)]["trotter_high"] / YR for L in Ls_tr]
trot_lo = [rt[str(L)]["trotter_low"] / YR for L in Ls_tr]


def qvar(key):
    v = rb[key]
    return (v["L"],
            [s / YR for s in v["throughput_s"]],
            [s / YR for s in v["reaction_qroam_s"]],
            [s / YR for s in v["reaction_log_s"]])


fig, ax = plt.subplots(figsize=(9, 6.8))

# --- Trotter (throughput only) ---
ax.plot(Ls_tr, trot_hi, "-", color="#7b1e1e", lw=2.4, marker="s", ms=4,
        label="Trotter, high $n_b$ (Watson)")
ax.plot(Ls_tr, trot_lo, "-", color="#d1607a", lw=2.4, marker="s", ms=4,
        label="Trotter, low $n_b$")

# --- Qubitization (throughput ceiling + reaction floor band) ---
qstyle = {
    "qubit_highnb": ("#e08214", "Qubitization, high $n_b$"),
    "qubit_best":   ("#1b7837", "Qubitization, fully optimized (frame+occ)"),
}
for key, (c, lab) in qstyle.items():
    L, thru, rq, rl = qvar(key)
    ax.plot(L, thru, "--", color=c, lw=1.6, alpha=0.75)          # throughput ceiling
    ax.plot(L, rq, "-", color=c, lw=2.6, marker="o", ms=4, label=lab)  # reaction floor (QROAM)
    ax.fill_between(L, rl, rq, color=c, alpha=0.20)              # log<->QROAM band

# --- reference timescales ---
for yv, lab in [(1 / 365.25, "1 day"), (1.0, "1 year"),
                (1e3, "1 kyr"), (1.38e10, "age of universe")]:
    ax.axhline(yv, color="gray", ls=":", lw=0.7, alpha=0.5)
    ax.text(10.15, yv, lab, fontsize=7.5, color="gray", va="center")

# --- classical: held annotation ---
ax.text(2.1, 3e-8,
        "classical (TrimCI): held\n(awaiting error-matched\ndeterminant count)",
        fontsize=8, color="#444", style="italic",
        bbox=dict(boxstyle="round,pad=0.35", fc="#f4f4f4", ec="#bbb", lw=0.7))

ax.set_yscale("log")
ax.set_xlim(1.8, 11.2)
ax.set_xlabel("Lattice size $L$  (3D, sites $= L^3$)", fontsize=11)
ax.set_ylabel("Wall-clock runtime (years)", fontsize=11)
ax.set_title("GSEE runtime vs $L$   (A=100, superconducting SOTA)\n"
             "Trotter: throughput   ·   Qubitization: throughput ceiling (dashed) + "
             "reaction floor (solid, log↔QROAM band)", fontsize=10.5)

# legend: curves + a throughput/reaction key
handles, labels = ax.get_legend_handles_labels()
handles += [Line2D([0], [0], color="gray", ls="--", lw=1.4),
            Line2D([0], [0], color="gray", ls="-", lw=2.4, marker="o", ms=4)]
labels += ["— throughput ceiling (qubit.)", "— reaction floor (qubit.)"]
ax.legend(handles, labels, loc="upper left", fontsize=8.3, framealpha=0.93, ncol=1,
          bbox_to_anchor=(0.015, 0.86))
ax.grid(True, which="both", alpha=0.13)
fig.tight_layout()

out_png = os.path.join(OUTDIR, "mainfig_runtime_full.png")
fig.savefig(out_png, dpi=150)
print("wrote", out_png)

# --- companion JSON (single-source-of-truth for the figure) ---
combined = {
    "meta": {"A": 100, "task": "GSEE", "profile": "superconducting_sota",
             "y_units": "years", "tau_react_s": 1e-6,
             "trotter_src": "mainfig_runtime.json (task 11)",
             "qubit_src": "reaction_band_A100.json (cluster 290451)"},
    "trotter_high_throughput_yr": dict(zip(Ls_tr, trot_hi)),
    "trotter_low_throughput_yr": dict(zip(Ls_tr, trot_lo)),
    "qubit": {},
}
for key in ("qubit_highnb", "qubit_realistic", "qubit_best"):
    L, thru, rq, rl = qvar(key)
    combined["qubit"][key] = {
        "L": L, "throughput_yr": thru, "reaction_qroam_yr": rq, "reaction_log_yr": rl}
out_json = os.path.join(OUTDIR, "mainfig_runtime_full.json")
json.dump(combined, open(out_json, "w"), indent=2)
print("wrote", out_json)

# --- console: the crossover / hierarchy at L=10 ---
print("\n== L=10, A=100 wall-clock (years) ==")
print(f"  Trotter high n_b (throughput):  {trot_hi[-1]:.2e}")
print(f"  Trotter low n_b  (throughput):  {trot_lo[-1]:.2e}")
for key, (_, lab) in qstyle.items():
    L, thru, rq, rl = qvar(key)
    print(f"  {lab:<42} throughput={thru[-1]:.2e}  reaction(QROAM)={rq[-1]:.2e}")
