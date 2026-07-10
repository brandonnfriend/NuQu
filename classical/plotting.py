"""
Plotting for the classical pipeline.

Two headline plots for the cost-confront story:
  * `plot_convergence`  — energy vs determinant count (does the method converge?)
  * `plot_runtime_vs_size` — wall-clock vs Hilbert-sector size (WHERE does the
    classical solver stop running quickly enough — the key question for task 14).

Method/frame aware: series are keyed on (method, transform) so TrimCI-bare, a
future TrimCI-COO / TrimCI-LF, and other methods overlay on the same axes.
"""

from __future__ import annotations

import os

from .analysis import index_runs, convergence_of
from .io import DATA_ROOT


def _ensure_mpl():
    import matplotlib
    matplotlib.use("Agg")  # headless (no display needed)
    import matplotlib.pyplot as plt
    return plt


def plot_convergence(run_dirs, out_path=None, title=None):
    """Energy vs n_dets for one or more saved runs (overlaid)."""
    plt = _ensure_mpl()
    if isinstance(run_dirs, str):
        run_dirs = [run_dirs]
    fig, ax = plt.subplots(figsize=(6, 4))
    for rd in run_dirs:
        import json
        with open(os.path.join(rd, "metadata.json")) as f:
            m = json.load(f)
        conv = m.get("convergence", [])
        if not conv:
            continue
        nd = [c[0] for c in conv]
        e = [c[1] for c in conv]
        lbl = f"{m['method']}-{m['transform']} L{m['system']['L']}d" \
              f"{m['system']['dim']} A{m['system']['A']} N_f{m['system']['N_f']}"
        ax.plot(nd, e, "o-", ms=4, label=lbl)
        ref = m.get("exact_reference") or {}
        if ref.get("energy") is not None:
            ax.axhline(ref["energy"], ls="--", lw=1, color="k", alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("determinants in core")
    ax.set_ylabel("ground-state energy (MeV)")
    ax.set_title(title or "TrimCI convergence")
    ax.legend(fontsize=7)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=140)
        print(f"[plot] wrote {out_path}")
    return fig


def plot_runtime_vs_sites(data_root=DATA_ROOT, out_path=None, title=None):
    """Wall-clock runtime vs #lattice sites — the actual runtime driver (runtime
    ≈ linear in #sites/#terms), with the Fock cutoff N_f shown by color to make
    the point that N_f is ~free."""
    plt = _ensure_mpl()
    import matplotlib
    df = index_runs(data_root)
    if df.empty:
        print(f"(no runs under {data_root})")
        return None
    df = df.copy()
    df["sites"] = (df["L"].astype(float) ** df["dim"].astype(float)).round().astype(int)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    sc = ax.scatter(df["sites"], df["runtime_s"], c=df["N_f"], cmap="plasma",
                    s=55, edgecolor="k", linewidth=0.4,
                    norm=matplotlib.colors.LogNorm())
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("Fock cutoff $N_f$")
    ax.set_xlabel("lattice sites (L$^d$)")
    ax.set_ylabel("classical solve wall-clock (s)")
    ax.set_title(title or "TrimCI runtime vs #sites (N_f ~ free)")
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=140)
        print(f"[plot] wrote {out_path}")
    return fig


def _l_series(data_root, dim, label, A=None, method="TrimCI", transform="bare"):
    """Saved runs for a fixed-dim L-scaling series, sorted by L. Returns a
    DataFrame with a `sites` (= L**dim) column. Filters on label so a dedicated
    sweep doesn't mix with ad-hoc runs."""
    df = index_runs(data_root)
    if df.empty:
        return df
    df = df[(df["dim"] == dim) & (df["method"] == method) &
            (df["transform"] == transform)].copy()
    if label is not None:
        df = df[df["label"] == label]
    if A is not None:
        df = df[df["A"] == A]
    df["sites"] = (df["L"].astype(float) ** df["dim"].astype(float)).round().astype(int)
    return df.sort_values("L").reset_index(drop=True)


def plot_runtime_vs_L(data_root=DATA_ROOT, dim=3, label=None, A=None,
                      out_path=None, title=None):
    """Classical wall-clock vs lattice side L (fixed dim) — the cost-confront
    curve: where does the solver stop running quickly? A power-law guide
    (runtime ~ sites^b, fit on the points) is overlaid for extrapolation."""
    plt = _ensure_mpl()
    import numpy as np
    df = _l_series(data_root, dim, label, A=A)
    if df is None or df.empty:
        print(f"(no dim={dim} label={label} runs under {data_root})")
        return None
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(df["L"], df["runtime_s"], "o-", ms=7, color="C3", label="measured")
    # power-law fit runtime = a * sites^b (needs >=2 points)
    if len(df) >= 2:
        b, loga = np.polyfit(np.log(df["sites"]), np.log(df["runtime_s"]), 1)
        Lg = np.linspace(df["L"].min(), df["L"].max() + 1, 50)
        sg = Lg.astype(float) ** dim
        ax.plot(Lg, np.exp(loga) * sg ** b, "--", color="gray", lw=1.2,
                label=f"~sites$^{{{b:.2f}}}$ fit")
        ax.legend(fontsize=8)
    for _, r in df.iterrows():
        ax.annotate(f"{r['runtime_s']:.1f}s", (r["L"], r["runtime_s"]),
                    textcoords="offset points", xytext=(6, -2), fontsize=7)
    ax.set_yscale("log")
    ax.set_xticks(df["L"].tolist())
    ax.set_xlabel(f"lattice side $L$  (dim={dim}; sites $= L^{dim}$)")
    ax.set_ylabel("classical solve wall-clock (s)")
    ax.set_title(title or f"TrimCI runtime vs L  (dim={dim}, A={A or 'all'})")
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=140)
        print(f"[plot] wrote {out_path}")
    return fig


def plot_compression_vs_L(data_root=DATA_ROOT, dim=3, label=None, A=None,
                          out_path=None, title=None):
    """The selected-CI compression: full Hilbert-sector size / #states kept in the
    compact ground state, vs L. This is the headline 'why TrimCI' number — the
    factor by which the wavefunction is compressed below the exact basis."""
    plt = _ensure_mpl()
    df = _l_series(data_root, dim, label, A=A)
    if df is None or df.empty:
        print(f"(no dim={dim} label={label} runs under {data_root})")
        return None
    df = df.copy()
    df["ratio"] = df["sector_size"].astype(float) / df["n_dets"].astype(float)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(df["L"], df["ratio"], "o-", ms=7, color="C0")
    for _, r in df.iterrows():
        ax.annotate(f"{r['sector_size']:.0e}/{int(r['n_dets'])}",
                    (r["L"], r["ratio"]), textcoords="offset points",
                    xytext=(6, -3), fontsize=6.5)
    ax.set_yscale("log")
    ax.set_xticks(df["L"].tolist())
    ax.set_xlabel(f"lattice side $L$  (dim={dim}; sites $= L^{dim}$)")
    ax.set_ylabel("Hilbert sector / compact-core states")
    ax.set_title(title or f"Selected-CI compression vs L  (dim={dim}, A={A or 'all'})")
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=140)
        print(f"[plot] wrote {out_path}")
    return fig


def plot_energy_vs_L(data_root=DATA_ROOT, dim=3, label=None, A=None,
                     out_path=None, title=None):
    """Ground-state energy vs L (fixed dim). Total E0 (extensive) plus E0/site to
    show the per-site energy approaching its thermodynamic value."""
    plt = _ensure_mpl()
    df = _l_series(data_root, dim, label, A=A)
    if df is None or df.empty:
        print(f"(no dim={dim} label={label} runs under {data_root})")
        return None
    df = df.copy()
    df["e_per_site"] = df["energy"].astype(float) / df["sites"]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(df["L"], df["energy"], "o-", ms=7, color="C2", label="$E_0$ (total)")
    ax.set_xlabel(f"lattice side $L$  (dim={dim}; sites $= L^{dim}$)")
    ax.set_ylabel("ground-state energy $E_0$ (MeV)", color="C2")
    ax.tick_params(axis="y", labelcolor="C2")
    ax.set_xticks(df["L"].tolist())
    ax2 = ax.twinx()
    ax2.plot(df["L"], df["e_per_site"], "s--", ms=5, color="C4",
             label="$E_0$/site")
    ax2.set_ylabel("$E_0$ / site (MeV)", color="C4")
    ax2.tick_params(axis="y", labelcolor="C4")
    ax.set_title(title or f"Ground-state energy vs L  (dim={dim}, A={A or 'all'})")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], fontsize=8, loc="best")
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=140)
        print(f"[plot] wrote {out_path}")
    return fig


def plot_core_convergence(records, L, dim, A=1, out_path=None, title=None):
    """Single-system energy-vs-core-size convergence, from independent-ladder
    records (list of dicts with 'core' and 'energy'). Two stacked panels:
      top    — E0 vs core size (log-x): the variational descent and flattening;
      bottom — |per-doubling relative drop| vs core (log-log): the convergence
               rate, i.e. "how much does the next doubling still buy?".
    Independent full solves are the reliable signal (a single growing run's ramp
    plateaus deceptively)."""
    plt = _ensure_mpl()
    import numpy as np
    recs = sorted(records, key=lambda r: r["core"])
    core = np.array([r["core"] for r in recs], dtype=float)
    E = np.array([r["energy"] for r in recs], dtype=float)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.6, 6.2), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 2]})
    ax1.plot(core, E, "o-", ms=7, color="C2")
    for c, e in zip(core, E):
        ax1.annotate(f"{e:.1f}", (c, e), textcoords="offset points",
                     xytext=(0, 7), fontsize=6.5, ha="center")
    ax1.set_ylabel("ground-state energy $E_0$ (MeV)")
    ax1.set_title(title or f"L={L} (dim={dim}, A={A}) energy convergence vs core size")
    ax1.grid(True, which="both", alpha=0.25)
    # per-doubling relative drop between successive (independent) cores
    rel = np.abs(np.diff(E)) / np.maximum(np.abs(E[1:]), 1e-12)
    ax2.plot(core[1:], rel, "s-", ms=6, color="C3")
    for c, rr in zip(core[1:], rel):
        ax2.annotate(f"{rr:.1e}", (c, rr), textcoords="offset points",
                     xytext=(0, 6), fontsize=6.5, ha="center")
    ax2.set_yscale("log")
    ax2.set_xscale("log")
    ax2.set_xlabel("core size (selected determinants)")
    ax2.set_ylabel("rel. drop |ΔE/E|\nvs previous core")
    ax2.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=140)
        print(f"[plot] wrote {out_path}")
    return fig


def plot_compaction_vs_frame(comparison, L=None, dim=None, A=1,
                             out_path=None, title=None):
    """STEP-4 headline: ground-state compactness across frames (bare / gaussian / LF /
    gaussian+LF), from `frame.frame_comparison(...)` — a `{frame: {n999, n99,
    participation_ratio, ...}}` dict. Grouped bars of the 99.9%-weight core size (and
    participation ratio), annotated with the ×-gain over bare, so "the layered frame
    gives X× fewer determinants at matched accuracy" is read straight off the plot."""
    plt = _ensure_mpl()
    import numpy as np
    order = [f for f in ("bare", "gaussian", "LF", "gaussian+LF") if f in comparison]
    n999 = np.array([comparison[f]["n999"] for f in order], dtype=float)
    pr = np.array([comparison[f]["participation_ratio"] for f in order], dtype=float)
    base = comparison["bare"]["n999"] if "bare" in comparison else n999.max()
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.bar(x - 0.19, n999, 0.38, color="C0", label="n(99.9% weight)")
    ax.bar(x + 0.19, pr, 0.38, color="C1", alpha=0.75, label="participation ratio")
    for xi, v in zip(x, n999):
        gain = base / max(v, 1)
        ax.annotate(f"{int(v)}\n({gain:.1f}×)", (xi - 0.19, v),
                    textcoords="offset points", xytext=(0, 4), fontsize=7.5, ha="center")
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylabel("compact-core size (lower = more compact)")
    sysstr = "" if L is None else f"  (L={L}, dim={dim}, A={A})"
    ax.set_title(title or f"Ground-state compaction by frame{sysstr}")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=140)
        print(f"[plot] wrote {out_path}")
    return fig


def plot_coresize_vs_L(data_root=DATA_ROOT, dim=3, label=None, A=None,
                       out_path=None, title=None):
    """Final (dynamically chosen) compact-core size vs L — how many determinants
    the convergence criterion needed, which should grow with the system. Points
    are annotated with the stop reason (converged vs hit the ceiling)."""
    plt = _ensure_mpl()
    import json
    df = _l_series(data_root, dim, label, A=A)
    if df is None or df.empty:
        print(f"(no dim={dim} label={label} runs under {data_root})")
        return None
    reasons = []
    for _, r in df.iterrows():
        try:
            with open(os.path.join(r["run_dir"], "metadata.json")) as f:
                reasons.append(json.load(f).get("solver", {}).get("stop_reason", ""))
        except Exception:
            reasons.append("")
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(df["L"], df["n_dets"], "o-", ms=7, color="C5")
    for (_, r), reason in zip(df.iterrows(), reasons):
        ax.annotate(f"{int(r['n_dets'])}\n{reason}", (r["L"], r["n_dets"]),
                    textcoords="offset points", xytext=(6, -4), fontsize=6.5)
    ax.set_xticks(df["L"].tolist())
    ax.set_xlabel(f"lattice side $L$  (dim={dim}; sites $= L^{dim}$)")
    ax.set_ylabel("final compact-core size (determinants)")
    ax.set_title(title or f"Converged core size vs L  (dim={dim}, A={A or 'all'})")
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=140)
        print(f"[plot] wrote {out_path}")
    return fig


def plot_convergence_vs_L(data_root=DATA_ROOT, dim=3, label=None, A=None,
                          out_path=None, title=None):
    """Per-L energy convergence: residual E(core) - E_final vs core size, one
    curve per L (log-log). Answers whether the fixed-n_dets energies in the other
    vs-L plots are converged — a curve dropping toward a small final gap is. Reads
    the saved `convergence` trajectory (the solve's core-ramp history, or the
    independent n_dets ladder if n_conv>1)."""
    plt = _ensure_mpl()
    import json
    import numpy as np
    df = _l_series(data_root, dim, label, A=A)
    if df is None or df.empty:
        print(f"(no dim={dim} label={label} runs under {data_root})")
        return None
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    cmap = plt.get_cmap("viridis")
    Ls = sorted(df["L"].unique())
    for i, L in enumerate(Ls):
        rd = df[df["L"] == L]["run_dir"].iloc[0]
        try:
            with open(os.path.join(rd, "metadata.json")) as f:
                meta = json.load(f)
        except Exception:
            continue
        conv = meta.get("convergence", [])
        if len(conv) < 2:
            continue
        # residual to the best (final, largest-core) energy: shows the variational
        # descent as the core grows. The last-core-doubling drop (robust metric)
        # goes in the legend as the honest "does more still help?" number.
        hd = (meta.get("solver", {}) or {}).get("last_doubling_drop_rel")
        nd = np.array([c[0] for c in conv], dtype=float)
        e = np.array([c[1] for c in conv], dtype=float)
        order = np.argsort(nd)
        nd, e = nd[order], e[order]
        r = e - e[-1]
        m = r > 0                          # log axis: keep strictly-positive
        color = cmap(i / max(1, len(Ls) - 1))
        hdlab = f", last 2x: {hd:.1e}" if hd is not None else ""
        ax.plot(nd[m], r[m], "o-", ms=5, color=color,
                label=f"L={L} (core {int(nd[-1])}{hdlab})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("core size (selected determinants)")
    ax.set_ylabel("energy above best (largest-core) value,  E − E_final  (MeV)")
    ax.set_title(title or f"Per-L energy convergence  (dim={dim}, A={A or 'all'})")
    ax.legend(fontsize=7, title="lattice")
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=140)
        print(f"[plot] wrote {out_path}")
    return fig


def plot_dets_vs_L(result, out_path=None, title=None):
    """PHASE C headline — N*(L) = determinants to reach a fixed per-site accuracy,
    vs system volume, from `run_cpp.dets_vs_L_at_fixed_accuracy`. Two panels that
    each straighten one scaling hypothesis, so the eye reads off which regime the
    data are in:
      * left  (log y, LINEAR sites)  — an EXPONENTIAL-in-volume N* ~ e^{γV} is a line;
      * right (log y, LOG sites)      — a POLYNOMIAL-in-volume N* ~ V^γ is a line.
    Fit-worthy points (reference pinned AND N* bracketed) are filled with a vertical
    error bar (ladder bracket ∪ reference-σ propagation); non-fit-worthy L's are drawn
    as open bounds — ▲ lower bound (N* not reached), ▼ upper bound (N* below the
    smallest core) — so the honest "these are bounds, not points" reading is visual.
    One color per eps target; fit lines overlaid where >= 2 fit-worthy points exist."""
    plt = _ensure_mpl()
    import numpy as np
    per_L = result.get("per_L", [])
    fits = result.get("fits", {})
    epslist = result.get("eps_persite_targets", [])
    if not per_L:
        print("(no per-L data to plot)")
        return None
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.6))
    colors = {str(eps): f"C{i}" for i, eps in enumerate(epslist)}

    def _err(e):
        """asymmetric (low, high) span of N* for a bracketed point: the ladder
        bracket (n_lo, n_hi) unioned with the reference-σ bracket."""
        base = e["nstar_pt2"]
        y = e["nstar_repr"]
        lo, hi = base.get("n_lo") or y, base.get("n_hi") or y
        sb = e.get("nstar_sigma_bracket") or [y, y]
        return max(1, min(lo, sb[0])), max(hi, sb[1])

    for eps in epslist:
        key = str(eps)
        col = colors[key]
        for ax in (axL, axR):
            fitx, fity, elo, ehi = [], [], [], []
            for p in per_L:
                e = p["eps"][key]
                y = e["nstar_repr"]
                if y is None:
                    continue
                V = p["sites"]
                if e["fit_worthy"]:
                    lo, hi = _err(e)
                    fitx.append(V); fity.append(y)
                    elo.append(y - lo); ehi.append(hi - y)
                elif e["nstar_pt2"]["status"] == "lower_bound":
                    ax.plot(V, y, "^", ms=9, mfc="none", mec=col, mew=1.6)
                    ax.annotate("", xy=(V, y * 1.9), xytext=(V, y),
                                arrowprops=dict(arrowstyle="->", color=col, lw=1.2))
                else:  # upper_bound
                    ax.plot(V, y, "v", ms=9, mfc="none", mec=col, mew=1.6)
                    ax.annotate("", xy=(V, y / 1.9), xytext=(V, y),
                                arrowprops=dict(arrowstyle="->", color=col, lw=1.2))
            if fitx:
                ax.errorbar(fitx, fity, yerr=[elo, ehi], fmt="o", ms=7, color=col,
                            capsize=3, label=f"ε={eps:g} MeV/site (N*)")
        # overlay fit lines
        f = fits.get(key, {})
        if f.get("ok"):
            Vs = np.array([p["sites"] for p in per_L], dtype=float)
            grid = np.linspace(Vs.min(), Vs.max(), 60)
            ex = f["exponential_in_V"]
            if ex.get("ok"):
                axL.plot(grid, np.exp(ex["intercept"] + ex["slope"] * grid), "--",
                         color=col, lw=1.2,
                         label=f"ε={eps:g}: e^({ex['slope']:.3g}·V), R²={ex['r2']:.2f}")
            po = f["polynomial_in_V"]
            if po.get("ok"):
                axR.plot(grid, np.exp(po["intercept"]) * grid ** po["slope"], "--",
                         color=col, lw=1.2,
                         label=f"ε={eps:g}: V^{po['slope']:.2g}, R²={po['r2']:.2f}")

    for ax in (axL, axR):
        ax.set_yscale("log")
        ax.set_ylabel("N* (determinants to reach ε accuracy)")
        ax.grid(True, which="both", alpha=0.25)
        if ax.get_legend_handles_labels()[0]:      # skip empty-legend warning
            ax.legend(fontsize=7)
    axL.set_xlabel("system volume  V = sites (linear)")
    axL.set_title("exponential test:  N* ~ e^{γV}  is a line")
    axR.set_xscale("log")
    axR.set_xlabel("system volume  V = sites (log)")
    axR.set_title("polynomial test:  N* ~ V^γ  is a line")
    fill = "dilute" if result.get("filling") is None else f"filling={result['filling']:g}"
    fig.suptitle(title or f"Dets-vs-L at fixed per-site accuracy  "
                          f"(dim={result.get('dim')}, {fill})", y=1.02)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
        print(f"[plot] wrote {out_path}")
    return fig


def plot_runtime_vs_size(data_root=DATA_ROOT, out_path=None, title=None,
                         color_by_sites=True):
    """Wall-clock runtime vs Hilbert-sector size across all saved runs — the
    'where does classical stop being fast' picture.

    Points are colored by #lattice sites (= L**dim), because runtime tracks
    #sites (≈ #terms), NOT sector size: at fixed sites, growing N_f inflates the
    sector by orders of magnitude while runtime stays flat (the selected-CI win).
    """
    plt = _ensure_mpl()
    df = index_runs(data_root)
    if df.empty:
        print(f"(no runs under {data_root})")
        return None
    df = df.copy()
    df["sites"] = (df["L"].astype(float) ** df["dim"].astype(float)).round().astype(int)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))

    markers = {("TrimCI", "COO"): "o", ("TrimCI", "LF"): "s"}
    sc = None
    for (method, transform), g in df.groupby(["method", "transform"]):
        mk = markers.get((method, transform), "o")
        if color_by_sites:
            sc = ax.scatter(g["sector_size"], g["runtime_s"], c=g["sites"],
                            cmap="viridis", marker=mk, s=55, edgecolor="k",
                            linewidth=0.4, label=f"{method}-{transform}",
                            norm=__import__("matplotlib").colors.LogNorm())
        else:
            ax.plot(g["sector_size"], g["runtime_s"], mk + "-", ms=6,
                    label=f"{method}-{transform}")
    if sc is not None:
        cb = fig.colorbar(sc, ax=ax)
        cb.set_label("lattice sites (L$^d$)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Hilbert-sector size (states)")
    ax.set_ylabel("classical solve wall-clock (s)")
    ax.set_title(title or "Classical solver runtime vs system size")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=140)
        print(f"[plot] wrote {out_path}")
    return fig
