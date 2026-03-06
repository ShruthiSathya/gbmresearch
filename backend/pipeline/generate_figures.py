"""
generate_figures.py — Generate all figures from saved pipeline results
=======================================================================
Reads results/pipeline_results.json (written by save_results.py) and
produces four figures. Every number comes from your actual pipeline run —
nothing is hardcoded.

Usage:
    python save_results.py          # run pipeline first
    python generate_figures.py      # then generate figures

Output:
    figures/fig1_cooccurrence.png    — H3K27M / CDKN2A mutual exclusivity
    figures/fig2_drug_rankings.png   — top N drugs, stacked component scores
    figures/fig3_score_scatter.png   — DepMap vs tissue scatter, BBB colour
    figures/fig4_confidence.png      — top hypothesis confidence breakdown

Add to README.md:
    ![Figure 1](figures/fig1_cooccurrence.png)
    ![Figure 2](figures/fig2_drug_rankings.png)
    ![Figure 3](figures/fig3_score_scatter.png)
    ![Figure 4](figures/fig4_confidence.png)
"""

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS_FILE = Path("results/pipeline_results.json")
FIGURES_DIR  = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

# ── Palette ───────────────────────────────────────────────────────────────────
UCLA_BLUE    = "#2774AE"
UCLA_GOLD    = "#FFD100"
DARK_GREY    = "#2C2C2C"
LIGHT_GREY   = "#E0E0E0"
RED_ACCENT   = "#C5383B"
GREEN_ACCENT = "#2E7D32"
PURPLE       = "#6A1B9A"

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
})

BBB_COLORS = {
    "HIGH":     GREEN_ACCENT,
    "MODERATE": UCLA_GOLD,
    "LOW":      RED_ACCENT,
    "UNKNOWN":  "grey",
}


# ─────────────────────────────────────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────────────────────────────────────

def load() -> dict:
    if not RESULTS_FILE.exists():
        print(f"ERROR: {RESULTS_FILE} not found.")
        print("Run 'python save_results.py' first.")
        sys.exit(1)
    with open(RESULTS_FILE) as f:
        data = json.load(f)
    s = data.get("stats", {})
    print(f"Loaded  {RESULTS_FILE}")
    print(f"  timestamp : {data.get('run_timestamp','?')}")
    print(f"  drugs     : {s.get('n_drugs_screened','?')}")
    print(f"  p-value   : {s.get('p_value_label','?')}")
    print(f"  candidates: {len(data.get('top_candidates', []))}")
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — H3K27M / CDKN2A mutual exclusivity
# ─────────────────────────────────────────────────────────────────────────────

def fig1_cooccurrence(data: dict) -> None:
    ct = data.get("contingency_table")
    if not ct:
        print("  fig1 skipped — no contingency_table in results "
              "(genomic data may not have loaded)")
        return

    a = ct["h3k27m_pos_cdkn2a_del"]   # double-hit
    b = ct["h3k27m_pos_cdkn2a_wt"]    # H3K27M only
    c = ct["h3k27m_neg_cdkn2a_del"]   # CDKN2A-del only
    d = ct["h3k27m_neg_cdkn2a_wt"]    # neither
    n    = ct["total"]
    h3n  = ct["h3k27m_count"]
    cdn  = ct["cdkn2a_del_count"]
    pval = ct.get("p_value")
    plbl = ct.get("p_value_label") or (f"{pval:.2e}" if pval else "N/A")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle(
        f"H3K27M Mutation vs CDKN2A Deletion — Mutual Exclusivity in DIPG\n"
        f"PNOC/PBTA Cohort  •  n = {n}  •  Fisher's exact  p = {plbl}",
        fontsize=13, fontweight="bold", y=1.02,
    )

    # ── left: heatmap ─────────────────────────────────────────────────────────
    ax   = axes[0]
    tbl  = np.array([[a, b], [c, d]])
    vmax = max(n // 2, 1)
    im   = ax.imshow(tbl, cmap="Blues", aspect="auto", vmin=0, vmax=vmax)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels([f"CDKN2A deleted\n(n={cdn})", f"CDKN2A WT\n(n={n-cdn})"], fontsize=11)
    ax.set_yticklabels([f"H3K27M+\n(n={h3n})", f"H3K27M−\n(n={n-h3n})"], fontsize=11)
    ax.set_xlabel("CDKN2A Status", fontsize=12, labelpad=8)
    ax.set_ylabel("H3K27M Status", fontsize=12, labelpad=8)
    ax.set_title("Contingency Table", fontsize=12, fontweight="bold")

    for (row, col), val in np.ndenumerate(tbl):
        pct   = val / n * 100 if n > 0 else 0
        color = "white" if val > vmax * 0.6 else DARK_GREY
        ax.text(col, row, f"{val}\n({pct:.0f}%)",
                ha="center", va="center",
                fontsize=13, fontweight="bold", color=color)

    plt.colorbar(im, ax=ax, label="Sample count", shrink=0.8)

    if pval is not None and not (isinstance(pval, float) and math.isnan(pval)):
        ax.text(0.5, -0.24,
                f"p = {plbl}  (two-sided Fisher's exact)\n"
                "Mutual exclusivity — alternative oncogenic mechanisms",
                ha="center", transform=ax.transAxes,
                fontsize=10, color=RED_ACCENT, style="italic")

    # ── right: subgroup bar chart ─────────────────────────────────────────────
    ax2 = axes[1]
    counts     = [b, a, c, d]
    labels     = [
        f"H3K27M+\nonly\n(n={b})",
        f"Double-hit\n(n={a})",
        f"CDKN2A-del\nonly\n(n={c})",
        f"Neither\n(n={d})",
    ]
    bar_colors = [UCLA_BLUE, RED_ACCENT, UCLA_GOLD, LIGHT_GREY]
    drugs      = ["Epigenetic\ndrugs", "Triple\ncombination",
                  "CDK4/6\ninhibition", "Standard\napproach"]

    bars = ax2.bar(range(4), counts, color=bar_colors,
                   edgecolor=DARK_GREY, linewidth=0.8, width=0.6)

    ymax = max(counts) if counts else 1
    for i, (bar, cnt, lbl, drug) in enumerate(zip(bars, counts, labels, drugs)):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 cnt + ymax * 0.02, str(cnt),
                 ha="center", va="bottom", fontweight="bold", fontsize=12)
        if cnt > ymax * 0.12:
            tc = "white" if bar_colors[i] != LIGHT_GREY else DARK_GREY
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     cnt / 2, f"→ {drug}",
                     ha="center", va="center",
                     fontsize=8.5, color=tc, fontweight="bold",
                     multialignment="center")

    ax2.set_xticks(range(4))
    ax2.set_xticklabels(labels, fontsize=9.5)
    ax2.set_ylabel("Number of samples", fontsize=12)
    ax2.set_title("Molecular Subgroups → Drug Stratification",
                  fontsize=12, fontweight="bold")
    ax2.set_ylim(0, ymax * 1.2)

    plt.tight_layout()
    out = FIGURES_DIR / "fig1_cooccurrence.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Top N drugs, stacked score components
# ─────────────────────────────────────────────────────────────────────────────

def fig2_drug_rankings(data: dict, top_n: int = 12) -> None:
    candidates = data.get("top_candidates", [])
    if not candidates:
        print("  fig2 skipped — no candidates in results")
        return

    top = candidates[:top_n]
    drugs   = [c["name"] for c in top]
    comps   = {
        "Tissue/GSC (40%)":    [c.get("tissue_expression_score", 0) * 0.40 for c in top],
        "DepMap CRISPR (30%)": [c.get("depmap_score",            0) * 0.30 for c in top],
        "Escape bypass (20%)": [c.get("escape_bypass_score",     0) * 0.20 for c in top],
        "PPI network (10%)":   [c.get("ppi_score",               0) * 0.10 for c in top],
    }
    composite  = [c.get("score", 0) for c in top]
    bbb_status = [c.get("bbb_penetrance", "UNKNOWN") for c in top]
    failed     = [c.get("clinical_failure", False) for c in top]
    h3k27m     = [c.get("h3k27m_relevant", False) for c in top]

    stats    = data.get("stats", {})
    n_screen = stats.get("n_drugs_screened", "?")
    plbl     = stats.get("p_value_label", "")

    fig, ax = plt.subplots(figsize=(max(12, len(drugs) * 1.1), 6.5))

    x       = np.arange(len(drugs))
    width   = 0.18
    offsets = [-1.5, -0.5, 0.5, 1.5]
    colors  = [UCLA_BLUE, UCLA_GOLD, GREEN_ACCENT, PURPLE]

    for (label, vals), offset, color in zip(comps.items(), offsets, colors):
        ax.bar(x + offset * width, vals, width,
               label=label, color=color, alpha=0.85,
               edgecolor="white", linewidth=0.4)

    # Composite line on twin axis
    ax2 = ax.twinx()
    ax2.plot(x, composite, "o-", color=RED_ACCENT,
             linewidth=2.5, markersize=8, label="Composite score", zorder=5)
    ax2.set_ylim(0, 1.15)
    ax2.set_ylabel("Composite score", color=RED_ACCENT, fontsize=12)
    ax2.tick_params(axis="y", colors=RED_ACCENT)
    ax2.spines["right"].set_color(RED_ACCENT)
    ax2.spines["top"].set_visible(False)

    # Per-bar annotations: BBB, failure, H3K27M
    for i in range(len(drugs)):
        markers = []
        if failed[i]:
            markers.append(("✗", RED_ACCENT))
        if bbb_status[i] == "HIGH":
            markers.append(("●", GREEN_ACCENT))
        if h3k27m[i]:
            markers.append(("H", UCLA_BLUE))
        for j, (m, mc) in enumerate(markers):
            ax2.text(i, composite[i] + 0.04 + j * 0.07, m,
                     ha="center", fontsize=9, color=mc, fontweight="bold")

    # Annotate top hypothesis
    cb = data.get("confidence_breakdown")
    if cb and composite:
        combo = cb.get("drug_combo", "")
        conf  = cb.get("confidence", 0)
        ax2.annotate(
            f"{combo}\nConf: {conf:.2f}",
            xy=(0, composite[0]),
            xytext=(min(2, len(drugs) - 1), 1.02),
            fontsize=8.5, color=RED_ACCENT,
            arrowprops=dict(arrowstyle="->", color=RED_ACCENT, lw=1.5),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=RED_ACCENT, alpha=0.9),
        )

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    extra  = [
        mpatches.Patch(color=GREEN_ACCENT, label="● HIGH BBB penetrance"),
        mpatches.Patch(color=RED_ACCENT,   label="✗ Known GBM failure (penalised)"),
        mpatches.Patch(color=UCLA_BLUE,    label="H H3K27M-relevant"),
    ]
    ax.legend(h1 + h2 + extra, l1 + l2 + [p.get_label() for p in extra],
              loc="upper right", fontsize=8.5, framealpha=0.9, ncol=2)

    ax.set_xticks(x)
    ax.set_xticklabels(drugs, rotation=30, ha="right", fontsize=11)
    ax.set_ylabel("Weighted component score", fontsize=12)
    ax.set_title(
        f"Top {len(drugs)} Drug Candidates — Multi-Omic Composite Scoring\n"
        f"{n_screen} drugs screened  •  p = {plbl}",
        fontsize=12, fontweight="bold",
    )
    max_stacked = max(
        sum(comps[k][i] for k in comps) for i in range(len(drugs))
    ) if drugs else 0.6
    ax.set_ylim(0, max_stacked * 1.35)

    plt.tight_layout()
    out = FIGURES_DIR / "fig2_drug_rankings.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — DepMap vs tissue scatter, coloured by BBB penetrance
# ─────────────────────────────────────────────────────────────────────────────

def fig3_score_scatter(data: dict) -> None:
    candidates = data.get("top_candidates", [])
    if not candidates:
        print("  fig3 skipped — no candidates in results")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    fig.suptitle("Score Component Relationships — All Screened Candidates",
                 fontsize=13, fontweight="bold")

    pairs = [
        ("depmap_score",        "tissue_expression_score",
         "DepMap CRISPR",       "Single-cell GSC"),
        ("ppi_score",           "escape_bypass_score",
         "PPI Network",         "Escape Bypass"),
        ("depmap_score",        "score",
         "DepMap CRISPR",       "Composite Score"),
    ]

    for ax, (xkey, ykey, xlabel, ylabel) in zip(axes, pairs):
        xs     = [c.get(xkey, 0) for c in candidates]
        ys     = [c.get(ykey, 0) for c in candidates]
        colors = [BBB_COLORS.get(c.get("bbb_penetrance", "UNKNOWN"), "grey")
                  for c in candidates]
        names  = [c.get("name", "?") for c in candidates]

        ax.scatter(xs, ys, c=colors, s=65, alpha=0.75,
                   edgecolors=DARK_GREY, linewidths=0.4, zorder=3)

        # Label top 5 by composite score
        top5 = sorted(range(len(candidates)),
                      key=lambda i: candidates[i].get("score", 0),
                      reverse=True)[:5]
        for i in top5:
            ax.annotate(names[i], (xs[i], ys[i]),
                        textcoords="offset points", xytext=(5, 3),
                        fontsize=8, color=DARK_GREY, fontweight="bold")

        # Trend line
        if len(xs) > 3:
            try:
                z = np.polyfit(xs, ys, 1)
                xr = np.linspace(min(xs), max(xs), 100)
                ax.plot(xr, np.poly1d(z)(xr), "--",
                        color="grey", alpha=0.45, linewidth=1.2)
            except Exception:
                pass

        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_xlim(-0.05, 1.1)
        ax.set_ylim(-0.05, 1.1)

    legend_elems = [
        mpatches.Patch(facecolor=GREEN_ACCENT, label="HIGH BBB penetrance"),
        mpatches.Patch(facecolor=UCLA_GOLD,    label="MODERATE BBB penetrance"),
        mpatches.Patch(facecolor=RED_ACCENT,   label="LOW BBB penetrance"),
        mpatches.Patch(facecolor="grey",       label="UNKNOWN"),
    ]
    axes[2].legend(handles=legend_elems, loc="lower right",
                   fontsize=9, framealpha=0.9)

    plt.tight_layout()
    out = FIGURES_DIR / "fig3_score_scatter.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Top hypothesis confidence breakdown
# ─────────────────────────────────────────────────────────────────────────────

def fig4_confidence(data: dict) -> None:
    cb = data.get("confidence_breakdown")
    if not cb:
        print("  fig4 skipped — no confidence_breakdown in results")
        return

    dep_raw = cb.get("depmap_essentiality", 0)
    bbb_raw = cb.get("bbb_penetrance", 0)
    div_raw = cb.get("mechanistic_diversity", 0)
    conf    = cb.get("confidence", 0)
    combo   = cb.get("drug_combo", "Top hypothesis")
    priority= cb.get("priority", "")
    p_sig   = cb.get("statistical_significance", "")

    dep_w, bbb_w, div_w = dep_raw * 0.45, bbb_raw * 0.35, div_raw * 0.20
    total = dep_w + bbb_w + div_w

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(
        f"{combo}  —  Confidence Breakdown\n"
        f"Confidence = {conf:.2f}  •  Priority: {priority}  •  {p_sig}",
        fontsize=12, fontweight="bold", y=1.02,
    )

    # ── left: component bars ──────────────────────────────────────────────────
    ax = axes[0]
    comp_labels = ["DepMap\nEssentiality\n(×0.45)",
                   "BBB\nPenetrance\n(×0.35)",
                   "Target\nDiversity\n(×0.20)"]
    raw_vals    = [dep_raw, bbb_raw, div_raw]
    weighted    = [dep_w,   bbb_w,   div_w]
    bar_colors  = [UCLA_BLUE, UCLA_GOLD, GREEN_ACCENT]
    sources     = ["Broad Institute\nCRISPR (external)",
                   "Curated PK\nliterature (external)",
                   "Target Jaccard\noverlap (computed)"]

    bars = ax.bar(comp_labels, weighted, color=bar_colors,
                  edgecolor=DARK_GREY, linewidth=0.8, width=0.5)

    ymax = max(weighted) if weighted else 0.5
    for bar, raw, w, src, col in zip(bars, raw_vals, weighted, sources, bar_colors):
        ax.text(bar.get_x() + bar.get_width() / 2,
                w + ymax * 0.04,
                f"raw = {raw:.2f}  →  {w:.3f}",
                ha="center", va="bottom", fontsize=9.5, fontweight="bold")
        if w > ymax * 0.1:
            tc = "white" if col != UCLA_GOLD else DARK_GREY
            ax.text(bar.get_x() + bar.get_width() / 2,
                    w / 2, src,
                    ha="center", va="center",
                    fontsize=8, color=tc, fontweight="bold",
                    multialignment="center")

    ax.axhline(total, color=RED_ACCENT, linestyle="--", linewidth=2, alpha=0.8)
    ax.text(len(comp_labels) - 0.6, total + ymax * 0.04,
            f"Total = {total:.2f}",
            color=RED_ACCENT, fontsize=11, fontweight="bold")
    ax.set_ylabel("Weighted contribution to confidence", fontsize=11)
    ax.set_title("Confidence Components\n"
                 "(externally grounded — not self-referential)",
                 fontsize=11, fontweight="bold")
    ax.set_ylim(0, ymax * 1.8)

    # ── right: horizontal stacked bar for top 8 candidates ───────────────────
    ax2  = axes[1]
    top8 = data.get("top_candidates", [])[:8]

    if top8:
        names = [c["name"] for c in top8]
        tis   = [c.get("tissue_expression_score", 0) for c in top8]
        dep   = [c.get("depmap_score",            0) for c in top8]
        esc   = [c.get("escape_bypass_score",     0) for c in top8]
        ppi   = [c.get("ppi_score",               0) for c in top8]

        y = np.arange(len(names))
        h = 0.55
        ax2.barh(y, [d * 0.30 for d in dep], h, color=UCLA_GOLD,    label="DepMap (30%)")
        ax2.barh(y, [t * 0.40 for t in tis], h,
                 left=[d * 0.30 for d in dep],
                 color=UCLA_BLUE,    label="Tissue/GSC (40%)")
        left2 = [d * 0.30 + t * 0.40 for d, t in zip(dep, tis)]
        ax2.barh(y, [e * 0.20 for e in esc], h,
                 left=left2, color=GREEN_ACCENT, label="Escape (20%)")
        left3 = [l + e * 0.20 for l, e in zip(left2, esc)]
        ax2.barh(y, [p * 0.10 for p in ppi], h,
                 left=left3, color=PURPLE, label="PPI (10%)")

        ax2.set_yticks(y)
        ax2.set_yticklabels(names, fontsize=10)
        ax2.set_xlabel("Composite score", fontsize=11)
        ax2.set_xlim(0, 1.05)
        ax2.set_title("Score Composition — Top 8 Candidates",
                      fontsize=11, fontweight="bold")
        ax2.legend(loc="lower right", fontsize=9, framealpha=0.9)
        ax2.axhline(len(names) - 1, color=RED_ACCENT,
                    linestyle=":", linewidth=1.5, alpha=0.6)

    plt.tight_layout()
    out = FIGURES_DIR / "fig4_confidence.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("generate_figures.py — reading from pipeline results")
    print("=" * 55)

    results = load()
    print()

    fig1_cooccurrence(results)
    fig2_drug_rankings(results, top_n=12)
    fig3_score_scatter(results)
    fig4_confidence(results)

    print("\n" + "=" * 55)
    print("Done. Add to README.md:")
    for name in ["fig1_cooccurrence", "fig2_drug_rankings",
                 "fig3_score_scatter", "fig4_confidence"]:
        print(f"  ![{name}](figures/{name}.png)")