"""Regenerate the report figures from data/processed and results/.

Run from the repo root after the pipeline: `uv run python reports/make_figures.py`.
Writes PNGs to reports/figures/.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from vgrec.config import RESULTS_DIR, T1_MS, T2_MS  # noqa: E402
from vgrec.data import load_dataset  # noqa: E402
from vgrec.protocol import make_split  # noqa: E402

OUT = Path(__file__).parent / "figures"

# Categorical slots in fixed order; one color per model across every figure.
MODELS = ["popularity", "als", "content", "ncf", "hybrid", "hybrid_pop"]
LABELS = {"popularity": "Popularity", "als": "ALS", "content": "Content (TF-IDF)", "ncf": "NeuMF",
          "hybrid": "Hybrid (CF + content)", "hybrid_pop": "Hybrid + popularity"}
COLORS = dict(zip(MODELS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"],
                  strict=True))
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#ffffff"
BLUE_RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]  # ordinal steps 250/400/550/700

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK_2,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left",
    "xtick.color": INK_2, "ytick.color": INK_2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "legend.frameon": False,
    "savefig.dpi": 200, "savefig.bbox": "tight",
})


def _bar_kw(color):
    return {"color": color, "edgecolor": SURFACE, "linewidth": 1.0}  # 2px-style surface gap between bars


def _save(fig, name):
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)


def _summary(model: str, stage: str) -> dict:
    return json.loads((RESULTS_DIR / model / f"{stage}_summary.json").read_text())


def eda_figures(ds) -> None:
    inter = ds.interactions

    shares = inter["rating"].value_counts(normalize=True).sort_index() * 100
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    bars = ax.bar([f"{int(r)}★" for r in shares.index], shares.to_numpy(), width=0.6,
                  **_bar_kw(COLORS["popularity"]))
    ax.bar_label(bars, labels=[f"{v:.1f}%" for v in shares], color=INK_2, fontsize=8, padding=2)
    ax.set(title="Rating distribution", ylabel="share of interactions (%)", ylim=(0, 72))
    ax.grid(axis="x", visible=False)
    _save(fig, "fig1_ratings")

    per_user = inter.groupby("user_idx").size().clip(upper=20)
    counts = per_user.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5.4, 2.6))
    ax.bar([str(i) if i < 20 else "20+" for i in counts.index], counts.to_numpy() / 1000, width=0.75,
           **_bar_kw(COLORS["popularity"]))
    ax.set(title="Interactions per user", xlabel="interactions (all splits)", ylabel="users (thousands)")
    ax.grid(axis="x", visible=False)
    ax.annotate("median = 6", xy=(1, counts.loc[6] / 1000), xytext=(5.5, 25), color=INK_2, fontsize=8,
                arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8})
    _save(fig, "fig2_user_activity")

    monthly = inter.set_index(pd.to_datetime(inter["timestamp"], unit="ms")).resample("MS").size()
    monthly = monthly[monthly.index >= "2005-01-01"]
    fig, ax = plt.subplots(figsize=(6.5, 2.4))
    ax.plot(monthly.index, monthly.to_numpy() / 1000, color=COLORS["popularity"], lw=1.6)
    for t, name in ((T1_MS, "valid starts\n(Aug 2021)"), (T2_MS, "test starts\n(Jul 2022)")):
        x = pd.to_datetime(t, unit="ms")
        ax.axvline(x, color=INK_2, ls="--", lw=0.9)
        ax.text(x, ax.get_ylim()[1] * 0.97, name, ha="right" if "valid" in name else "left", va="top",
                color=INK_2, fontsize=7.5)
    ax.set(title="Interactions per month (2005 onward)", ylabel="interactions (thousands)")
    _save(fig, "fig3_monthly")

    stats = {stage: make_split(ds, stage).stats for stage in ("valid", "test")}
    parts = [("rankable (kept)", "kept_rows", COLORS["popularity"]),
             ("user has no history", "cold_user_rows", COLORS["als"]),
             ("item never seen in history", "cold_item_rows", COLORS["content"])]
    fig, ax = plt.subplots(figsize=(6.5, 1.9))
    for row, stage in enumerate(("test", "valid")):
        left = 0.0
        total = stats[stage]["target_rows"]
        for label, key, color in parts:
            share = stats[stage][key] / total * 100
            ax.barh(row, share, left=left, height=0.55, label=label if row == 0 else None, **_bar_kw(color))
            ax.text(left + share / 2, row, f"{share:.1f}%", ha="center", va="center", fontsize=8,
                    fontweight="bold", color=INK if key == "cold_item_rows" else SURFACE)
            left += share
    ax.set_yticks([0, 1], ["test stage", "valid stage"])
    ax.set(title="Where the target interactions go", xlabel="share of target rows (%)", xlim=(0, 100))
    ax.grid(axis="y", visible=False)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.38), fontsize=8)
    _save(fig, "fig4_cold_start")


def result_figures() -> None:
    test = {m: _summary(m, "test") for m in MODELS}

    order = sorted(MODELS, key=lambda m: test[m]["ndcg@10"])
    fig, ax = plt.subplots(figsize=(6.0, 2.6))
    vals = [test[m]["ndcg@10"] for m in order]
    bars = ax.barh([LABELS[m] for m in order], vals, height=0.6, color=[COLORS[m] for m in order],
                   edgecolor=SURFACE, linewidth=1.0)
    ax.bar_label(bars, labels=[f"{v:.4f}" for v in vals], color=INK_2, fontsize=8, padding=3)
    ax.set(title="Test NDCG@10 by model (10,049 users)", xlabel="NDCG@10", xlim=(0, 0.025))
    ax.grid(axis="y", visible=False)
    _save(fig, "fig5_test_ndcg")

    by_hist = pd.read_csv(RESULTS_DIR / "ndcg_by_history.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(6.5, 2.9))
    x = np.arange(len(by_hist))
    width = 0.13
    for i, m in enumerate(MODELS):
        ax.bar(x + (i - 2.5) * width, by_hist[m], width=width, label=LABELS[m], **_bar_kw(COLORS[m]))
    ax.set_xticks(x, [f"{b}\n(n = {n:,})" for b, n in by_hist["users"].items()])
    ax.set(title="Test NDCG@10 by history length", xlabel="history interactions per user",
           ylabel="NDCG@10")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.32), fontsize=8)
    _save(fig, "fig6_by_history")

    rows = []
    for m in ("als", "content", "ncf"):
        grid = pd.read_csv(RESULTS_DIR / m / "valid_grid.csv")
        rows.append((m, grid.loc[grid["half_life_days"] == 0, "ndcg@10"].max(),
                     grid.loc[grid["half_life_days"] > 0, "ndcg@10"].max()))
    pop = pd.read_csv(RESULTS_DIR / "popularity" / "valid_grid.csv")
    rows.insert(0, ("popularity", pop.loc[pop["window_days"] == 0, "ndcg@10"].max(),
                    pop.loc[pop["window_days"] > 0, "ndcg@10"].max()))
    fig, ax = plt.subplots(figsize=(6.0, 2.6))
    x = np.arange(len(rows))
    b0 = ax.bar(x - 0.19, [r[1] for r in rows], width=0.38, label="all history weighted equally",
                **_bar_kw("#a9a8a2"))
    b1 = ax.bar(x + 0.19, [r[2] for r in rows], width=0.38, label="recent interactions weighted more",
                **_bar_kw(COLORS["popularity"]))
    ax.bar_label(b0, labels=[f"{r[1]:.4f}" for r in rows], color=INK_2, fontsize=7.5, padding=2)
    ax.bar_label(b1, labels=[f"{r[2]:.4f}\n(+{r[2] / r[1] - 1:.0%})" for r in rows], color=INK_2,
                 fontsize=7.5, padding=2)
    ax.set_xticks(x, [LABELS[r[0]] for r in rows])
    ax.set(title="Effect of recency weighting (best valid NDCG@10)", ylabel="valid NDCG@10", ylim=(0, 0.024))
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=8)
    _save(fig, "fig7_time_decay")

    grid = pd.read_csv(RESULTS_DIR / "hybrid" / "valid_grid.csv")
    fig, ax = plt.subplots(figsize=(6.0, 2.7))
    for color, (k, g) in zip(BLUE_RAMP, grid.groupby("adaptive_k"), strict=True):
        g = g.sort_values("alpha")
        label = "fixed weight (k = 0)" if k == 0 else f"history-adaptive, k = {k:g}"
        ax.plot(g["alpha"], g["ndcg@10"], color=color, lw=2, marker="o", ms=4.5, label=label)
    ax.axhline(_summary("als", "valid")["ndcg@10"], color=COLORS["als"], ls="--", lw=1)
    ax.text(0.302, _summary("als", "valid")["ndcg@10"] + 0.0002, "ALS alone", color=INK_2, fontsize=7.5)
    ax.set(title="Hybrid weight sweep (valid stage)", xlabel="α (weight on the CF score)",
           ylabel="valid NDCG@10")
    ax.legend(loc="lower right", fontsize=7.5)
    _save(fig, "fig8_hybrid_sweep")

    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    for m in MODELS:
        ax.scatter(test[m]["personalization"], test[m]["ndcg@10"], s=60, color=INK_2, edgecolor=SURFACE,
                   linewidth=1.5, zorder=3)
        dx, ha = (-0.02, "right") if test[m]["personalization"] > 0.5 else (0.025, "left")
        dy = {"ncf": 0.0006, "popularity": -0.0006}.get(m, 0.0)  # the two sit almost on top of each other
        ax.annotate(f"{LABELS[m]}\n(coverage {test[m]['coverage']:.2%})",
                    (test[m]["personalization"], test[m]["ndcg@10"]),
                    xytext=(test[m]["personalization"] + dx, test[m]["ndcg@10"] + dy), ha=ha, va="center",
                    fontsize=7.5, color=INK)
    ax.set(title="Accuracy vs. personalization (test stage)",
           xlabel="personalization (1 = every user gets a different list)", ylabel="NDCG@10",
           xlim=(-0.05, 1.05), ylim=(0.008, 0.024))
    _save(fig, "fig9_personalization")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    eda_figures(load_dataset())
    result_figures()
    print(f"wrote {len(list(OUT.glob('*.png')))} figures to {OUT}")


if __name__ == "__main__":
    main()
