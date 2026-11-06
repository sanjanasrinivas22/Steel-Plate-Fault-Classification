"""
Builds all result figures used in the report from results/detailed_results.json.

Run from the repository root (after src/detailed_analysis.py):
    python src/make_figures.py

Output: figures/*.png
"""
import json, os, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.model_selection import train_test_split
import hybrid_vs_nonhybrid as S

ROOT = Path(__file__).resolve().parents[1]
D = json.load(open(ROOT / "results" / "detailed_results.json"))
R = {"report": {n: {"cm": v["confusion"], "f1": v["f1"]} for n, v in D["pooled_report"].items()},
     "curves": D["gwo_curves"], "weights": D["gwo_feature_weights"], "per_split": D["per_split"]}
OUT = str(ROOT / "figures") + "/"; os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10, "figure.dpi": 200,
                     "axes.spines.top": False, "axes.spines.right": False})
SHORT = ["Pastry", "Z_Scratch", "K_Scatch", "Stains", "Dirtiness", "Bumps", "Other"]
GREY, BLUE, LBLUE, GREEN = "#8f8d87", "#2a78d6", "#a9c7ef", "#1baf7a"


def box(ax, x, y, w, h, text, fc, ec="#5b7fa6", fs=8.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.04",
                                fc=fc, ec=ec, lw=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color="#1b1b1b")


def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="-|>", color="#4a4a4a", lw=1))


# ---------------- Figure 4.1 overall block diagram
fig, ax = plt.subplots(figsize=(8, 4.6)); ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
box(ax, 2.6, 5.15, 4.8, 0.65, "Steel Plates Faults (UCI #198)\n1941 plates · 27 features · 7 fault classes", "#eef3fa")
box(ax, 2.6, 4.1, 4.8, 0.65, "Stratified subset: 1000 plates (seed 42)\nsteel_faults_subset.csv", "#eef3fa")
box(ax, 2.6, 3.05, 4.8, 0.65, "5 stratified 70/30 splits (random_state = 0 … 4)\n700 training plates · 300 test plates per split", "#eef3fa")
arrow(ax, 5, 5.15, 5, 4.75); arrow(ax, 5, 4.1, 5, 3.7)
box(ax, 0.3, 1.45, 3.6, 1.0, "Non-hybrid branch\nz-score → MLP (32 ReLU units)", "#f1efe9", ec="#8f8d87")
box(ax, 5.2, 1.45, 4.5, 1.0, "Hybrid branch (GWO-NFC)\nsigned log + z-score → fuzzifier → MLP committee\n+ GWO-weighted Fuzzy KNN → average", "#e3eefb")
arrow(ax, 4.2, 3.05, 2.1, 2.45); arrow(ax, 5.8, 3.05, 7.4, 2.45)
box(ax, 2.6, 0.15, 4.8, 0.7, "Same 300 test plates per split → accuracy, macro F1,\nper-class report, confusion matrix, Wilcoxon test", "#eef3fa")
arrow(ax, 2.1, 1.45, 4.0, 0.85); arrow(ax, 7.4, 1.45, 6.0, 0.85)
fig.savefig(OUT + "fig4_1_block.png", bbox_inches="tight", facecolor="white"); plt.close(fig)

# ---------------- Figure 5.1 MLP architecture
fig, ax = plt.subplots(figsize=(8, 1.5)); ax.set_xlim(0, 10); ax.set_ylim(0, 1.6); ax.axis("off")
items = [("27 inputs\n(z-score)", "#eef3fa"), ("Dense 32\nReLU", "#fbe6d4"), ("Dense 7\nsoftmax", "#fbe6d4"),
         ("argmax", "#fbe6d4"), ("fault\nclass", "#e8f4ea")]
for i, (t, c) in enumerate(items):
    box(ax, 0.2 + i * 2, 0.35, 1.5, 0.9, t, c, ec="#c98c5a")
    if i: arrow(ax, 0.2 + (i - 1) * 2 + 1.5, 0.8, 0.2 + i * 2, 0.8)
fig.savefig(OUT + "fig5_1_mlp.png", bbox_inches="tight", facecolor="white"); plt.close(fig)


# ---------------- confusion matrices
def cm_plot(name, title, fn):
    C = np.array(R["report"][name]["cm"])
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    im = ax.imshow(C, cmap="Blues")
    for i in range(7):
        for j in range(7):
            ax.text(j, i, C[i, j], ha="center", va="center", fontsize=8,
                    color="white" if C[i, j] > C.max() * 0.55 else "#1b1b1b")
    ax.set_xticks(range(7), SHORT, rotation=40, ha="right", fontsize=8); ax.set_yticks(range(7), SHORT, fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title, fontsize=10)
    for s in ax.spines.values(): s.set_visible(False)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.savefig(OUT + fn, bbox_inches="tight", facecolor="white"); plt.close(fig)


cm_plot("MLP", "MLP confusion matrix (5 test splits pooled)", "fig5_2_cm_mlp.png")
cm_plot("Hybrid", "GWO-NFC confusion matrix (5 test splits pooled)", "fig6_4_cm_hybrid.png")

# ---------------- Figure 6.1 hybrid architecture
fig, ax = plt.subplots(figsize=(8.2, 4.4)); ax.set_xlim(0, 10); ax.set_ylim(0, 5.4); ax.axis("off")
box(ax, 0.1, 2.3, 1.5, 0.8, "27 raw\nfeatures", "#eef3fa")
box(ax, 2.0, 2.3, 1.6, 0.8, "signed log\n+ z-score", "#eef3fa")
box(ax, 4.0, 3.9, 2.3, 0.9, "Fuzzifier\nLow / Medium / High\n(27 + 81 = 108 inputs)", "#e4dcf7", ec="#6250d6")
box(ax, 4.0, 2.55, 2.3, 0.9, "Committee of 5 MLPs\n(bootstrap samples,\n32 ReLU units each)", "#e4dcf7", ec="#6250d6")
box(ax, 4.0, 1.05, 2.3, 0.9, "Fuzzy KNN\n(Keller), weighted\nfeatures, k, m", "#dff2ea", ec="#1baf7a")
box(ax, 4.0, 0.0, 2.3, 0.65, "Grey Wolf Optimizer\n12 wolves × 15 iterations", "#dde9f8", ec="#2a78d6")
box(ax, 7.0, 1.8, 1.6, 1.0, "Average\n0.5·P_neural +\n0.5·P_fuzzy", "#cfe2f8", ec="#2a78d6")
box(ax, 8.9, 1.95, 1.0, 0.7, "fault\nclass", "#e8f4ea")
arrow(ax, 1.6, 2.7, 2.0, 2.7); arrow(ax, 3.6, 2.9, 4.0, 4.3); arrow(ax, 3.6, 2.5, 4.0, 1.5)
arrow(ax, 5.15, 3.9, 5.15, 3.45); arrow(ax, 5.15, 0.65, 5.15, 1.05)
arrow(ax, 6.3, 3.0, 7.0, 2.55); arrow(ax, 6.3, 1.5, 7.0, 2.05); arrow(ax, 8.6, 2.3, 8.9, 2.3)
ax.text(6.45, 4.35, "neural branch", fontsize=8, color="#6250d6", style="italic")
ax.text(6.45, 0.25, "fuzzy branch tuned on\ntraining data only (3-fold CV)", fontsize=8, color="#2a78d6", style="italic")
fig.savefig(OUT + "fig6_1_hybrid.png", bbox_inches="tight", facecolor="white"); plt.close(fig)

# ---------------- Figure 6.2 fuzzy membership functions (split 0, training data)
X, y = S.load_subset(); fc = D["features"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
a, b = S.preprocess(Xtr, Xte)
fig, axes = plt.subplots(1, 3, figsize=(9, 2.6), sharey=True)
for ax, fname in zip(axes, ["LogOfAreas", "Luminosity_Index", "Steel_Plate_Thickness"]):
    j = fc.index(fname); col = a[:, j]
    c = np.quantile(col, [0.25, 0.5, 0.75]); s = (np.quantile(col, .9) - np.quantile(col, .1)) / 3
    xs = np.linspace(col.min(), col.max(), 400)
    for ci, lab, colr in zip(c, ["Low", "Medium", "High"], ["#6250d6", "#1baf7a", "#d9822b"]):
        ax.plot(xs, np.exp(-((xs - ci) ** 2) / (2 * s * s)), color=colr, lw=1.8, label=lab)
    ax.set_title(fname.replace("_", " "), fontsize=9); ax.set_xlabel("standardised value", fontsize=8)
    ax.tick_params(labelsize=7)
axes[0].set_ylabel("membership degree"); axes[0].legend(fontsize=7, frameon=False)
fig.tight_layout(); fig.savefig(OUT + "fig6_2_membership.png", bbox_inches="tight", facecolor="white"); plt.close(fig)

# ---------------- Figure 6.3 GWO convergence
C = np.array(R["curves"]) * 100
fig, ax = plt.subplots(figsize=(6, 3.2))
for i, c in enumerate(C): ax.plot(c, color=LBLUE, lw=1)
ax.plot(C.mean(0), color=BLUE, lw=2.2, label="mean of 5 splits")
ax.set_xlabel("GWO iteration"); ax.set_ylabel("best 3-fold CV accuracy (%)"); ax.grid(axis="y", color="#e4e3df")
ax.legend(frameon=False, loc="lower right")
fig.savefig(OUT + "fig6_3_gwo.png", bbox_inches="tight", facecolor="white"); plt.close(fig)

# ---------------- Figure 6.5 GWO feature weights
w = np.mean(R["weights"], 0); o = np.argsort(w)
fig, ax = plt.subplots(figsize=(6.4, 6))
ax.barh(range(27), w[o], color=[BLUE if v > 0.5 else LBLUE for v in w[o]], height=0.7)
ax.set_yticks(range(27), [fc[i].replace("_", " ") for i in o], fontsize=7.5)
ax.axvline(0.5, color="#4a4a4a", lw=0.8, ls="--"); ax.set_xlim(0, 1)
ax.set_xlabel("mean GWO feature weight over 5 splits"); ax.grid(axis="x", color="#e4e3df"); ax.set_axisbelow(True)
fig.savefig(OUT + "fig6_5_weights.png", bbox_inches="tight", facecolor="white"); plt.close(fig)

# ---------------- Figure 7.1 per-class F1 MLP vs hybrid
fm = np.array(R["report"]["MLP"]["f1"]); fh = np.array(R["report"]["Hybrid"]["f1"])
fig, ax = plt.subplots(figsize=(7.2, 3.4)); xx = np.arange(7); wd = 0.38
ax.bar(xx - wd / 2, fm, wd - 0.03, color=GREY, label="MLP (non-hybrid)")
ax.bar(xx + wd / 2, fh, wd - 0.03, color=BLUE, label="GWO-NFC (hybrid)")
for i in range(7): ax.text(xx[i] + wd / 2, fh[i] + 0.01, f"{fh[i]-fm[i]:+.2f}", ha="center", fontsize=7.5)
ax.set_xticks(xx, SHORT, fontsize=8.5); ax.set_ylim(0, 1.08); ax.set_ylabel("F1-score")
ax.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=8.5); ax.grid(axis="y", color="#e4e3df"); ax.set_axisbelow(True)
fig.savefig(OUT + "fig7_1_perclass.png", bbox_inches="tight", facecolor="white"); plt.close(fig)

# ---------------- Figure 7.2 per-split paired accuracy
P = pd.DataFrame(R["per_split"])
fig, ax = plt.subplots(figsize=(6.4, 3.2)); s = np.arange(1, 6)
for i in range(5): ax.plot([s[i], s[i]], [P.acc_MLP[i] * 100, P.acc_Hybrid[i] * 100], color="#d4d3cf", lw=2.5, zorder=1)
ax.scatter(s, P.acc_MLP * 100, s=55, color=GREY, zorder=2, label="MLP (non-hybrid)")
ax.scatter(s, P.acc_Hybrid * 100, s=55, color=BLUE, zorder=3, label="GWO-NFC (hybrid)")
ax.set_xticks(s, [f"split {i-1}" for i in s]); ax.set_ylabel("test accuracy (%)"); ax.grid(axis="y", color="#e4e3df")
ax.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=8.5); ax.set_axisbelow(True)
fig.savefig(OUT + "fig7_2_splits.png", bbox_inches="tight", facecolor="white"); plt.close(fig)
print("figures written to", OUT)
