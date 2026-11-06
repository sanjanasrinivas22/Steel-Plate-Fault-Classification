"""
Hybrid vs non-hybrid soft computing on the Steel Plates Faults subset.

Non-hybrid : single MLP (32 ReLU units, back-propagation)
Hybrid     : GWO-NFC (Grey-Wolf-Optimised Neuro-Fuzzy Committee)
             = fuzzified-input MLP committee + GWO-weighted Fuzzy KNN

Settings:
  - 5 stratified 70/30 splits
  - 5 MLPs in the committee, 300 iterations each
  - GWO: 12 wolves x 15 iterations, 3-fold CV on training data only

Run from the repository root:
    python src/hybrid_vs_nonhybrid.py

Input  : data/steel_faults_subset.csv
Outputs: results/per_split_results.csv, results/results.json
"""
import json, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "steel_faults_subset.csv"
RESULTS = ROOT / "results"

CLASSES = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']
N_SPLITS = 5
N_BAG = 5
MLP_ITER = 300
GWO_WOLVES = 12
GWO_ITERS = 15
CV_FOLDS = 3


# ----------------------------------------------------------------------------- data
def load_subset():
    df = pd.read_csv(DATA)
    feature_cols = [c for c in df.columns if c != "Fault"]
    X = df[feature_cols].values.astype(float)
    y = df["Fault"].map({c: i for i, c in enumerate(CLASSES)}).values
    return X, y


def preprocess(Xtr, Xte):
    """Signed log to tame skewed features, then z-score (fit on train only)."""
    slog = lambda X: np.sign(X) * np.log1p(np.abs(X))
    sc = StandardScaler().fit(slog(Xtr))
    return sc.transform(slog(Xtr)), sc.transform(slog(Xte))


# ------------------------------------------------------------- hybrid building blocks
def fuzzify(Xtr, Xte, n_sets=3):
    """Low / Medium / High Gaussian memberships per feature (fit on train only)."""
    out = []
    for A in (Xtr, Xte):
        cols = []
        for j in range(Xtr.shape[1]):
            c = np.quantile(Xtr[:, j], np.linspace(0, 1, n_sets + 2)[1:-1])
            s = (np.quantile(Xtr[:, j], .9) - np.quantile(Xtr[:, j], .1)) / n_sets + 1e-9
            cols.append(np.exp(-((A[:, [j]] - c) ** 2) / (2 * s * s)))
        out.append(np.hstack([A, np.hstack(cols)]))
    return out


def neural_committee(Atr, ytr, Ate, seed):
    """Bagged committee of MLPs; returns averaged class probabilities."""
    rng = np.random.default_rng(seed)
    P = np.zeros((len(Ate), len(CLASSES)))
    for k in range(N_BAG):
        i = rng.choice(len(Atr), len(Atr))
        m = MLPClassifier((32,), max_iter=MLP_ITER, random_state=seed * 100 + k).fit(Atr[i], ytr[i])
        p = np.zeros_like(P); p[:, m.classes_] = m.predict_proba(Ate); P += p
    return P / N_BAG


def fuzzy_knn(Xtr, ytr, Xte, k, m):
    """Keller's Fuzzy KNN: graded class memberships from distance-weighted neighbours."""
    d, idx = NearestNeighbors(n_neighbors=k).fit(Xtr).kneighbors(Xte)
    w = 1.0 / (d ** (2.0 / (m - 1.0)) + 1e-9)
    P = np.stack([(w * (ytr[idx] == c)).sum(1) for c in range(len(CLASSES))], 1)
    return P / P.sum(1, keepdims=True)


def gwo_feature_weights(a, ytr, seed):
    """Grey Wolf Optimizer: evolves 27 feature weights + k and m of the Fuzzy KNN."""
    D = a.shape[1]
    folds = list(StratifiedKFold(CV_FOLDS, shuffle=True, random_state=seed).split(a, ytr))
    decode = lambda p: (p[:D], int(round(3 + p[D] * 22)), 1.2 + p[D + 1] * 2.8)

    def fitness(p):
        fw, k, m = decode(p)
        if fw.sum() < 1e-6: return 0.0
        return np.mean([(fuzzy_knn(a[tr] * fw, ytr[tr], a[va] * fw, k, m).argmax(1) == ytr[va]).mean()
                        for tr, va in folds])

    rng = np.random.default_rng(seed); dim = D + 2
    X = rng.random((GWO_WOLVES, dim)); F = np.array([fitness(x) for x in X]); curve = [F.max()]
    for t in range(GWO_ITERS):
        order = np.argsort(-F)
        alpha, beta, delta = X[order[0]].copy(), X[order[1]].copy(), X[order[2]].copy()
        a_ = 2 - 2 * t / GWO_ITERS
        for i in range(GWO_WOLVES):
            new = np.zeros(dim)
            for leader in (alpha, beta, delta):
                A = 2 * a_ * rng.random(dim) - a_; C = 2 * rng.random(dim)
                new += leader - A * np.abs(C * leader - X[i])
            X[i] = np.clip(new / 3, 0, 1); F[i] = fitness(X[i])
        curve.append(F.max())
    return decode(X[np.argmax(F)]), curve


# ------------------------------------------------------------------------ experiment
def main():
    RESULTS.mkdir(exist_ok=True)
    X, y = load_subset()
    print(f"Subset: {X.shape[0]} samples, {X.shape[1]} features, {len(CLASSES)} classes")
    rows = []
    for s in range(N_SPLITS):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=s)

        # NON-HYBRID: single MLP
        sc = StandardScaler().fit(Xtr)
        mlp = MLPClassifier((32,), max_iter=MLP_ITER, random_state=s).fit(sc.transform(Xtr), ytr)
        p_mlp = mlp.predict(sc.transform(Xte))

        # HYBRID: GWO-NFC
        a, b = preprocess(Xtr, Xte); A, B = fuzzify(a, b)
        Pn = neural_committee(A, ytr, B, s)
        (fw, k, m), _ = gwo_feature_weights(a, ytr, s)
        Pf = fuzzy_knn(a * fw, ytr, b * fw, k, m)
        p_hyb = (0.5 * Pn + 0.5 * Pf).argmax(1)

        rows.append(dict(
            split=s, k=k, m=round(m, 2), strong_features=int((fw > 0.5).sum()),
            acc_MLP=(p_mlp == yte).mean(), f1_MLP=f1_score(yte, p_mlp, average="macro"),
            acc_Hybrid=(p_hyb == yte).mean(), f1_Hybrid=f1_score(yte, p_hyb, average="macro"),
        ))
        r = rows[-1]
        print(f"split {s}: Non-Hybrid {r['acc_MLP']:.3f} | Hybrid {r['acc_Hybrid']:.3f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "per_split_results.csv", index=False)
    diff_acc = df["acc_Hybrid"] - df["acc_MLP"]
    diff_f1 = df["f1_Hybrid"] - df["f1_MLP"]
    p_val = wilcoxon(df["acc_Hybrid"], df["acc_MLP"], alternative="greater").pvalue

    print("\n================ HYBRID vs NON-HYBRID ================")
    print(f"{'Model':<22}{'Accuracy (mean±std)':>24}{'F1 (mean±std)':>24}")
    print(f"{'Non-hybrid':<22}"
          f"{df.acc_MLP.mean():>16.4f} ± {df.acc_MLP.std():.4f}"
          f"{df.f1_MLP.mean():>16.4f} ± {df.f1_MLP.std():.4f}")
    print(f"{'Hybrid (GWO-NFC)':<22}"
          f"{df.acc_Hybrid.mean():>16.4f} ± {df.acc_Hybrid.std():.4f}"
          f"{df.f1_Hybrid.mean():>16.4f} ± {df.f1_Hybrid.std():.4f}")
    print("-" * 70)
    print(f"Hybrid wins {int((diff_acc > 0).sum())} of {N_SPLITS} splits; "
          f"one-sided Wilcoxon p = {p_val:.4f}")

    summary = {
        "n_samples": int(len(y)), "n_splits": N_SPLITS,
        "mlp": {"acc_mean": df.acc_MLP.mean(), "acc_std": df.acc_MLP.std(),
                "f1_mean": df.f1_MLP.mean(), "f1_std": df.f1_MLP.std()},
        "hybrid": {"acc_mean": df.acc_Hybrid.mean(), "acc_std": df.acc_Hybrid.std(),
                   "f1_mean": df.f1_Hybrid.mean(), "f1_std": df.f1_Hybrid.std()},
        "diff_acc_mean": diff_acc.mean(), "diff_f1_mean": diff_f1.mean(),
        "hybrid_wins": int((diff_acc > 0).sum()), "wilcoxon_p_one_sided": p_val,
        "per_split": rows,
    }
    with open(RESULTS / "results.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nSaved: {RESULTS / 'per_split_results.csv'}, {RESULTS / 'results.json'}")


if __name__ == "__main__":
    main()
