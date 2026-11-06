"""
Detailed analysis for the report. Re-runs exactly the same experiment as
hybrid_vs_nonhybrid.py (same splits, same seeds, same models) and additionally
records:
  - per-class precision / recall / F1 and confusion matrices (5 test splits pooled)
  - an ablation: fuzzified MLP committee alone, Fuzzy KNN alone, GWO-weighted Fuzzy KNN alone
  - GWO convergence curves and chosen feature weights
  - training times

Run from the repository root:
    python src/detailed_analysis.py

Output: results/detailed_results.json, results/detailed_per_split.csv
"""
import json, time, warnings

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

import hybrid_vs_nonhybrid as H

warnings.filterwarnings("ignore")
MODELS = ["MLP", "Hybrid", "Committee", "FKNN_plain", "FKNN_gwo"]


def main():
    H.RESULTS.mkdir(exist_ok=True)
    df_raw = pd.read_csv(H.DATA)
    features = [c for c in df_raw.columns if c != "Fault"]
    X, y = H.load_subset()
    rows, curves, weights, y_all = [], [], [], []
    preds = {n: [] for n in MODELS}

    for s in range(H.N_SPLITS):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=s)

        t = time.time()
        sc = StandardScaler().fit(Xtr)
        mlp = MLPClassifier((32,), max_iter=H.MLP_ITER, random_state=s).fit(sc.transform(Xtr), ytr)
        p_mlp = mlp.predict(sc.transform(Xte)); t_mlp = time.time() - t

        t = time.time()
        a, b = H.preprocess(Xtr, Xte); A, B = H.fuzzify(a, b)
        t1 = time.time(); Pn = H.neural_committee(A, ytr, B, s); t_comm = time.time() - t1
        t1 = time.time(); (fw, k, m), curve = H.gwo_feature_weights(a, ytr, s); t_gwo = time.time() - t1
        Pf = H.fuzzy_knn(a * fw, ytr, b * fw, k, m)
        p_hyb = (0.5 * Pn + 0.5 * Pf).argmax(1); t_hyb = time.time() - t
        Pf0 = H.fuzzy_knn(a, ytr, b, 10, 2.0)          # ablation: Fuzzy KNN without GWO

        split_preds = {"MLP": p_mlp, "Hybrid": p_hyb, "Committee": Pn.argmax(1),
                       "FKNN_plain": Pf0.argmax(1), "FKNN_gwo": Pf.argmax(1)}
        r = dict(split=s, k=k, m=round(m, 2), strong_features=int((fw > 0.5).sum()),
                 t_mlp=t_mlp, t_hybrid=t_hyb, t_committee=t_comm, t_gwo=t_gwo,
                 gwo_cv_start=curve[0], gwo_cv_end=curve[-1])
        for n, p in split_preds.items():
            preds[n].extend(p.tolist())
            r[f"acc_{n}"] = float((p == yte).mean())
            r[f"f1_{n}"] = float(f1_score(yte, p, average="macro"))
        y_all.extend(yte.tolist()); rows.append(r); curves.append(curve); weights.append(fw.tolist())
        print(f"split {s}: MLP {r['acc_MLP']:.3f} | Hybrid {r['acc_Hybrid']:.3f}  (k={k}, m={m:.2f})", flush=True)

    df = pd.DataFrame(rows); df.to_csv(H.RESULTS / "detailed_per_split.csv", index=False)
    y_all = np.array(y_all)
    report = {}
    for n, p in preds.items():
        p = np.array(p)
        pr, rc, f1, sup = precision_recall_fscore_support(y_all, p, zero_division=0)
        report[n] = dict(precision=pr.tolist(), recall=rc.tolist(), f1=f1.tolist(), support=sup.tolist(),
                         accuracy=float((p == y_all).mean()),
                         macro=[float(pr.mean()), float(rc.mean()), float(f1.mean())],
                         confusion=confusion_matrix(y_all, p).tolist())
    out = dict(classes=H.CLASSES, features=features, per_split=rows, gwo_curves=curves,
               gwo_feature_weights=weights, pooled_report=report,
               wilcoxon_acc=float(wilcoxon(df.acc_Hybrid, df.acc_MLP, alternative="greater").pvalue),
               wilcoxon_f1=float(wilcoxon(df.f1_Hybrid, df.f1_MLP, alternative="greater").pvalue))
    with open(H.RESULTS / "detailed_results.json", "w") as f:
        json.dump(out, f, indent=1, default=float)

    print("\nMean over splits (accuracy | macro F1):")
    for n in MODELS:
        print(f"  {n:12s} {df['acc_'+n].mean():.4f} ± {df['acc_'+n].std():.4f} | "
              f"{df['f1_'+n].mean():.4f} ± {df['f1_'+n].std():.4f}")
    print(f"Wilcoxon (one-sided) accuracy p = {out['wilcoxon_acc']:.4f}, macro F1 p = {out['wilcoxon_f1']:.4f}")
    print("Mean times (s):", df[["t_mlp", "t_hybrid", "t_committee", "t_gwo"]].mean().round(2).to_dict())
    print(f"Saved: {H.RESULTS / 'detailed_results.json'}")


if __name__ == "__main__":
    main()
