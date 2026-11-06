"""
Builds the 1000-sample stratified subset used in the project from the original
UCI file (data/Faults_full_UCI.NNA).

The subset is already included as data/steel_faults_subset.csv; run this only
if you want to regenerate it.

Run from the repository root:
    python src/make_subset.py
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ['X_Minimum', 'X_Maximum', 'Y_Minimum', 'Y_Maximum', 'Pixels_Areas',
            'X_Perimeter', 'Y_Perimeter', 'Sum_of_Luminosity', 'Minimum_of_Luminosity',
            'Maximum_of_Luminosity', 'Length_of_Conveyer', 'TypeOfSteel_A300',
            'TypeOfSteel_A400', 'Steel_Plate_Thickness', 'Edges_Index', 'Empty_Index',
            'Square_Index', 'Outside_X_Index', 'Edges_X_Index', 'Edges_Y_Index',
            'Outside_Global_Index', 'LogOfAreas', 'Log_X_Index', 'Log_Y_Index',
            'Orientation_Index', 'Luminosity_Index', 'SigmoidOfAreas']
CLASSES = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']
N_SUBSET = 1000


def main():
    raw = pd.read_csv(ROOT / "data" / "Faults_full_UCI.NNA", sep=r"\s+", header=None)
    raw.columns = FEATURES + CLASSES
    raw["Fault"] = raw[CLASSES].values.argmax(1)
    raw["Fault"] = raw["Fault"].map(dict(enumerate(CLASSES)))
    print(f"Full dataset: {raw.shape[0]} samples, {len(FEATURES)} features, {len(CLASSES)} classes")

    subset, _ = train_test_split(raw[FEATURES + ["Fault"]], train_size=N_SUBSET,
                                 stratify=raw["Fault"], random_state=42)
    out = ROOT / "data" / "steel_faults_subset.csv"
    subset.reset_index(drop=True).to_csv(out, index=False)
    print(f"Subset written to {out}")
    print(subset["Fault"].value_counts().to_string())


if __name__ == "__main__":
    main()
