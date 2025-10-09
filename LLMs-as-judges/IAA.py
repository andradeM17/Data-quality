import pandas as pd
from sklearn.metrics import cohen_kappa_score

csv_file = 'LLMs-as-judges/results.csv'
df = pd.read_csv(csv_file)

annotator_cols = df.columns[19:26]  # adjust as needed

kappa_matrix = pd.DataFrame(index=annotator_cols, columns=annotator_cols, dtype=float)
for i, ann1 in enumerate(annotator_cols):
    for j, ann2 in enumerate(annotator_cols[i:], start=i):
        if ann1 == ann2:
            kappa_matrix.loc[ann1, ann2] = 1.0
        else:
            kappa = cohen_kappa_score(df[ann1], df[ann2])
            kappa_matrix.loc[ann1, ann2] = kappa
            kappa_matrix.loc[ann2, ann1] = kappa  # mirror

        print(f"Cohen's Kappa between {ann1} and {ann2}: {kappa_matrix.loc[ann1, ann2]:.2f}")

kappa_matrix = kappa_matrix.round(2)
print("\nCohen's Kappa Matrix (rounded):")
print(kappa_matrix)
kappa_matrix.to_csv("LLMs-as-judges/QED-Cohen's_kappa.csv", index=True)