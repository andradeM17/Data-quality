import sys
import pandas as pd
import numpy as np
from itertools import combinations
from math import comb
import matplotlib.pyplot as plt
import seaborn as sns


# -------------------------
# Helper functions
# -------------------------

def percent_agreement(df):
    n_items, n_annotators = df.shape
    total_pairs = comb(n_annotators, 2) * n_items
    agree_pairs = 0
    per_item = []
    for i in df.index:
        row = df.loc[i].values
        pairs = 0
        agrees = 0
        for a, b in combinations(range(len(row)), 2):
            pairs += 1
            if row[a] == row[b]:
                agrees += 1
        per_item.append(agrees / pairs)
        agree_pairs += agrees
    overall = agree_pairs / total_pairs
    return overall, per_item


def fleiss_kappa(table):
    table = np.array(table, dtype=float)
    n_items, n_categories = table.shape
    n = int(table.sum(axis=1)[0])
    p_j = table.sum(axis=0) / (n_items * n)
    P_i = ((table * (table - 1)).sum(axis=1)) / (n * (n - 1))
    P_bar = P_i.mean()
    P_e = (p_j ** 2).sum()
    return (P_bar - P_e) / (1 - P_e)


def krippendorff_alpha(data, values=None, missing=None):
    items = [list(item) for item in data]
    if values is None:
        values = sorted({v for row in items for v in row if v is not missing})
    value_to_index = {v: i for i, v in enumerate(values)}
    K = len(values)
    coincidence = np.zeros((K, K), dtype=float)
    for row in items:
        row_vals = [v for v in row if v is not missing]
        n = len(row_vals)
        if n <= 1:
            continue
        counts = np.zeros(K)
        for v in row_vals:
            counts[value_to_index[v]] += 1
        for k in range(K):
            for l in range(K):
                if k == l:
                    coincidence[k, l] += counts[k] * (counts[k] - 1)
                else:
                    coincidence[k, l] += counts[k] * counts[l]
    total_pairs = coincidence.sum()
    if total_pairs == 0:
        return float("nan")
    Do = sum(coincidence[k, l] for k in range(K) for l in range(K) if k != l) / total_pairs
    marg = coincidence.sum(axis=1)
    p = marg / total_pairs
    De = 1 - (p ** 2).sum()
    return 1 - Do / De


def cohens_kappa(a, b):
    a, b = np.array(a), np.array(b)
    categories = sorted(set(a) | set(b))
    mapping = {cat: i for i, cat in enumerate(categories)}
    n = len(a)
    conf = np.zeros((len(categories), len(categories)))
    for i in range(n):
        conf[mapping[a[i]], mapping[b[i]]] += 1
    po = np.trace(conf) / n
    row_marg = conf.sum(axis=1) / n
    col_marg = conf.sum(axis=0) / n
    pe = (row_marg * col_marg).sum()
    return (po - pe) / (1 - pe)


# -------------------------
# Main execution
# -------------------------

def main():
    df = pd.read_csv("Deduplication-study/annotations.csv")
    df = df.set_index(df.columns[0])

    print("\n=== Annotation Matrix ===")
    print(df, "\n")

    categories = sorted(pd.unique(df.values.ravel()))
    print(f"Detected categories: {categories}")

    # Percent agreement
    overall_pa, per_item_pa = percent_agreement(df)
    print(f"\nOverall Percent Agreement: {overall_pa:.4f}")
    for i, v in zip(df.index, per_item_pa):
        print(f" Item {i}: {v:.4f}")

    # Fleiss’ kappa
    count_table = [[list(df.loc[i]).count(cat) for cat in categories] for i in df.index]
    fleiss = fleiss_kappa(count_table)
    print(f"\nFleiss' kappa: {fleiss:.4f}")

    # Krippendorff’s alpha
    alpha = krippendorff_alpha(df.values, values=categories)
    print(f"Krippendorff’s alpha (nominal): {alpha:.4f}")

    # Pairwise Cohen’s kappa (matrix + heatmap)
    annotators = df.columns
    pairwise_kappas = pd.DataFrame(index=annotators, columns=annotators, dtype=float)

    for a, b in combinations(annotators, 2):
        k = cohens_kappa(df[a], df[b])
        pairwise_kappas.loc[a, b] = k
        pairwise_kappas.loc[b, a] = k
    np.fill_diagonal(pairwise_kappas.values, 1.0)

    print("\n=== Pairwise Cohen’s Kappa ===")
    print(pairwise_kappas.round(3))
    print(f"\nAverage pairwise Cohen’s kappa: {np.nanmean(pairwise_kappas.values):.4f}")

    # Plot heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(pairwise_kappas.astype(float), annot=True, cmap="coolwarm", vmin=0, vmax=1)
    plt.title("Inter-Annotator Agreement (Cohen’s Kappa)")
    plt.tight_layout()
    plt.savefig("Deduplication-study/annotator_kappa_heatmap.png")
    print("\nSaved heatmap as 'annotator_kappa_heatmap.png'")
    plt.close()  # close figure to prevent GUI issues in VS Code



if __name__ == "__main__":
    main()