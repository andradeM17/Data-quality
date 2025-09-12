import numpy as np
import pandas as pd
import krippendorff
from statsmodels.stats.inter_rater import fleiss_kappa
from collections import Counter
from sklearn.metrics import cohen_kappa_score

files = ["H", "N", "P", "C", "X", "O", "Q", "EUB", "EUc", "T"]
measures = ["C", "H", "NS"]

for f in files:
    for m in measures:
        print(f + m)
        data = pd.read_csv(f"FG/{f}{m}.csv", delimiter='\t')
        data = data.fillna('BLANK')  # replace NaN with a default value
        data = data.values.tolist()  # list of lists (rows = items, cols = annotators)

        # Transpose for Krippendorff (rows = annotators, cols = items)
        transposed_data = list(map(list, zip(*data)))
        try:
            alpha = krippendorff.alpha(
                reliability_data=transposed_data,
                level_of_measurement='nominal'
            )
            print("\tKrippendorff’s Alpha:", round(alpha, 4))
        except ValueError as e:
            print("\tKrippendorff’s Alpha: Error -", e)


        # Cohen’s Kappa (only works with 2 annotators!)
        if len(data[0]) == 2:
            annotator1 = [row[0] for row in data]
            annotator2 = [row[1] for row in data]
            kappa = cohen_kappa_score(annotator1, annotator2)
            print("\tCohen’s Kappa:", round(kappa, 4))

        # Fleiss’ Kappa (works for any # of annotators)
        # Step 1: find all unique labels
        labels = sorted(set(val for row in data for val in row))
        label_index = {label: i for i, label in enumerate(labels)}

        # Step 2: build the counts matrix
        counts = np.zeros((len(data), len(labels)), dtype=int)
        for i, row in enumerate(data):
            counter = Counter(row)
            for label, count in counter.items():
                counts[i, label_index[label]] = count

        # Step 3: compute Fleiss’ Kappa
        fleiss = fleiss_kappa(counts)
        print("\tFleiss’ Kappa:", round(fleiss, 4))

        print("\n")
    print("--------------------------------------------------")