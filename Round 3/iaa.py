import csv
import krippendorff
from sklearn.metrics import cohen_kappa_score
import itertools

csv_file = f"Round 3/data-for-iaa.csv"
rows = []

with open(csv_file, newline='') as f:
    reader = csv.reader(f, delimiter='\t')
    rows = list(reader)

    # ---------- Pairwise Cohen's Kappa + Krippendorff ----------
    print("  Pairwise agreement metrics:")

    all_data = list(map(list, zip(*rows)))
    num_annotators = len(all_data)

    for (a, b) in itertools.combinations(range(num_annotators), 2):
        annotator_a = all_data[a]
        annotator_b = all_data[b]

        agreements = sum(x == y for x, y in zip(annotator_a, annotator_b))
        total = len(annotator_a)
        observed_agreement = agreements / total
        print(f"\n  Annotator {a+1} vs {b+1}")
        print("    Observed agreement:", round(observed_agreement, 2))

        # Cohen
        kappa = cohen_kappa_score(annotator_a, annotator_b)
        print("    Cohen's kappa:", round(kappa, 3))

        # Krippendorff (pairwise)
        try:
            alpha_pair = krippendorff.alpha(
                reliability_data=[annotator_a, annotator_b],
                level_of_measurement='nominal'
            )
        except ValueError:
            alpha_pair = 1

        print("    Krippendorff's alpha:", round(alpha_pair, 3))


for i in range(0, 14):
    print(f"\nCalculating IAA for group {i+1} (row {i*6+1}-{6*(i+1)})...")
    current_rows = rows[6*i:6*(i+1)]

    # Transpose: annotators × items
    data = list(map(list, zip(*current_rows)))

    # ---------- Krippendorff ----------
    try:
        alpha = krippendorff.alpha(
            reliability_data=data,
            level_of_measurement='nominal'
        )
    except ValueError:
        alpha = 1

    print(f"  Krippendorff's alpha: {round(alpha, 3)}")