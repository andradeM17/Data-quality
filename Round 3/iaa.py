import csv
import krippendorff
from sklearn.metrics import cohen_kappa_score
import itertools

csv_file = f"Round 3/data-for-iaa.csv"
rows = []

with open(csv_file, newline='') as f:
    reader = csv.reader(f, delimiter='\t')
    rows = list(reader)

    # ---------- Cohen's Kappa (pairwise between annotators) ----------
    print("  Pairwise Cohen's kappa:")

    all_data = list(map(list, zip(*rows)))
    num_annotators = len(all_data)

    for (a, b) in itertools.combinations(range(num_annotators), 2):
        annotator_a = all_data[a]
        annotator_b = all_data[b]

        agreements = sum(x == y for x, y in zip(annotator_a, annotator_b))
        total = len(annotator_a)
        observed_agreement = agreements / total
        print("(Observed agreement:", round(observed_agreement, 2), ")")

        kappa = cohen_kappa_score(annotator_a, annotator_b)
        print(f"    Annotator {a+1} vs {b+1}: {round(kappa, 3)}")


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