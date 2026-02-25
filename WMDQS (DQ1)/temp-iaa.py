import csv
import krippendorff
from sklearn.metrics import cohen_kappa_score
import itertools

csv_file = f"WMDQS (DQ1)/round1-alldata.csv"
rows = []

with open(csv_file, newline='') as f:
    reader = csv.reader(f, delimiter='\t')
    rows = list(reader)

    # Transpose: annotators × items
    data = list(map(list, zip(*rows)))

    # ---------- Krippendorff ----------
    try:
        alpha = krippendorff.alpha(
            reliability_data=data,
            level_of_measurement='nominal'
        )
    except ValueError:
        alpha = 1

    print(f"  Krippendorff's alpha: {round(alpha, 3)}")