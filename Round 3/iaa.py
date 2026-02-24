import csv
import krippendorff

csv_file = f"Round 3/data-for-iaa.csv"
rows = []

with open(csv_file, newline='') as f:
    reader = csv.reader(f, delimiter='\t')
    rows = list(reader)
    for i in range(0, 14):
        print(f"Calculating IAA for group {i+1} (row {i*6+1}-{6*(i+1)})...")
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

        print(f"  Krippendorff's alpha for group {i+1} (row {i*6+1}-{6*(i+1)}): {round(alpha, 3)}")