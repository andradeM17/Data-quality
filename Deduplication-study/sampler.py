import os
import csv
import random

# Folders and datasets
cleaning = ["Complete duplicates", "Near duplicates", "Triplet duplicates"]
datasets = ["EUbookshop", "Wikipedia", "XLEnt"]

num_samples = 50
lines_per_sample = 5

# Output CSV
csv_filename = "Deduplication-study/combined_samples.csv"

with open(csv_filename, "w", newline="", encoding="utf-8-sig") as csvfile:
    writer = csv.writer(csvfile)
    # Header
    writer.writerow(["CleaningType", "Dataset", "SampleID", "SampleText"])
    
    for clean in cleaning:
        folder_path = "Deduplication-study/" + clean  # assuming folder name matches cleaning type
        for dataset in datasets:
            file_path = os.path.join(folder_path, f"{dataset}.txt")
            
            # Read all lines from the file
            with open(file_path, "r", encoding="utf-8") as f:
                all_lines = [line.strip() for line in f if line.strip()]
            
            # Generate 100 random samples of 10 lines each
            for sample_id in range(1, num_samples + 1):
                sample_lines = random.sample(all_lines, lines_per_sample)
                # Join all lines into a single string (use '\n' or another separator)
                sample_text = "\n".join(sample_lines)
                writer.writerow([clean, dataset, sample_id, sample_text])

print(f"CSV file '{csv_filename}' created with 450 rows")
