import csv
import json

# -------- CONFIG --------

datasets = ["HPLT", "NLLB", "Paracrawl", "CCMatrix", "XLEnt", "OpenSubtitles", "QED", "EUBookshops", "EUconst", "Tatoeba"]

for d in datasets:
    input_csv = f"Dingo/csv/{d}.csv"      # Path to your CSV file
    column_name = "ga"         # Column you want to convert
    output_jsonl = f"Dingo/JSONL/{d}.jsonl"  # Output JSONL file
    # ------------------------

    with open(input_csv, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        with open(output_jsonl, "w", encoding="utf-8") as jsonlfile:
            for row in reader:
                # Take the value of the desired column
                value = row[column_name]
                # Write as JSON object with key 'text'
                json_obj = {"text": value}
                jsonlfile.write(json.dumps(json_obj, ensure_ascii=False) + "\n")

    print(f"Saved {output_jsonl} with values from column '{column_name}'")
