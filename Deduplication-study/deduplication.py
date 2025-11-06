from collections import Counter

datasets = ["EUbookshop", "Wikipedia", "XLEnt"]

for d in datasets:
    filename = f"Deduplication-study/{d}.txt"

    with open(filename, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]  # remove empty lines & whitespace

    # Count each line’s frequency
    line_counts = Counter(lines)

    # Find duplicates
    duplicates = {line: count for line, count in line_counts.items() if count > 1}

    print(f"Total completely duplicate lines in {d}: {len(duplicates)}\n")