from collections import Counter

def find_duplicates(lines):
    line_counts = Counter(lines)
    duplicates = {line: count for line, count in line_counts.items() if count > 1}
    return duplicates

def find_duplicate_triplets(lines):
    triplets = [
        tuple(lines[i:i+3]) for i in range(len(lines) - 2)
    ]
    counts = Counter(triplets)
    repeated = {triplet: count for triplet, count in counts.items() if count > 1}
    return repeated


def main():
    datasets = ["EUbookshop", "Wikipedia", "XLEnt"]

    for d in datasets:
        filename = f"Deduplication-study/{d}.txt"

        with open(filename, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]  # remove empty lines & whitespace

            # Count each line’s frequency
            duplicates= find_duplicates(lines)
            print(f"Total completely duplicate lines in {d}: {len(duplicates)}.")

            three_in_a_row_duplicates = find_duplicate_triplets(lines)
            print(f"Found {len(three_in_a_row_duplicates)} sets of duplicate triplets of lines in {d}.\n")

if __name__ == "__main__":
    main()