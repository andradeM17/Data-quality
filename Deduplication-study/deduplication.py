from collections import Counter

def line_similarity(a, b):
    """Return True if lines differ by at most one word"""
    words_a = a.split()
    words_b = b.split()
    if abs(len(words_a) - len(words_b)) > 1:
        return False
    mismatches = sum(w1 != w2 for w1, w2 in zip(words_a, words_b))
    mismatches += abs(len(words_a) - len(words_b))
    return mismatches <= 1

def find_near_duplicates(lines):
    duplicates = set()
    n = len(lines)
    for i, line in enumerate(lines):
        print(f"{round(i/n*100)}% ({i} of {n} completed)", end="\r")
        for other in lines[i+1:]:
            if line_similarity(line, other):
                duplicates.add(line)
                duplicates.add(other)
    return duplicates

def find_duplicates(lines):
    line_counts = Counter(lines)
    duplicates = {line for line, count in line_counts.items() if count > 1}
    return duplicates

def find_duplicate_triplets(lines):
    triplets = [tuple(lines[i:i+3]) for i in range(len(lines) - 2)]
    counts = Counter(triplets)
    repeated_triplets = {triplet for triplet, count in counts.items() if count > 1}
    # Flatten triplets into individual lines to remove
    lines_to_remove = set()
    for triplet in repeated_triplets:
        lines_to_remove.update(triplet)
    return lines_to_remove

def main():
    datasets = ["EUbookshop", "Wikipedia", "XLEnt"]

    for d in datasets:
        filename = f"Deduplication-study/{d}.txt"
        output_filename = f"Deduplication-study/{d}_cleaned.txt"

        with open(filename, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
            lines = lines[:10000]

        # Find duplicates
        duplicates = find_duplicates(lines)
        print(f"Total completely duplicate lines in {d}: {len(duplicates)}.")

        # Find duplicate triplets
        triplet_lines = find_duplicate_triplets(lines)
        print(f"Found {len(triplet_lines)} lines in duplicate triplets in {d}.")

        # Find near duplicates
        near_duplicates = find_near_duplicates(lines)
        print(f"Total near duplicate lines in {d}: {len(near_duplicates)}.\n")

        # Combine all lines to remove
        all_to_remove = duplicates.union(triplet_lines).union(near_duplicates)

        # Write cleaned file
        cleaned_lines = [line for line in lines if line not in all_to_remove]
        with open(output_filename, "w", encoding="utf-8") as f:
            for line in cleaned_lines:
                f.write(line + "\n")

        print(f"Cleaned file written to {output_filename} with {len(cleaned_lines)} lines remaining.\n")

if __name__ == "__main__":
    main()