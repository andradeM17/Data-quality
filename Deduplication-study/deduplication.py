from collections import Counter, defaultdict

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
    """
    Fast near-duplicate detection using signatures.
    Only compare lines that have the same number of words or off by one.
    """
    seen_lines = set()
    duplicates = set()

    # Bucket lines by length to reduce unnecessary comparisons
    length_buckets = defaultdict(list)
    for line in lines:
        length_buckets[len(line.split())].append(line)

    # Compare lines only within the same or adjacent length buckets
    print(f"Checking {len(length_buckets)} groups for near-duplicates...")
    for length in length_buckets:
        print(f"\tBucket length {length} with {len(length_buckets[length])} lines")
        candidates = length_buckets[length] + length_buckets.get(length - 1, []) + length_buckets.get(length + 1, [])
        for j, line in enumerate(candidates):
            print(f"\t{round(((j/len(candidates))*100)**0.5)}%", end="\r")
            if line in seen_lines:
                continue
            for other in candidates[j+1:]:
                if line_similarity(line, other):
                    duplicates.add(other)  # keep first occurrence, remove later
            seen_lines.add(line)

    return duplicates

def find_duplicates(lines):
    """Find exact duplicates, keeping the first copy"""
    seen = set()
    duplicates = set()
    for line in lines:
        if line in seen:
            duplicates.add(line)
        else:
            seen.add(line)
    return duplicates

def find_duplicate_triplets(lines):
    """Find repeated triplets, excluding first occurrence"""
    triplets = [tuple(lines[i:i+3]) for i in range(len(lines) - 2)]
    counts = Counter(triplets)
    repeated_triplets = {triplet for triplet, count in counts.items() if count > 1}

    lines_to_remove = set()
    first_seen = set()
    for triplet in repeated_triplets:
        if triplet not in first_seen:
            first_seen.add(triplet)
            for i in range(1, len(lines) - 2):
                if tuple(lines[i:i+3]) == triplet:
                    lines_to_remove.update(lines[i:i+3])
    return lines_to_remove

def main():
    datasets = ["EUbookshop", "Wikipedia", "XLEnt"]

    for d in datasets:
        filename = f"Deduplication-study/{d}.txt"
        output_filename = f"Deduplication-study/{d}_cleaned.txt"

        with open(filename, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
            lines = lines[:10000]

        print("Finding exact duplicates...")
        duplicates = find_duplicates(lines)
        print(f"{d}: {len(duplicates)} exact duplicates")

        print("Finding duplicate triplets...")
        triplet_lines = find_duplicate_triplets(lines)
        print(f"{d}: {len(triplet_lines)} lines in duplicate triplets")

        print("Finding near duplicates...")
        near_duplicates = find_near_duplicates(lines)
        print(f"{d}: {len(near_duplicates)} near duplicates\n")

        # Combine all lines to remove
        all_to_remove = duplicates.union(triplet_lines).union(near_duplicates)

        # Keep first occurrence of each line
        cleaned_lines = []
        seen = set()
        for line in lines:
            if line not in all_to_remove or line not in seen:
                cleaned_lines.append(line)
                seen.add(line)

        # Write cleaned file
        with open(output_filename, "w", encoding="utf-8") as f:
            for line in cleaned_lines:
                f.write(line + "\n")

        print(f"Cleaned file written to {output_filename} with {len(cleaned_lines)} lines remaining.\n")

if __name__ == "__main__":
    main()
