from collections import Counter, defaultdict
from datasketch import MinHash, MinHashLSH

def line_similarity(a, b):
    """Return True if lines differ by at most one word"""
    words_a = a.split()
    words_b = b.split()
    if abs(len(words_a) - len(words_b)) > 1:
        return False
    mismatches = sum(w1 != w2 for w1, w2 in zip(words_a, words_b))
    mismatches += abs(len(words_a) - len(words_b))
    return mismatches <= 1

def find_near_duplicates_minhash(lines, num_perm=128, threshold=0.8):
    """
    Fast near-duplicate detection using MinHash + LSH.
    threshold: approximate Jaccard similarity to consider as near-duplicate
    """
    # Step 1: Create MinHash objects
    print("Creating MinHash signatures...")
    minhashes = []
    for line in lines:
        words = set(line.split())
        m = MinHash(num_perm=num_perm)
        for word in words:
            m.update(word.encode('utf8'))
        minhashes.append(m)

    # Step 2: Build LSH index
    print("Building LSH index...")
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    for i, m in enumerate(minhashes):
        lsh.insert(f"line_{i}", m)

    # Step 3: Query for near-duplicates
    print("Querying for near-duplicates...")
    duplicates = set()
    for i, m in enumerate(minhashes):
        print(i/len(minhashes) * 100, "%", end="\r")
        result = lsh.query(m)
        for j_str in result:
            j = int(j_str.split("_")[1])
            if i != j:
                # Only mark the later occurrence as duplicate
                duplicates.add(lines[j])

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
    """
    Find repeated triplets, keeping only the first occurrence of each triplet.
    Returns a set of triplets that occur more than once (like find_duplicates returns duplicates).
    """
    triplets = [tuple(lines[i:i+3]) for i in range(len(lines) - 2)]
    seen = set()
    duplicates = set()

    for triplet in triplets:
        if triplet in seen:
            duplicates.add(triplet)
        else:
            seen.add(triplet)

    return duplicates

def main():
    datasets = ["EUbookshop", "Wikipedia", "XLEnt"]

    for d in datasets:
        filename = f"Deduplication-study/{d}.txt"

        with open(filename, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
            lines = lines[:50000]  # keep the limit

        print("Finding exact duplicates...")
        duplicates = find_duplicates(lines)
        print(f"{d}: {len(duplicates)} exact duplicates")

        print("Finding duplicate triplets...")
        triplet_lines = find_duplicate_triplets(lines)
        print(f"{d}: {len(triplet_lines)} lines in duplicate triplets")

        print("Finding near-duplicates...")
        near_duplicates = find_near_duplicates_minhash(lines)
        print(f"{d}: {len(near_duplicates)} near-duplicates found")

        d_cleaned_lines = []
        seen = set()
        for line in lines:
            if line not in seen:
                d_cleaned_lines.append(line)  # keep first occurrence
                seen.add(line)

        cleaned_triplet_lines = []
        seen_triplets = set()
        i = 0
        while i < len(lines):
            if i <= len(lines) - 3:
                triplet = tuple(lines[i:i+3])
                if triplet in triplet_lines:
                    if triplet not in seen_triplets:
                        cleaned_triplet_lines.extend(lines[i:i+3]) # keep first occurrence
                        seen_triplets.add(triplet)
                    i += 1
                    continue
            if lines[i] not in lines[i-2:i]:
                cleaned_triplet_lines.append(lines[i])
            i += 1

        near_duplicates_cleaned_lines = []
        seen_near = set()
        for line in lines:
            if line not in near_duplicates:
                if line not in seen_near:
                    near_duplicates_cleaned_lines.append(line)  # keep first occurrence
                    seen_near.add(line)

        with open(f"Deduplication-study/Complete duplicates/{d}.txt", "w", encoding="utf-8") as f:
            for line in d_cleaned_lines:
                f.write(line + "\n")

        with open(f"Deduplication-study/Triplet duplicates/{d}.txt", "w", encoding="utf-8") as f:
            for line in cleaned_triplet_lines:
                f.write(line + "\n")

        with open(f"Deduplication-study/Near duplicates/{d}.txt", "w", encoding="utf-8") as f:
            for line in near_duplicates_cleaned_lines:
                f.write(line + "\n")
if __name__ == "__main__":
    main()