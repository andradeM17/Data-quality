import re
import json
import os

def extract_tmx_to_jsonl_efficient(input_file, output_file):
    """
    Most efficient version: streaming line-by-line processing with progress counter
    Best for very large files to avoid memory issues
    """
    pattern = re.compile(r'<tuv xml:lang="ga"><seg>(.*?)</seg></tuv>')
    
    # Get file size for progress calculation
    file_size = os.path.getsize(input_file)
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        lines_processed = 0
        matches_found = 0
        bytes_processed = 0
        
        for line in infile:
            lines_processed += 1
            bytes_processed += len(line.encode('utf-8'))
            
            match = pattern.search(line)
            if match:
                matches_found += 1
                json_obj = {"text": match.group(1)}
                outfile.write(json.dumps(json_obj, ensure_ascii=False) + '\n')
            
            # Show progress every 1000 lines
            if lines_processed % 1000 == 0:
                progress = (bytes_processed / file_size) * 100
                print(f"\rProcessed: {lines_processed:,} lines | Found: {matches_found:,} matches | Progress: {progress:.1f}%", end='', flush=True)
        
        # Final summary
        print(f"\n✓ Complete! Processed {lines_processed:,} lines, found {matches_found:,} Irish text segments")

def extract_tmx_to_jsonl_batch(input_file, output_file, batch_size=1000):
    """
    Batch processing version with progress counter
    """
    pattern = re.compile(r'<tuv xml:lang="ga"><seg>(.*?)</seg></tuv>')
    file_size = os.path.getsize(input_file)
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        batch = []
        lines_processed = 0
        matches_found = 0
        bytes_processed = 0
        
        for line in infile:
            lines_processed += 1
            bytes_processed += len(line.encode('utf-8'))
            
            match = pattern.search(line)
            if match:
                matches_found += 1
                batch.append(json.dumps({"text": match.group(1)}, ensure_ascii=False))
                
                if len(batch) >= batch_size:
                    outfile.write('\n'.join(batch) + '\n')
                    batch.clear()
            
            # Show progress every 1000 lines
            if lines_processed % 1000 == 0:
                progress = (bytes_processed / file_size) * 100
                print(f"\rProcessed: {lines_processed:,} lines | Found: {matches_found:,} matches | Progress: {progress:.1f}%", end='', flush=True)
        
        # Write remaining items
        if batch:
            outfile.write('\n'.join(batch) + '\n')
        
        print(f"\n✓ Complete! Processed {lines_processed:,} lines, found {matches_found:,} Irish text segments")

def extract_tmx_to_jsonl_optimized_regex(input_file, output_file):
    """
    For files that fit in memory: optimized regex with progress tracking
    """
    file_size = os.path.getsize(input_file)
    print(f"Loading file ({file_size / (1024*1024):.1f} MB)...")
    
    with open(input_file, 'r', encoding='utf-8') as infile:
        content = infile.read()
    
    print("Extracting Irish text segments...")
    
    # Use compiled regex for better performance
    pattern = re.compile(r'<tuv xml:lang="ga"><seg>(.*?)</seg></tuv>')
    matches = pattern.findall(content)
    
    print(f"Found {len(matches):,} Irish text segments. Writing to JSONL...")
    
    # Write all at once using generator expression
    with open(output_file, 'w', encoding='utf-8') as outfile:
        outfile.write('\n'.join(
            json.dumps({"text": match}, ensure_ascii=False) 
            for match in matches
        ) + '\n')
    
    print(f"✓ Complete! Saved {len(matches):,} entries to {output_file}")

# Use the most appropriate version for your needs:

# For very large files (recommended):

datasets = ["c", "eub", "euconst", "h", "n", "o", "p", "q", "t", "x"]

for dataset in datasets:
    extract_tmx_to_jsonl_efficient(f'full-datasets/{dataset}-en-ga.tmx', f'JSON-files/{dataset}-en-ga.jsonl')

# For moderate files with batch processing:
# extract_tmx_to_jsonl_batch('t-en-ga.tmx', 't-en-ga.jsonl', batch_size=1000)

# For smaller files that fit comfortably in memory:
# extract_tmx_to_jsonl_optimized_regex('t-en-ga.tmx', 't-en-ga.jsonl')