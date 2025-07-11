import os

# --- Updated List of Required Files for the Project ---
# This dictionary now reflects the structured data organization.
REQUIRED_FILES = {
    "data/alignments/kunitz_seed_training.sto": "# STOCKHOLM", # Your new training seed alignment
    "data/validation_datasets/positive_validation.fasta": ">", # Renamed and moved
    "data/validation_datasets/negative_validation.fasta": ">", # New for negative validation set
    "data/validation_datasets/validation_labels.txt": "", # Combined labels for positive
    "data/validation_datasets/negative_labels.txt": "", # Labels for negative
    "data/swissprot_database/uniprot_sprot.fasta": ">" # For final SwissProt scan
}

def check_file(path, must_startwith):
    """
    Checks if a file exists, is not empty, and starts with a specified string.
    """
    if not os.path.isfile(path):
        print(f"ERROR: {path} is missing.", file=sys.stderr)
        return False
    if os.path.getsize(path) == 0:
        print(f"ERROR: {path} is empty.", file=sys.stderr)
        return False
    if must_startwith:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f: # Added encoding for robustness
            first_line = f.readline()
            if not first_line.startswith(must_startwith):
                print(f"ERROR: {path} does not start with '{must_startwith}'. Found: '{first_line.strip()}'", file=sys.stderr)
                return False
    print(f"OK: {path}")
    return True

if __name__ == "__main__":
    import sys # Import sys for stderr
    ok = True
    print("--- Checking Required Data Files ---")
    for file, startswith in REQUIRED_FILES.items():
        # Ensure parent directories exist before checking the file
        parent_dir = os.path.dirname(file)
        if parent_dir and not os.path.exists(parent_dir):
            print(f"WARNING: Directory '{parent_dir}' does not exist. Skipping file check for {file}.", file=sys.stderr)
            ok = False # Mark as not fully OK if directories are missing
            continue # Skip file check if directory is missing
            
        ok = check_file(file, startswith) and ok
    
    if not ok:
        print("\nSome data files are missing or invalid. Please follow the project setup instructions.", file=sys.stderr)
        sys.exit(1)
    else:
        print("\nAll essential data files are present and look valid. Proceeding with pipeline.")
