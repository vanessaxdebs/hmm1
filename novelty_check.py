#!/usr/bin/env python3
"""
Novelty check for Kunitz HMM hits:
Flags hits not already annotated as Kunitz (by keyword, Pfam, or domain description) in UniProt.
Works with HMMER .tblout with UniProt IDs (e.g., sp|P84875|PCPI_SABMA).
"""

import os
from pathlib import Path
import requests
import time
import sys

# --- Configuration ---
# Assuming results are saved in a structured 'results/run_YYYYMMDD_HHMMSS/' directory
RESULTS_BASE_DIR = Path("results")
NOVELTY_OUTPUT_FILE = RESULTS_BASE_DIR / "novel_kunitz_candidates.txt" # This will be created in the base results dir

# UniProt REST API endpoint for fetching entries
UNIPROT_ENTRY_API = "https://rest.uniprot.org/uniprotkb/"

# --- Functions ---

def get_latest_results_tbl(base_dir: Path) -> Path:
    """Finds the most recent hmmsearch_swissprot.tbl in results/run_* subdirs."""
    subdirs = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
    if not subdirs:
        raise FileNotFoundError(f"No 'run_*' subdirectories found in {base_dir}. Run the HMM pipeline first.")
    
    # Sort by creation time (or modification time) to get the latest
    latest_subdir = max(subdirs, key=os.path.getmtime)
    tbl_file = latest_subdir / "hmmsearch_swissprot_scan.tbl" # Use the explicit name from hmm.py
    
    if not tbl_file.exists():
        raise FileNotFoundError(f"'{tbl_file}' does not exist in the latest run directory '{latest_subdir}'.")
    return tbl_file

def extract_uniprot_accession(hit_id: str) -> str:
    """
    For a hit like sp|P84875|PCPI_SABMA, returns P84875.
    Handles tr| for TrEMBL entries as well.
    """
    if hit_id.startswith("sp|") or hit_id.startswith("tr|"):
        parts = hit_id.split("|")
        if len(parts) >= 2:
            return parts[1] # UniProt Accession
    # If not canonical UniProt ID, take the part before potential fragment info
    return hit_id.split("/")[0]

def load_hmm_hits(hmm_results_file: Path) -> Set[str]:
    """
    Loads all unique accessions from the results .tbl file.
    Uses the correct E-value cutoff from the pipeline.
    """
    hits = set()
    evalue_cutoff = None # We need to get this from the run's config or log

    # Try to infer E-value cutoff from the run directory's config or logs
    run_dir = hmm_results_file.parent
    config_path = run_dir / "config.yaml" # If config is copied to run_dir
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                evalue_cutoff = float(config.get('e_value_cutoff', 1e-5))
                print(f"Using E-value cutoff from run config: {evalue_cutoff}")
        except Exception as e:
            print(f"WARNING: Could not load E-value from {config_path}: {e}. Using default 1e-5.", file=sys.stderr)
            evalue_cutoff = 1e-5
    else:
        print("WARNING: Could not find run-specific config.yaml. Using default E-value cutoff 1e-5.", file=sys.stderr)
        evalue_cutoff = 1e-5


    with open(hmm_results_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 5: # Ensure enough columns for E-value (main table)
                try:
                    e_value = float(parts[4]) # E-value in hmmsearch --tblout is typically column 5 (index 4)
                    if e_value < evalue_cutoff:
                        raw_id = parts[0]
                        accession = extract_uniprot_accession(raw_id)
                        hits.add(accession)
                except ValueError:
                    continue # Skip malformed lines where E-value isn't a number
    return hits

def is_kunitz_annotated(uniprot_acc: str, retries: int = 3, delay: float = 0.5) -> bool:
    """
    Queries UniProt API to check for Kunitz annotation (keyword, Pfam, or domain feature).
    """
    url = f"{UNIPROT_ENTRY_API}{uniprot_acc}.json"
    
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=15) # Increased timeout
            r.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            data = r.json()

            # Check features for 'Kunitz' (case-insensitive)
            for feat in data.get("features", []):
                if "kunitz" in feat.get("description", "").lower():
                    return True
            
            # Check keywords for 'Kunitz' (case-insensitive)
            for keyword in data.get("keywords", []):
                if "kunitz" in keyword.get("value", "").lower():
                    return True
            
            # Check Pfam domains for PF00014
            for dbref in data.get("dbReferences", []):
                if dbref.get("type") == "Pfam" and dbref.get("id") == "PF00014":
                    return True
            
            return False # No Kunitz annotation found
        
        except requests.exceptions.RequestException as e:
            print(f"Warning: Attempt {attempt + 1} failed to fetch UniProt entry for {uniprot_acc}: {e}", file=sys.stderr)
            if r.status_code == 404: # Not Found
                return False # If entry doesn't exist, it's not annotated as Kunitz
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1)) # Exponential back-off
            else:
                print(f"Error: All retries failed for {uniprot_acc}. Skipping.", file=sys.stderr)
                return False # Failed after all retries
        except Exception as e:
            print(f"Error parsing UniProt entry for {uniprot_acc}: {e}", file=sys.stderr)
            return False # General parsing error

def main():
    print("--- Running Novelty Check for Kunitz HMM Hits ---")
    try:
        tbl_file = get_latest_results_tbl(RESULTS_BASE_DIR)
        print(f"Using HMMER results file: {tbl_file}")
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        print("Please run the main HMM pipeline (`hmm.py`) first to generate the results.", file=sys.stderr)
        sys.exit(1)

    accessions = load_hmm_hits(tbl_file)
    print(f"Total unique UniProt accessions found by HMM: {len(accessions)}")

    if not accessions:
        print("No HMM hits found to check for novelty. Exiting.", file=sys.stderr)
        sys.exit(0)

    novel_candidates = []
    checked_count = 0
    total_accessions = len(accessions)

    for accession in sorted(accessions):
        checked_count += 1
        print(f"[{checked_count}/{total_accessions}] Checking {accession}...", end=" ")
        
        # Add a small delay between requests to be polite to UniProt servers
        time.sleep(0.1) 
        
        if not is_kunitz_annotated(accession):
            print("NOVEL ✅")
            novel_candidates.append(accession)
        else:
            print("known 🔬")
        
        # Another small delay for the next request
        time.sleep(0.1) 

    print(f"\nFound {len(novel_candidates)} novel candidate hits (not currently annotated as Kunitz in UniProt).")
    
    # Ensure the output directory exists
    NOVELTY_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(NOVELTY_OUTPUT_FILE, "w") as out:
        if novel_candidates:
            for n in novel_candidates:
                out.write(n + "\n")
            print(f"List of novel Kunitz candidates saved to {NOVELTY_OUTPUT_FILE}")
        else:
            out.write("# No novel Kunitz candidates found based on current criteria.\n")
            print(f"No novel candidates found. An empty placeholder file created at {NOVELTY_OUTPUT_FILE}")

    print("\nDone. Manual review of novel candidates is recommended.")

if __name__ == "__main__":
    main()
