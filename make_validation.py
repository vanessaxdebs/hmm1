import gzip
import os
import sys
from Bio import SeqIO
from pathlib import Path
import requests
import time # For polite API requests

# --- Configuration ---
# Path to the Stockholm file used for training. This is used to EXCLUDE sequences from validation.
TRAINING_SEED_STO = Path("data/alignments/kunitz_seed_training.sto")

# Output directories for validation data
VALIDATION_DATASETS_DIR = Path("data/validation_datasets")
os.makedirs(VALIDATION_DATASETS_DIR, exist_ok=True)

# Output files for the positive validation set (Kunitz domains NOT in training)
OUTPUT_POSITIVE_FASTA = VALIDATION_DATASETS_DIR / "positive_validation.fasta"
OUTPUT_POSITIVE_LABELS = VALIDATION_DATASETS_DIR / "validation_labels.txt" # Labels for positives (value '1')

# Output files for the negative validation set (non-Kunitz proteins)
OUTPUT_NEGATIVE_FASTA = VALIDATION_DATASETS_DIR / "negative_validation.fasta"
OUTPUT_NEGATIVE_LABELS = VALIDATION_DATASETS_DIR / "negative_labels.txt" # Labels for negatives (value '0')

# UniProt query parameters for fetching data
PFAM_KUNITZ_ID = "PF00014"
UNIPROT_REST_API = "https://rest.uniprot.org/uniprotkb/search"

# --- Functions ---

def get_training_accessions(sto_path: Path) -> Set[str]:
    """
    Extracts UniProt accessions or PDB IDs used in the training Stockholm alignment.
    This set will be EXCLUDED from the positive validation set.
    """
    training_ids = set()
    if not sto_path.exists():
        print(f"ERROR: Training seed alignment '{sto_path}' not found.", file=sys.stderr)
        sys.exit(1)

    print(f"Extracting training accessions from {sto_path}...")
    try:
        # Using Bio.AlignIO.parse for robustness in parsing Stockholm
        with open(sto_path, "r") as f:
            for record in SeqIO.parse(f, "stockholm"):
                # UniProt IDs are typically like 'sp|P12345|NAME_HUMAN' or 'tr|Q9ABC0|NAME_BACSU'
                # PDB IDs might be like '1BPI_A'
                if "|" in record.id:
                    # Extract accession for UniProt entries
                    parts = record.id.split("|")
                    if len(parts) >= 2:
                        training_ids.add(parts[1])
                elif "_" in record.id:
                    # For PDB IDs like 3TGI_I, add the PDB ID itself
                    training_ids.add(record.id.split('_')[0].upper())
                else:
                    training_ids.add(record.id) # Fallback for other IDs
        print(f"Found {len(training_ids)} unique training IDs to exclude.")
    except Exception as e:
        print(f"ERROR: Failed to parse training Stockholm file {sto_path}: {e}", file=sys.stderr)
        sys.exit(1)
    return training_ids

def fetch_uniprot_sequences(query: str, format: str = "fasta", retries: int = 3, delay: float = 1.0) -> str:
    """Fetches sequences from UniProt REST API with retry mechanism."""
    params = {"query": query, "format": format, "size": 10000} # Max size for non-streaming
    for attempt in range(retries):
        try:
            response = requests.get(UNIPROT_REST_API, params=params)
            response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
            return response.text
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1} failed to fetch UniProt data: {e}", file=sys.stderr)
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1)) # Exponential back-off
            else:
                raise
    return "" # Should not be reached if retries exhausted

def create_filtered_fasta_and_labels(
    input_fasta_content: str,
    output_fasta_path: Path,
    output_labels_path: Path,
    label_value: str,
    exclude_ids: Set[str] = None
):
    """
    Parses FASTA content, filters sequences based on exclude_ids,
    and writes to FASTA and label files.
    """
    if exclude_ids is None:
        exclude_ids = set()

    filtered_records = []
    
    # Use io.StringIO to treat the string content as a file
    from io import StringIO
    handle = StringIO(input_fasta_content)

    for record in SeqIO.parse(handle, "fasta"):
        # Extract UniProt accession from the FASTA header
        # e.g., >sp|O17644|O17644_CAEEL Kunitz-type protease inhibitor [Fragment] OS=Caenorhabditis elegans OX=6239...
        uniprot_acc = record.id.split('|')[1] if '|' in record.id else record.id.split(' ')[0]

        if uniprot_acc not in exclude_ids:
            filtered_records.append(record)

    if not filtered_records:
        print(f"WARNING: No sequences found after filtering for {output_fasta_path.name}. Check query or exclusions.", file=sys.stderr)
        
    SeqIO.write(filtered_records, output_fasta_path, "fasta")
    fasta_to_label_txt(output_fasta_path, output_labels_path, label=label_value)
    print(f"Created {output_fasta_path} ({len(filtered_records)} sequences) and {output_labels_path}")


# --- Main Script Logic ---
if __name__ == "__main__":
    print("\n--- Preparing Validation Datasets ---")

    # Step 1: Get training IDs to exclude from positive validation set
    training_accessions_to_exclude = get_training_accessions(TRAINING_SEED_STO)

    # Step 2: Fetch and prepare Positive Validation Set (True Kunitz, NOT in training)
    print("\nFetching positive validation set (Kunitz domain, Swiss-Prot, excluding training PDBs/UniProt IDs)...")
    
    # Query for reviewed Kunitz domains (PFAM:PF00014) in UniProt
    positive_query = f"family:{PFAM_KUNITZ_ID} AND reviewed:true"
    
    try:
        positive_fasta_content = fetch_uniprot_sequences(positive_query)
        create_filtered_fasta_and_labels(
            positive_fasta_content,
            OUTPUT_POSITIVE_FASTA,
            OUTPUT_POSITIVE_LABELS,
            label_value="1",
            exclude_ids=training_accessions_to_exclude
        )
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Failed to fetch positive validation data from UniProt: {e}", file=sys.stderr)
        print("Please check your internet connection or UniProt API status.", file=sys.stderr)
        sys.exit(1)

    # Step 3: Fetch and prepare Negative Validation Set (NOT Kunitz)
    print("\nFetching negative validation set (NOT Kunitz domain, Swiss-Prot, diverse)...")
    
    # Query for reviewed proteins NOT in PF00014, within a typical domain length range
    # and explicitly excluding those with known 3D structures (as structural data might imply domains)
    negative_query = f"NOT family:{PFAM_KUNITZ_ID} AND reviewed:true AND len:[50 TO 200] AND NOT database:pdb AND taxonomy_id:9606" # Human (9606) for consistency, not strictly necessary but helps control size/diversity.
    # Consider adjusting length range and taxonomy for broader representation if needed.

    try:
        negative_fasta_content = fetch_uniprot_sequences(negative_query)
        create_filtered_fasta_and_labels(
            negative_fasta_content,
            OUTPUT_NEGATIVE_FASTA,
            OUTPUT_NEGATIVE_LABELS,
            label_value="0" # Label for negative class
        )
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Failed to fetch negative validation data from UniProt: {e}", file=sys.stderr)
        print("Please check your internet connection or UniProt API status.", file=sys.stderr)
        sys.exit(1)

    print("\n🎉 All validation datasets prepared successfully.")
