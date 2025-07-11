import gzip
import os
from Bio import SeqIO
from pathlib import Path

# --- Configuration ---
TRAINING_SEED_STO = "data/alignments/kunitz_seed_training.sto"
UNIPROT_KUNITZ_FASTA_GZ = "data/uniprotkb_PF00014_2025_07_11.fasta.gz" # Ensure this file is downloaded and correct
VALIDATION_DATASETS_DIR = "data/validation_datasets/"
os.makedirs(VALIDATION_DATASETS_DIR, exist_ok=True) # Ensure directory exists

OUTPUT_POSITIVE_FASTA = os.path.join(VALIDATION_DATASETS_DIR, "positive_validation.fasta")
OUTPUT_POSITIVE_LABELS = os.path.join(VALIDATION_DATASETS_DIR, "validation_labels.txt") # Labels for positives

# 1. Extract training accessions from the new kunitz_seed_training.sto
training_accessions = set()
if not os.path.exists(TRAINING_SEED_STO):
    print(f"ERROR: Training seed alignment '{TRAINING_SEED_STO}' not found. Please run build_seed_from_structures.py first.", file=sys.stderr)
    sys.exit(1)

print(f"Extracting training accessions from {TRAINING_SEED_STO}...")
with open(TRAINING_SEED_STO, "r") as sto:
    for record in SeqIO.parse(sto, "stockholm"):
        # Assuming Stockholm IDs are directly usable as accessions or can be parsed
        # For PDB IDs like "4PTI_A", you might want to extract just "4PTI"
        # For UniProt IDs (e.g., from UniProt-derived Stockholm), parse "sp|ACC|NAME"
        if "|" in record.id: # UniProt format (e.g., sp|P12345|NAME_ORG)
            parts = record.id.split("|")
            if len(parts) > 1:
                training_accessions.add(parts[1]) # Add UniProt Accession
        else: # PDB Chain ID format (e.g., 3TGI_I)
            training_accessions.add(record.id.split('_')[0]) # Add PDB ID

print(f"Found {len(training_accessions)} unique training accessions/PDB IDs.")


# 2. Read gzipped UniProt Kunitz domain FASTA (PF00014) and filter out training set
if not os.path.exists(UNIPROT_KUNITZ_FASTA_GZ):
    print(f"ERROR: UniProt Kunitz FASTA '{UNIPROT_KUNITZ_FASTA_GZ}' not found. Please download it.", file=sys.stderr)
    print("Example download URL (for latest PF00014 sequences): https://www.uniprot.org/uniprotkb?query=family:PF00014%20AND%20reviewed:true&format=fasta", file=sys.stderr)
    print("Remember to save it as `uniprotkb_PF00014_2025_07_11.fasta.gz` or similar, and manually gzip it.", file=sys.stderr)
    sys.exit(1)

filtered_sequences = []
print(f"Filtering validation sequences from {UNIPROT_KUNITZ_FASTA_GZ}...")
with gzip.open(UNIPROT_KUNITZ_FASTA_GZ, "rt") as in_f:
    for record in SeqIO.parse(in_f, "fasta"):
        # UniProt FASTA headers look like: >sp|O17644|O17644_CAEEL ...
        # We need to extract the UniProt accession (e.g., O17644)
        if "|" in record.id:
            acc = record.id.split("|")[1]
        else: # Handle cases where it might be just a UniProt ID
            acc = record.id.split(" ")[0] # Take the first part before space

        if acc not in training_accessions:
            filtered_sequences.append(record)

print(f"Filtered {len(filtered_sequences)} sequences for the positive validation set.")

# 3. Write filtered FASTA
with open(OUTPUT_POSITIVE_FASTA, "w") as out_f:
    SeqIO.write(filtered_sequences, out_f, "fasta")
print(f"Positive validation set written to {OUTPUT_POSITIVE_FASTA}")

# 4. Write labels (all 1s for positive class)
with open(OUTPUT_POSITIVE_LABELS, "w") as labels_f:
    for record in filtered_sequences:
        # Use the full record.id (e.g., sp|O17644|O17644_CAEEL) as the sequence ID for consistency with hmmsearch
        labels_f.write(f"{record.id}\t1\n")
print(f"Labels for positive validation set written to {OUTPUT_POSITIVE_LABELS}")

print("\n🎉 Positive validation dataset preparation completed.")
