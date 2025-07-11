#!/usr/bin/env python3
"""
Build a Stockholm-format seed alignment for the Kunitz domain
based on structural alignment of known 3D PDB chains.
This alignment will be used EXCLUSIVELY for HMM training.
"""

import os
import subprocess
from Bio import AlignIO, SeqIO
from Bio.PDB import PDBList
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

# --- Configuration for Training Data ---
# These PDB IDs and their respective chains will be used EXCLUSIVELY for building
# the HMM training alignment. They MUST NOT overlap with any PDB IDs or sequences
# that will be used for model validation.
TRAINING_PDB_IDS = ["3TGI", "1BPI", "5PTI", "5ZJ3", "1Y62", "3OFW", "5YV7", "1DTX", "5M4V", "3M7Q"]
TRAINING_PDB_CHAINS = {
    "3TGI": "I", "1BPI": "A", "5PTI": "A", "5ZJ3": "A", "1Y62": "A",
    "3OFW": "A", "5YV7": "A", "1DTX": "A", "5M4V": "A", "3M7Q": "B"
}

# --- Directory Setup (aligned with project structure) ---
RAW_PDB_DIR = "data/raw_pdb_files" # Stores full PDB files
CHAIN_DIR = os.path.join(RAW_PDB_DIR, "chains_for_training") # Stores extracted chains
ALIGNMENTS_DIR = "data/alignments" # Stores the final Stockholm alignment for HMMER

os.makedirs(RAW_PDB_DIR, exist_ok=True)
os.makedirs(CHAIN_DIR, exist_ok=True)
os.makedirs(ALIGNMENTS_DIR, exist_ok=True)

pdbl = PDBList() # Initialize PDBList for downloading

# Step 1: Download PDB files and Extract Specific Chains for Training
print("\n--- Step 1: Downloading PDBs and Extracting Chains for Training Data ---")
for pdb_id in TRAINING_PDB_IDS:
    chain_id = TRAINING_PDB_CHAINS[pdb_id]
    print(f"Downloading PDB structure '{pdb_id}'...")
    
    # Download the full PDB file to raw_pdb_files directory
    # pdbl.retrieve_pdb_file returns the path to the downloaded file
    # This might return the path in the pdb_dir, typically something like pdb/xx/pdbXXXX.ent
    # We construct the expected path based on how pdbl.retrieve_pdb_file works.
    pdbl_download_path = pdbl.retrieve_pdb_file(pdb_id, pdir=RAW_PDB_DIR, file_format="pdb")
    
    # Define the output path for the selected chain file
    chain_file_path = os.path.join(CHAIN_DIR, f"{pdb_id}_{chain_id}.pdb")
    
    # Use pdb_selchain to extract the specific chain
    print(f"Extracting chain {chain_id} from {pdb_id}...")
    try:
        # Check if pdb_selchain is available
        subprocess.run(["which", "pdb_selchain"], check=True, capture_output=True)
        with open(chain_file_path, "w") as out:
            subprocess.run(["pdb_selchain", f"-{chain_id}", pdbl_download_path], stdout=out, check=True)
        print(f"Chain {chain_id} saved to {chain_file_path}")
    except FileNotFoundError:
        print("ERROR: pdb_selchain command not found. Please install EMBOSS or ensure it's in your PATH.", file=sys.stderr)
        print("Skipping chain extraction for now. Manual extraction needed or use alternative.", file=sys.stderr)
        # As a fallback or for development, you might skip this for now or use Bio.PDB to parse and write.
    except subprocess.CalledProcessError as e:
        print(f"ERROR: pdb_selchain failed for {pdb_id} chain {chain_id}: {e}", file=sys.stderr)
        print(f"Stderr: {e.stderr.decode()}", file=sys.stderr)
        # Decide if you want to exit or continue with other PDBs
        sys.exit(1)


# Step 2: Align structures using MUSTANG
# Define the common output prefix for MUSTANG's files (e.g., kunitz_alignment.afasta)
output_prefix = os.path.join(ALIGNMENTS_DIR, "kunitz_training_alignment") # Output goes to ALIGNMENTS_DIR

# Define the path where the FASTA alignment output is expected (MUSTANG default is .afasta for FASTA)
aligned_fasta_path = os.path.join(ALIGNMENTS_DIR, "kunitz_training_alignment.afasta")

# Get list of all extracted chain PDB files to input to MUSTANG
input_chain_files = [os.path.join(CHAIN_DIR, f"{pdb_id}_{TRAINING_PDB_CHAINS[pdb_id]}.pdb") for pdb_id in TRAINING_PDB_IDS]
# Filter out any files that might not have been created if pdb_selchain failed for some
input_chain_files = [f for f in input_chain_files if os.path.exists(f)]

if not input_chain_files:
    print("ERROR: No PDB chain files found for MUSTANG alignment. Please check previous steps.", file=sys.stderr)
    sys.exit(1)

# Construct the command for MUSTANG
cmd = [
    "mustang",
    "-i", # Input structures flag
    *input_chain_files, # List all the generated chain PDB files as inputs
    "-o", output_prefix, # Output identifier/prefix flag
    "-F", "fasta"        # Crucial: This tells MUSTANG to output in FASTA format
]

# Run the MUSTANG command
print("\n--- Step 2: Running MUSTANG for Structural Alignment ---")
print(f"Running MUSTANG: {' '.join(cmd)}")
try:
    subprocess.run(cmd, check=True)
    print("MUSTANG alignment completed successfully.")
except FileNotFoundError:
    print("ERROR: mustang command not found. Please install MUSTANG or ensure it's in your PATH.", file=sys.stderr)
    sys.exit(1)
except subprocess.CalledProcessError as e:
    print(f"ERROR: MUSTANG failed: {e}", file=sys.stderr)
    print(f"Stderr: {e.stderr.decode()}", file=sys.stderr)
    sys.exit(1)

# Step 3: Convert aligned FASTA to Stockholm format
input_fasta_for_sto = aligned_fasta_path # Use the path to the FASTA file MUSTANG just created
output_sto_path = os.path.join(ALIGNMENTS_DIR, "kunitz_seed_training.sto") # Explicitly name it for training

print(f"\n--- Step 3: Converting FASTA alignment to Stockholm format ---")
print(f"Converting FASTA alignment from '{input_fasta_for_sto}' to Stockholm format...")

if not os.path.exists(input_fasta_for_sto):
    print(f"ERROR: Expected FASTA alignment file '{input_fasta_for_sto}' not found. Cannot convert to Stockholm.", file=sys.stderr)
    sys.exit(1)

try:
    # Read the FASTA alignment using Biopython's AlignIO
    align = AlignIO.read(input_fasta_for_sto, "fasta")

    # Write the alignment to Stockholm format
    with open(output_sto_path, "w") as out:
        AlignIO.write(align, out, "stockholm")

    print(f"Structural Stockholm alignment for HMM training created at {output_sto_path}")
except Exception as e:
    print(f"ERROR: Failed to convert FASTA to Stockholm: {e}", file=sys.stderr)
    sys.exit(1)

print("\n🎉 Training seed alignment pipeline completed successfully!")
