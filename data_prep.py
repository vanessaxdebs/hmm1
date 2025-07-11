#!/usr/bin/env python3
"""
Script to prepare data for the HMM pipeline.
It generates:
- positive_validation.fasta (Kunitz domains NOT in training seed)
- non_kunitz_proteins.fasta (sequences from Swiss-Prot that do not hit the Kunitz HMM)
- validation_labels.txt (combined labels for positive_validation.fasta and negative_validation.fasta)
"""

import gzip
from pathlib import Path
import sys
import subprocess
from Bio import SeqIO
import yaml

def find_config() -> Path:
    """Finds the config.yaml file relative to the script's location."""
    script_dir = Path(__file__).parent
    config_path = script_dir.parent / "config" / "config.yaml"
    if config_path.exists():
        return config_path.resolve()
    raise FileNotFoundError("config.yaml not found! Ensure it's in 'config/' relative to the project root.")

def load_config() -> dict:
    """Loads and returns the configuration from config.yaml."""
    try:
        config_file = find_config()
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Resolve paths relative to the project root
        project_root = Path(__file__).resolve().parent.parent
        for key in ['seed_alignment', 'pfam_kunitz_full_alignment',
                    'positive_validation_fasta', 'non_kunitz_validation_fasta',
                    'validation_labels_txt', 'swissprot_fasta']:
            if key in config and config[key] is not None:
                config[key] = project_root / Path(config[key])
        return config
    except Exception as e:
        print(f"ERROR: Could not load configuration: {e}", file=sys.stderr)
        sys.exit(1)

def run_hmmbuild(seed_alignment: Path, hmm_file: Path):
    """Trains a Profile HMM using HMMER's hmmbuild."""
    cmd = ["hmmbuild", str(hmm_file), str(seed_alignment)]
    print(f"Building temporary HMM for negative set generation: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"Temporary HMM built successfully: {hmm_file}")
    except FileNotFoundError:
        print("ERROR: hmmbuild command not found. Please ensure HMMER is installed and in your PATH.", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: hmmbuild failed during negative set generation. Stderr:\n{e.stderr}", file=sys.stderr)
        sys.exit(1)

def run_hmmsearch_temp(hmm_file: Path, fasta_file: Path, tblout_file: Path, e_value: float = 1e-5):
    """Runs hmmsearch for temporary purposes (e.g., to find non-hits)."""
    cmd = [
        "hmmsearch",
        "--tblout", str(tblout_file),
        "-E", str(e_value),
        "--noali",
        str(hmm_file), str(fasta_file)
    ]
    print(f"Running temporary hmmsearch on {fasta_file.name} to find Kunitz sequences...")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"Temporary hmmsearch results saved to {tblout_file}")
    except FileNotFoundError:
        print("ERROR: hmmsearch command not found. Please ensure HMMER is installed and in your PATH.", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: temporary hmmsearch failed on {fasta_file.name}. Stderr:\n{e.stderr}", file=sys.stderr)
        sys.exit(1)

def parse_tblout_for_hits(tbl_file: Path, e_value_cutoff: float) -> set[str]:
    """Parses hmmsearch --tblout and returns a set of unique hit IDs passing the E-value cutoff."""
    hits = set()
    if not tbl_file.exists() or tbl_file.stat().st_size == 0:
        print(f"WARNING: tblout file {tbl_file} is missing or empty. No hits parsed.", file=sys.stderr)
        return hits
    try:
        with open(tbl_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 13: # Standard HMMER tblout has at least 13 columns for hit line
                    try:
                        target_name = parts[0]
                        e_value = float(parts[4])
                        if e_value < e_value_cutoff:
                            hits.add(target_name)
                    except ValueError:
                        continue # Skip malformed lines
        return hits
    except Exception as e:
        print(f"ERROR: Error parsing temporary tblout file {tbl_file}: {e}", file=sys.stderr)
        sys.exit(1)


def prepare_data():
    config = load_config()

    seed_sto_path = config['seed_alignment']
    pfam_gz_path = config['pfam_kunitz_full_alignment']
    swissprot_fasta_path = config['swissprot_fasta']

    output_pos_fasta_path = config['positive_validation_fasta']
    output_neg_fasta_path = config['non_kunitz_validation_fasta'] # This is now the auto-generated one
    output_labels_txt_path = config['validation_labels_txt']
    
    negative_set_filter_e_value = config['negative_set_filter_e_value']
    max_negative_sequences = config['max_negative_sequences']

    # Create parent directories if they don't exist
    output_pos_fasta_path.parent.mkdir(parents=True, exist_ok=True)


    print(f"--- Preparing Validation Datasets ---")

    # --- Step 1: Prepare Positive Validation Set ---
    print(f"\n1. Preparing Positive Validation Set (Kunitz domains not in training data)...")
    print(f"Loading training IDs from: {seed_sto_path}")
    training_ids = set()
    try:
        for record in SeqIO.parse(seed_sto_path, "stockholm"):
            training_ids.add(record.id.split('/')[0]) # Handle UniProt IDs with /start-end
        print(f"Found {len(training_ids)} training sequence IDs.")
    except FileNotFoundError:
        print(f"ERROR: Training seed file '{seed_sto_path}' not found.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Could not parse training seed '{seed_sto_path}': {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Extracting Kunitz sequences from {pfam_gz_path} (excluding training sequences)...")
    positive_validation_sequences = {}
    try:
        with gzip.open(pfam_gz_path, "rt") as f_in:
            for record in SeqIO.parse(f_in, "stockholm"):
                record_id_base = record.id.split('/')[0] # Get base ID
                if record_id_base not in training_ids:
                    # Remove any alignment gaps ('-') or padding ('.')
                    positive_validation_sequences[record_id_base] = str(record.seq).replace('-', '').replace('.', '')
        print(f"Identified {len(positive_validation_sequences)} unique positive validation sequences.")
    except FileNotFoundError:
        print(f"ERROR: Pfam Kunitz full alignment '{pfam_gz_path}' not found.", file=sys.stderr)
        print("Please download it (e.g., PF00014.full.gz from Pfam FTP) and place it in 'data/validation_datasets/'.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Could not parse Pfam source file '{pfam_gz_path}': {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Writing positive validation FASTA to: {output_pos_fasta_path}")
    with open(output_pos_fasta_path, "w") as fasta_f:
        for seq_id, seq in positive_validation_sequences.items():
            fasta_f.write(f">{seq_id}\n{seq}\n")
    print(f"Successfully wrote {len(positive_validation_sequences)} sequences to {output_pos_fasta_path}.")


    # --- Step 2: Automatically Generate Negative Validation Set ---
    print(f"\n2. Automatically generating Negative Validation Set (non-Kunitz proteins from Swiss-Prot)...")
    if not swissprot_fasta_path.exists():
        print(f"ERROR: Swiss-Prot FASTA file '{swissprot_fasta_path}' not found.", file=sys.stderr)
        print("Please manually download 'uniprot_sprot.fasta' from UniProt FTP and place it in 'data/swissprot_database/'.", file=sys.stderr)
        sys.exit(1)

    # Create a temporary HMM for filtering
    temp_hmm_path = output_pos_fasta_path.parent / "temp_kunitz_filter.hmm"
    run_hmmbuild(seed_sto_path, temp_hmm_path) # Build HMM from training seed

    # Run hmmsearch against Swiss-Prot to identify Kunitz hits
    temp_tblout_path = output_pos_fasta_path.parent / "temp_swissprot_kunitz_hits.tbl"
    run_hmmsearch_temp(temp_hmm_path, swissprot_fasta_path, temp_tblout_path, e_value=negative_set_filter_e_value)
    
    kunitz_hits_in_swissprot = parse_tblout_for_hits(temp_tblout_path, negative_set_filter_e_value)
    print(f"Found {len(kunitz_hits_in_swissprot)} Kunitz-like hits in Swiss-Prot using filter E-value {negative_set_filter_e_value}.")

    # Collect non-Kunitz sequences from Swiss-Prot
    non_kunitz_sequences = {}
    print(f"Collecting non-Kunitz sequences from {swissprot_fasta_path} (excluding hits and too short/long sequences)...")
    
    seq_count = 0
    with open(swissprot_fasta_path, "r") as f_in:
        for record in SeqIO.parse(f_in, "fasta"):
            if record.id not in kunitz_hits_in_swissprot:
                # Basic filtering for sequence length (e.g., protein sequences between 20 and 20000 AA)
                # Kunitz domains are small, typically 30-60 residues. So we want sequences of various lengths
                # but avoid extremely short or extremely long sequences that might be artifacts or obscure.
                seq_len = len(record.seq)
                if 20 <= seq_len <= 20000:
                    non_kunitz_sequences[record.id] = str(record.seq)
                    seq_count += 1
                    if seq_count >= max_negative_sequences:
                        print(f"Reached maximum of {max_negative_sequences} negative sequences. Stopping collection.")
                        break
    
    print(f"Identified {len(non_kunitz_sequences)} non-Kunitz sequences for validation (max {max_negative_sequences}).")

    print(f"Writing negative validation FASTA to: {output_neg_fasta_path}")
    with open(output_neg_fasta_path, "w") as fasta_f:
        for seq_id, seq in non_kunitz_sequences.items():
            fasta_f.write(f">{seq_id}\n{seq}\n")
    print(f"Successfully wrote {len(non_kunitz_sequences)} sequences to {output_neg_fasta_path}.")

    # Clean up temporary HMMER files
    temp_hmm_path.unlink(missing_ok=True)
    temp_tblout_path.unlink(missing_ok=True)
    print("Cleaned up temporary HMMER files.")


    # --- Step 3: Write Combined Labels ---
    print(f"\n3. Writing combined validation labels to: {output_labels_txt_path}")
    with open(output_labels_txt_path, "w") as labels_f:
        # Write positive labels (1)
        for seq_id in positive_validation_sequences.keys():
            labels_f.write(f"{seq_id}\t1\n")
        # Write negative labels (0)
        for seq_id in non_kunitz_sequences.keys():
            labels_f.write(f"{seq_id}\t0\n")
    print(f"Successfully wrote {len(positive_validation_sequences) + len(non_kunitz_sequences)} total labels.")

    print("\n--- Data Preparation Complete ---")

if __name__ == "__main__":
    prepare_data()
