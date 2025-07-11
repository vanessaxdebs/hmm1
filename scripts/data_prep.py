#!/usr/bin/env python3
"""
Prepares validation datasets (positive and negative) and the full SwissProt database
for the Kunitz HMM pipeline.
This script ensures the validation sets are independent of the training data.
"""

import os
import sys
import subprocess
import gzip
from pathlib import Path
import yaml
from Bio import AlignIO, SeqIO
from Bio.PDB import PDBList # For downloading PDBs for structurally similar negatives
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

# --- Helper Functions (from your existing scripts) ---

def load_config(config_file: Path = None) -> dict:
    """Loads configuration from a YAML file."""
    if config_file is None:
        # Assume config.yaml is in a 'config' directory one level up from 'scripts'
        script_dir = Path(__file__).resolve().parent
        project_root_guess = script_dir.parent
        config_file = project_root_guess / "config" / "config.yaml"

    print(f"DEBUG: script_dir (parent of data_prep.py): {script_dir}")
    print(f"DEBUG: project_root_guess: {project_root_guess}")
    print(f"DEBUG: config_path that script is looking for: {config_file}")
    print(f"DEBUG: config_path.exists() returned {config_file.exists()}: {config_file}")

    if not config_file.exists():
        print(f"ERROR: Config file not found at {config_file}", file=sys.stderr)
        sys.exit(1)

    required_fields = {
        'data_dir': str,
        'results_dir': str,
        'seed_alignment': str,
        'positive_validation_fasta': str,
        'non_kunitz_validation_fasta': str,
        'validation_labels_txt': str,
        'swissprot_fasta': str,
        'e_value_cutoff': float,
        'negative_set_strategy': str, # New required field
        'clustering_identity_threshold': float, # New required field
        'clustering_length_difference_cutoff': float, # New required field
    }

    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        # Validate required fields
        for field, expected_type in required_fields.items():
            if field not in config or not isinstance(config[field], expected_type):
                print(f"ERROR: Missing or invalid field '{field}' in config.yaml. Expected type: {expected_type.__name__}", file=sys.stderr)
                sys.exit(1)

        # Convert path strings to Path objects for easier handling
        config['data_dir'] = Path(config['data_dir'])
        config['results_dir'] = Path(config['results_dir'])
        config['seed_alignment'] = Path(config['seed_alignment'])
        config['positive_validation_fasta'] = Path(config['positive_validation_fasta'])
        config['non_kunitz_validation_fasta'] = Path(config['non_kunitz_validation_fasta'])
        config['validation_labels_txt'] = Path(config['validation_labels_txt'])
        config['swissprot_fasta'] = Path(config['swissprot_fasta'])

        # Validate strategy-specific fields
        if config['negative_set_strategy'] == 'random_swissprot':
            if 'num_negative_samples' not in config or not isinstance(config['num_negative_samples'], int):
                print("ERROR: 'num_negative_samples' missing or invalid for 'random_swissprot' strategy.", file=sys.stderr)
                sys.exit(1)
        elif config['negative_set_strategy'] == 'structurally_similar':
            if 'structurally_similar_negative_pdb_ids' not in config or not isinstance(config['structurally_similar_negative_pdb_ids'], list):
                print("ERROR: 'structurally_similar_negative_pdb_ids' missing or invalid for 'structurally_similar' strategy.", file=sys.stderr)
                sys.exit(1)
            if 'structurally_similar_negative_pdb_chains' not in config or not isinstance(config['structurally_similar_negative_pdb_chains'], dict):
                print("ERROR: 'structurally_similar_negative_pdb_chains' missing or invalid for 'structurally_similar' strategy.", file=sys.stderr)
                sys.exit(1)
        else:
            print(f"ERROR: Unknown negative_set_strategy: {config['negative_set_strategy']}", file=sys.stderr)
            sys.exit(1)

        return config

    except yaml.YAMLError as e:
        print(f"ERROR: Error parsing config.yaml: {e}", file=sys.stderr)
        sys.exit(1)

def get_full_path(relative_path: Path) -> Path:
    """Converts a relative Path object to an absolute Path relative to project root."""
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent # Assuming scripts/data_prep.py is one level below project root
    return project_root / relative_path

def check_stockholm(file_path: Path) -> bool:
    """Checks if a Stockholm file exists and is not empty."""
    if not file_path.exists():
        print(f"ERROR: Stockholm file '{file_path}' not found.", file=sys.stderr)
        return False
    if file_path.stat().st_size == 0:
        print(f"ERROR: Stockholm file '{file_path}' is empty.", file=sys.stderr)
        return False
    return True

def check_fasta(file_path: Path) -> bool:
    """Checks if a FASTA file exists and is not empty."""
    if not file_path.exists():
        print(f"ERROR: FASTA file '{file_path}' not found.", file=sys.stderr)
        return False
    if file_path.stat().st_size == 0:
        print(f"ERROR: FASTA file '{file_path}' is empty.", file=sys.stderr)
        return False
    return True

def check_label_txt(file_path: Path) -> bool:
    """Checks if a label file exists and is not empty."""
    if not file_path.exists():
        print(f"ERROR: Label file '{file_path}' not found.", file=sys.stderr)
        return False
    if file_path.stat().st_size == 0:
        print(f"ERROR: Label file '{file_path}' is empty.", file=sys.stderr)
        return False
    return True

def parse_stockholm_ids(stockholm_file: Path) -> set:
    """Parses a Stockholm file and returns a set of sequence IDs."""
    ids = set()
    try:
        # AlignIO can handle gzipped files directly if given the 'gz' mode
        if str(stockholm_file).endswith('.gz'):
            with gzip.open(stockholm_file, 'rt') as f:
                for record in AlignIO.read(f, "stockholm"):
                    ids.add(record.id)
        else:
            for record in AlignIO.read(stockholm_file, "stockholm"):
                ids.add(record.id)
    except Exception as e:
        print(f"ERROR: Could not parse Stockholm file '{stockholm_file}': {e}", file=sys.stderr)
        sys.exit(1)
    return ids

def parse_fasta_ids(fasta_file: Path) -> set:
    """Parses a FASTA file and returns a set of sequence IDs."""
    ids = set()
    try:
        for record in SeqIO.parse(fasta_file, "fasta"):
            ids.add(record.id)
    except Exception as e:
        print(f"ERROR: Could not parse FASTA file '{fasta_file}': {e}", file=sys.stderr)
        sys.exit(1)
    return ids

def run_cdhit(input_fasta: Path, output_fasta: Path, identity_threshold: float, length_diff_cutoff: float) -> int:
    """
    Runs CD-HIT to perform sequence clustering (redundancy reduction).
    Args:
        input_fasta (Path): Path to the input FASTA file.
        output_fasta (Path): Path to the output clustered FASTA file.
        identity_threshold (float): Sequence identity threshold (e.g., 0.9 for 90%).
        length_diff_cutoff (float): Length difference cutoff (e.g., 0.9 for 90% length overlap).
    Returns:
        int: Number of sequences in the clustered output.
    """
    print(f"Running CD-HIT on {input_fasta.name} at {identity_threshold*100}% identity...")
    
    # CD-HIT output files
    clstr_file = output_fasta.with_suffix('.clstr')

    # Construct the CD-HIT command
    # -i input_fasta: input FASTA file
    # -o output_fasta: output non-redundant FASTA file
    # -c identity_threshold: sequence identity threshold (e.g., 0.9 for 90%)
    # -aS length_diff_cutoff: alignment coverage for the longer sequence (e.g., 0.9 for 90%)
    # -g 1: greedy clustering (default)
    # -T 0: use all available threads (0 means auto)
    # -M 0: use all available memory (0 means auto)
    cmd = [
        "cd-hit",
        "-i", str(input_fasta),
        "-o", str(output_fasta),
        "-c", str(identity_threshold),
        "-aS", str(length_diff_cutoff),
        "-g", "1",
        "-T", "0",
        "-M", "0"
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        # Check if output file was actually created and is not empty
        if not output_fasta.exists() or output_fasta.stat().st_size == 0:
            print(f"WARNING: CD-HIT ran, but output file '{output_fasta}' is missing or empty. Check CD-HIT logs.", file=sys.stderr)
            return 0
        
        # Count sequences in the output FASTA
        clustered_count = sum(1 for _ in SeqIO.parse(output_fasta, "fasta"))
        print(f"CD-HIT clustering completed. Retained {clustered_count} non-redundant sequences.")
        return clustered_count
    except FileNotFoundError:
        print("ERROR: cd-hit command not found. Please install CD-HIT or ensure it's in your PATH.", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: CD-HIT failed: {e.stderr}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during CD-HIT execution: {e}", file=sys.stderr)
        sys.exit(1)


def extract_pfam_kunitz_sequences(pfam_full_alignment_gz: Path, training_ids: set, output_fasta: Path, config: dict) -> int:
    """
    Extracts Kunitz sequences from the Pfam full alignment,
    excluding those present in the training_ids set, and writes to a FASTA file.
    Applies CD-HIT for redundancy reduction.
    """
    print(f"Extracting Kunitz sequences from {pfam_full_alignment_gz} (excluding training sequences)...")
    temp_fasta_path = output_fasta.parent / f"temp_{output_fasta.name}" # Temp file before clustering
    extracted_count = 0
    unique_extracted_ids = set()

    try:
        with gzip.open(pfam_full_alignment_gz, "rt") as f_in, open(temp_fasta_path, "w") as f_out:
            for record in AlignIO.read(f_in, "stockholm"):
                # Pfam IDs are typically like 'UNIPROTACCESSION/START-END' or 'PDBID_CHAIN/START-END'
                # We need to ensure consistency with training_ids format.
                # For now, assume record.id matches training_ids format directly.
                if record.id not in training_ids:
                    # Create a new SeqRecord without alignment gaps for FASTA output
                    clean_seq = str(record.seq).replace('-', '') # Remove gaps
                    clean_record = SeqRecord(Seq(clean_seq), id=record.id, description=record.description)
                    if clean_record.id not in unique_extracted_ids:
                        SeqIO.write(clean_record, f_out, "fasta")
                        unique_extracted_ids.add(clean_record.id)
                        extracted_count += 1
    except FileNotFoundError:
        print(f"ERROR: Pfam Kunitz full alignment '{pfam_full_alignment_gz}' not found.", file=sys.stderr)
        print(f"Please download it (e.g., PF00014.full.gz from Pfam FTP) and place it in '{pfam_full_alignment_gz.parent}'.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to extract Kunitz sequences from Pfam alignment: {e}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Identified {extracted_count} raw positive validation sequences before clustering.")

    # Apply CD-HIT for redundancy reduction
    if extracted_count > 0:
        clustered_count = run_cdhit(temp_fasta_path, output_fasta, 
                                     config['clustering_identity_threshold'], 
                                     config['clustering_length_difference_cutoff'])
        os.remove(temp_fasta_path) # Clean up temporary file
        print(f"Successfully wrote {clustered_count} clustered positive validation sequences to {output_fasta}.")
        return clustered_count
    else:
        print(f"No sequences to cluster for positive validation set.")
        if temp_fasta_path.exists(): os.remove(temp_fasta_path)
        return 0


def sample_non_kunitz_from_swissprot(swissprot_fasta: Path, num_samples: int, output_fasta: Path) -> int:
    """
    Samples a specified number of non-Kunitz protein sequences from the Swiss-Prot database.
    (This function does NOT explicitly filter out Kunitz domains, assuming they are rare
    in a random sample and will be handled by the HMM's specificity. For rigorous
    negative set, pre-filtering for known Kunitz is recommended but complex for this script.)
    """
    print(f"Automatically generating Negative Validation Set (non-Kunitz proteins from Swiss-Prot)...")
    if not swissprot_fasta.exists():
        print(f"ERROR: Swiss-Prot FASTA file '{swissprot_fasta}' not found.", file=sys.stderr)
        print(f"Please manually download 'uniprot_sprot.fasta' from UniProt FTP and place it in '{swissprot_fasta.parent}'.", file=sys.stderr)
        sys.exit(1)

    all_sequences = []
    try:
        for record in SeqIO.parse(swissprot_fasta, "fasta"):
            all_sequences.append(record)
    except Exception as e:
        print(f"ERROR: Could not parse Swiss-Prot FASTA file '{swissprot_fasta}': {e}", file=sys.stderr)
        sys.exit(1)

    if len(all_sequences) < num_samples:
        print(f"WARNING: Swiss-Prot contains only {len(all_sequences)} sequences, less than requested {num_samples}.", file=sys.stderr)
        num_samples = len(all_sequences)

    # Use random.sample for efficient sampling without replacement
    import random
    random.seed(42) # For reproducibility of sampling
    sampled_sequences = random.sample(all_sequences, num_samples)

    try:
        with open(output_fasta, "w") as f_out:
            SeqIO.write(sampled_sequences, f_out, "fasta")
    except Exception as e:
        print(f"ERROR: Failed to write negative validation FASTA: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Successfully wrote {len(sampled_sequences)} negative validation sequences to {output_fasta}.")
    return len(sampled_sequences)


def prepare_structurally_similar_negative_set(
    pdb_ids: list, pdb_chains: dict, output_fasta: Path, raw_pdb_dir: Path, chain_dir: Path, config: dict
) -> int:
    """
    Downloads PDBs for a list of structurally similar (but non-Kunitz) proteins,
    extracts specified chains, and writes their sequences to a FASTA file.
    Applies CD-HIT for redundancy reduction.
    """
    print(f"Preparing Structurally Similar Negative Validation Set (from {len(pdb_ids)} PDBs)...")
    pdbl = PDBList()
    extracted_sequences = []

    os.makedirs(raw_pdb_dir, exist_ok=True)
    os.makedirs(chain_dir, exist_ok=True)

    temp_fasta_path = output_fasta.parent / f"temp_{output_fasta.name}" # Temp file before clustering

    for pdb_id in pdb_ids:
        chain_id = pdb_chains.get(pdb_id)
        if not chain_id:
            print(f"WARNING: No chain ID specified for PDB {pdb_id}. Skipping.", file=sys.stderr)
            continue

        print(f"  Processing PDB '{pdb_id}' chain '{chain_id}'...")
        try:
            pdbl_download_path = pdbl.retrieve_pdb_file(pdb_id, pdir=raw_pdb_dir, file_format="pdb")
            chain_file_path = chain_dir / f"{pdb_id}_{chain_id}.pdb"

            # Use pdb_selchain to extract the specific chain
            subprocess.run(["pdb_selchain", f"-{chain_id}", pdbl_download_path], stdout=open(chain_file_path, "w"), check=True)

            # Read the extracted chain PDB and get its sequence
            for record in SeqIO.parse(chain_file_path, "pdb-atom"):
                # Modify ID to be consistent with how we'd label it for validation
                record.id = f"{pdb_id}_{chain_id}"
                record.description = "" # Clear description to keep IDs clean
                extracted_sequences.append(record)
            print(f"  Extracted sequence for {pdb_id}_{chain_id}.")

        except FileNotFoundError:
            print(f"ERROR: pdb_selchain command not found or PDB {pdb_id} not downloaded. Skipping.", file=sys.stderr)
            print("  Please ensure EMBOSS (pdb_selchain) is installed and in your PATH.", file=sys.stderr)
            continue
        except subprocess.CalledProcessError as e:
            print(f"ERROR: pdb_selchain failed for {pdb_id} chain {chain_id}: {e}", file=sys.stderr)
            print(f"  Stderr: {e.stderr.decode()}", file=sys.stderr)
            continue
        except Exception as e:
            print(f"ERROR: Failed to process {pdb_id} chain {chain_id}: {e}", file=sys.stderr)
            continue

    if not extracted_sequences:
        print("ERROR: No sequences extracted for structurally similar negative set. Check PDB IDs/chains and pdb_selchain installation.", file=sys.stderr)
        sys.exit(1)

    # Write raw extracted sequences to a temporary FASTA
    try:
        with open(temp_fasta_path, "w") as f_out:
            SeqIO.write(extracted_sequences, f_out, "fasta")
    except Exception as e:
        print(f"ERROR: Failed to write temporary structurally similar negative validation FASTA: {e}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Identified {len(extracted_sequences)} raw structurally similar negative sequences before clustering.")

    # Apply CD-HIT for redundancy reduction
    clustered_count = run_cdhit(temp_fasta_path, output_fasta, 
                                 config['clustering_identity_threshold'], 
                                 config['clustering_length_difference_cutoff'])
    os.remove(temp_fasta_path) # Clean up temporary file

    print(f"Successfully wrote {clustered_count} clustered structurally similar negative validation sequences to {output_fasta}.")
    return clustered_count


def create_validation_labels(positive_fasta: Path, negative_fasta: Path, output_labels: Path) -> None:
    """
    Creates the validation_labels.txt file by combining IDs from positive and negative FASTA files.
    """
    print(f"Creating validation labels file: {output_labels}...")
    all_labels = {}

    # Process positive sequences
    for record in SeqIO.parse(positive_fasta, "fasta"):
        all_labels[record.id] = "1" # Label as positive

    # Process negative sequences
    for record in SeqIO.parse(negative_fasta, "fasta"):
        if record.id in all_labels:
            print(f"WARNING: Duplicate ID '{record.id}' found in both positive and negative sets. This should not happen.", file=sys.stderr)
        all_labels[record.id] = "0" # Label as negative

    try:
        with open(output_labels, "w") as f_out:
            for seq_id in sorted(all_labels.keys()):
                f_out.write(f"{seq_id}\t{all_labels[seq_id]}\n")
    except Exception as e:
        print(f"ERROR: Failed to write validation labels file: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"Successfully created validation labels file with {len(all_labels)} entries.")


# --- Main pipeline for data preparation ---
if __name__ == "__main__":
    CONFIG = load_config()

    # --- Paths from config (using get_full_path for robustness) ---
    seed_alignment_path = get_full_path(CONFIG["seed_alignment"])
    positive_validation_fasta_path = get_full_path(CONFIG["positive_validation_fasta"])
    non_kunitz_validation_fasta_path = get_full_path(CONFIG["non_kunitz_validation_fasta"])
    validation_labels_txt_path = get_full_path(CONFIG["validation_labels_txt"])
    swissprot_fasta_path = get_full_path(CONFIG["swissprot_fasta"])

    # Ensure output directories exist
    os.makedirs(positive_validation_fasta_path.parent, exist_ok=True)
    os.makedirs(non_kunitz_validation_fasta_path.parent, exist_ok=True)
    os.makedirs(swissprot_fasta_path.parent, exist_ok=True) # For swissprot_database dir

    print("--- Preparing Validation Datasets ---")

    # 1. Prepare Positive Validation Set
    print("\n1. Preparing Positive Validation Set (Kunitz domains not in training data)...")
    if not check_stockholm(seed_alignment_path):
        print("ERROR: Training seed alignment is missing or invalid. Cannot prepare positive validation set.", file=sys.stderr)
        sys.exit(1)
    training_ids = parse_stockholm_ids(seed_alignment_path)
    print(f"Found {len(training_ids)} training sequence IDs.")

    # Pfam full alignment for Kunitz (PF00014.full.gz)
    pfam_full_alignment_gz = positive_validation_fasta_path.parent / "PF00014.full.gz"
    if not pfam_full_alignment_gz.exists() or pfam_full_alignment_gz.stat().st_size == 0:
        print(f"ERROR: Pfam Kunitz full alignment '{pfam_full_alignment_gz}' not found or empty.", file=sys.stderr)
        print(f"Please download 'PF00014.full.gz' from Pfam (https://pfam.xfam.org/family/PF00014) and place it in '{pfam_full_alignment_gz.parent}'.", file=sys.stderr)
        sys.exit(1)

    extract_pfam_kunitz_sequences(pfam_full_alignment_gz, training_ids, positive_validation_fasta_path, CONFIG)
    if not positive_validation_fasta_path.exists() or positive_validation_fasta_path.stat().st_size == 0:
        print("ERROR: Positive validation FASTA was not generated correctly.", file=sys.stderr)
        sys.exit(1)


    # 2. Prepare Negative Validation Set (Conditional based on strategy)
    print("\n2. Preparing Negative Validation Set...")
    negative_set_strategy = CONFIG['negative_set_strategy']

    if negative_set_strategy == 'random_swissprot':
        sample_non_kunitz_from_swissprot(swissprot_fasta_path, CONFIG['num_negative_samples'], non_kunitz_validation_fasta_path)
    elif negative_set_strategy == 'structurally_similar':
        # Need directories for PDBs if using structurally similar negatives
        raw_pdb_dir = get_full_path(CONFIG['data_dir']) / "raw_pdb_files_neg" # Separate dir for negatives
        chain_dir = raw_pdb_dir / "chains_for_neg_validation"
        prepare_structurally_similar_negative_set(
            CONFIG['structurally_similar_negative_pdb_ids'],
            CONFIG['structurally_similar_negative_pdb_chains'],
            non_kunitz_validation_fasta_path,
            raw_pdb_dir,
            chain_dir,
            CONFIG # Pass config for clustering parameters
        )
    
    if not non_kunitz_validation_fasta_path.exists() or non_kunitz_validation_fasta_path.stat().st_size == 0:
        print("ERROR: Negative validation FASTA was not generated correctly.", file=sys.stderr)
        sys.exit(1)


    # 3. Create Combined Validation Labels File
    print("\n3. Creating Combined Validation Labels File...")
    create_validation_labels(positive_validation_fasta_path, non_kunitz_validation_fasta_path, validation_labels_txt_path)
    if not validation_labels_txt_path.exists() or validation_labels_txt_path.stat().st_size == 0:
        print("ERROR: Validation labels file was not generated correctly.", file=sys.stderr)
        sys.exit(1)

    print("\n✅ Data preparation completed successfully!")
