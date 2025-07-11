#!/usr/bin/env python3
"""
Kunitz-type Protease Inhibitor Domain - HMM Profile Pipeline (Biopython version)
Compatible with hmmologs project standards.
Includes independent positive and negative validation.
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, Set, Tuple
import yaml
from datetime import datetime
from Bio import SeqIO

# ---------- Biopython-based utilities ----------

def sto_to_fasta(sto_path: Path, fasta_path: Path):
    """Converts a Stockholm alignment to FASTA format."""
    try:
        count = SeqIO.write(SeqIO.parse(sto_path, "stockholm"), fasta_path, "fasta")
        print(f"Converted {count} sequences from {sto_path} to {fasta_path}")
    except Exception as e:
        print(f"ERROR: Failed to convert Stockholm to FASTA for {sto_path}: {e}", file=sys.stderr)
        sys.exit(1)

def fasta_to_label_txt(fasta_path: Path, txt_path: Path, label: str = "1"):
    """Converts a FASTA file to a two-column label text file (ID \t label)."""
    count = 0
    try:
        with open(txt_path, "w") as txt:
            for record in SeqIO.parse(fasta_path, "fasta"):
                txt.write(f"{record.id}\t{label}\n")
                count += 1
        print(f"Wrote {count} sequence labels to {txt_path}")
    except Exception as e:
        print(f"ERROR: Failed to create label file from {fasta_path}: {e}", file=sys.stderr)
        sys.exit(1)

def check_stockholm(path: Path) -> bool:
    """Checks if a file is a valid Stockholm format."""
    if not path.exists() or path.stat().st_size == 0:
        print(f"ERROR: {path} is missing or empty.", file=sys.stderr)
        return False
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            next(SeqIO.parse(f, "stockholm"))
        return True
    except Exception:
        print(f"ERROR: {path} is not a valid Stockholm file.", file=sys.stderr)
        return False

def check_fasta(path: Path) -> bool:
    """Checks if a file is a valid FASTA format."""
    if not path.exists() or path.stat().st_size == 0:
        print(f"ERROR: {path} is missing or empty.", file=sys.stderr)
        return False
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            next(SeqIO.parse(f, "fasta"))
        return True
    except Exception:
        print(f"ERROR: {path} is not a valid FASTA file.", file=sys.stderr)
        return False

def check_label_txt(path: Path) -> bool:
    """Checks if a label text file exists and has the correct format."""
    if not path.exists() or path.stat().st_size == 0:
        print(f"ERROR: {path} is missing or empty.", file=sys.stderr)
        return False
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if line and len(line.split()) != 2: # Expecting "ID\tLABEL"
                    print(f"ERROR: {path} has a line with wrong format: '{line}' (expected ID\\tLABEL)", file=sys.stderr)
                    return False
        return True
    except Exception as e:
        print(f"ERROR: Error reading label file {path}: {e}", file=sys.stderr)
        return False

# ---------- Config loading & output dir creation ----------

def find_config() -> Path:
    """Finds the config.yaml file."""
    script_dir = Path(__file__).parent
    for path in [
        script_dir / "../config/config.yaml", # Common for running from scripts/
        script_dir / "config/config.yaml",
        Path("config/config.yaml") # For running from root
    ]:
        if path.exists():
            return path.resolve()
    raise FileNotFoundError("config.yaml not found! Ensure it's in config/ or project root.")

def load_config(config_file: Path = None) -> Dict:
    """Loads and validates configuration from config.yaml."""
    if config_file is None:
        try:
            config_file = find_config()
        except FileNotFoundError as e:
            print(f"Configuration error: {e}", file=sys.stderr)
            sys.exit(1)

    required_fields = {
        'output_dir': str,
        'seed_alignment': str, # Path to kunitz_seed_training.sto
        'validation_positives_fasta': str, # Path to positive_validation.fasta
        'validation_positives_labels': str, # Path to validation_labels.txt
        'validation_negatives_fasta': str, # Path to negative_validation.fasta
        'validation_negatives_labels': str, # Path to negative_labels.txt
        'swissprot_fasta': str, # Path to uniprot_sprot.fasta
        'e_value_cutoff': float,
        # 'pdb_id': str # This field seems redundant for the overall HMM pipeline
    }
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        for field, field_type in required_fields.items():
            if field not in config:
                raise ValueError(f"Missing required config field: {field}")
            if not isinstance(config[field], field_type):
                if field_type == float and isinstance(config[field], int):
                    config[field] = float(config[field])
                else:
                    raise ValueError(f"Invalid type for {field}, expected {field_type.__name__} but got {type(config[field]).__name__}")
        
        # Resolve all paths relative to the project root or data directory
        for key in ['seed_alignment', 'validation_positives_fasta', 'validation_positives_labels',
                    'validation_negatives_fasta', 'validation_negatives_labels', 'swissprot_fasta']:
            config[key] = Path(config[key])
        
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S") # Added seconds for more uniqueness
        output_dir = Path(config['output_dir']) / f"run_{run_id}"
        output_dir.mkdir(parents=True, exist_ok=True)
        config['output_dir'] = output_dir
        
        print(f"Configuration loaded. Results will be saved to: {output_dir}")
        return config
    except Exception as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        sys.exit(1)

# ---------- HMM and metrics functions ----------

def run_hmmbuild(seed_alignment: Path, hmm_file: Path):
    """Trains a Profile HMM using HMMER's hmmbuild."""
    cmd = ["hmmbuild", str(hmm_file), str(seed_alignment)]
    print(f"Building HMM: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True) # Capture output for better debugging
        print(f"HMM built successfully: {hmm_file}")
    except FileNotFoundError:
        print("ERROR: hmmbuild command not found. Please ensure HMMER is installed and in your PATH.", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: hmmbuild failed. Stderr:\n{e.stderr}", file=sys.stderr)
        sys.exit(1)

def run_hmmsearch(hmm_file: Path, fasta_file: Path, output_dir: Path, tag: str = "search", e_value: float = 1e-5) -> Path:
    """Runs hmmsearch and returns the path to the .tblout file."""
    tblout = output_dir / f"hmmsearch_{tag}.tbl"
    cmd = [
        "hmmsearch",
        "--tblout", str(tblout),
        "-E", str(e_value),
        "--noali", # Don't output alignment, just hits, to save space
        str(hmm_file), str(fasta_file)
    ]
    print(f"Running hmmsearch on {fasta_file.name}: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"hmmsearch results saved to {tblout}")
        return tblout
    except FileNotFoundError:
        print("ERROR: hmmsearch command not found. Please ensure HMMER is installed and in your PATH.", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: hmmsearch failed on {fasta_file.name}. Stderr:\n{e.stderr}", file=sys.stderr)
        sys.exit(1)

def parse_tblout(tbl_file: Path, e_value_cutoff: float) -> Set[str]:
    """Parses hmmsearch --tblout and returns a set of unique hit IDs passing the E-value cutoff."""
    hits = set()
    if not tbl_file.exists() or tbl_file.stat().st_size == 0:
        print(f"WARNING: tblout file {tbl_file} is missing or empty. No hits parsed.", file=sys.stderr)
        return hits
    try:
        with open(tbl_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if line.startswith("#"): # Skip comments
                    continue
                parts = line.split()
                if len(parts) >= 13: # Ensure enough columns for E-value
                    try:
                        target_name = parts[0]
                        e_value = float(parts[4]) # E-value is typically the 5th column (index 4) in the main table, NOT the domain table.
                                                   # For --domtblout, it's parts[12]. For --tblout, it's parts[4].
                        if e_value < e_value_cutoff:
                            hits.add(target_name)
                    except ValueError:
                        # Skip lines where E-value isn't a float (e.g., malformed lines)
                        continue
        return hits
    except Exception as e:
        print(f"ERROR: Error parsing tblout file {tbl_file}: {e}", file=sys.stderr)
        sys.exit(1)

def load_labels(label_file: Path) -> Dict[str, str]:
    """Loads sequence ID -> label mapping from a label file."""
    labels = {}
    if not label_file.exists() or label_file.stat().st_size == 0:
        print(f"WARNING: Label file {label_file} is missing or empty. No labels loaded.", file=sys.stderr)
        return labels
    try:
        with open(label_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    parts = line.split('\t') # Assuming tab-separated
                    if len(parts) == 2:
                        labels[parts[0]] = parts[1]
                    else:
                        print(f"WARNING: Skipping malformed line in {label_file}: '{line}'", file=sys.stderr)
        return labels
    except Exception as e:
        print(f"ERROR: Error loading labels from {label_file}: {e}", file=sys.stderr)
        sys.exit(1)

def evaluate_performance(predicted_hits: Set[str], all_true_labels: Dict[str, str]) -> Dict:
    """
    Evaluates classification performance given predicted hits and true labels.
    Assumes '1' for positive, '0' for negative.
    """
    tp = 0
    fp = 0
    fn = 0
    tn = 0

    for seq_id, true_label in all_true_labels.items():
        is_predicted = seq_id in predicted_hits
        is_positive = (true_label == "1")

        if is_predicted and is_positive:
            tp += 1
        elif is_predicted and not is_positive:
            fp += 1
        elif not is_predicted and is_positive:
            fn += 1
        elif not is_predicted and not is_positive:
            tn += 1

    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    
    # Specificity for negative class
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return dict(TP=tp, FP=fp, FN=fn, TN=tn, accuracy=acc, precision=prec, recall=rec, f1=f1, specificity=specificity)

def run_hmmlogo(hmm_file: Path, output_dir: Path):
    """Generates an HMM logo using hmmlogo."""
    logo_file = output_dir / "hmm_logo.png"
    cmd = ["hmmlogo", "-o", str(logo_file), str(hmm_file)]
    print(f"Generating HMM logo: {' '.join(cmd)}")
    try:
        # hmmlogo often writes directly to file without useful stdout/stderr,
        # so check for existence after run.
        subprocess.run(cmd, check=True, capture_output=True)
        if logo_file.exists() and logo_file.stat().st_size > 0:
            print(f"HMM logo saved to {logo_file}")
        else:
            print(f"WARNING: hmmlogo ran but {logo_file} was not created or is empty. Check hmmlogo installation.", file=sys.stderr)
    except FileNotFoundError:
        print("ERROR: hmmlogo command not found. Please install HMMER's easel tools or ensure it's in your PATH.", file=sys.stderr)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to generate HMM logo. Stderr:\n{e.stderr.decode()}", file=sys.stderr)


# ---------- Main pipeline ----------

if __name__ == "__main__":
    CONFIG = load_config()

    # --- Paths from config ---
    sto_file = CONFIG["seed_alignment"] # data/alignments/kunitz_seed_training.sto
    validation_pos_fasta = CONFIG["validation_positives_fasta"] # data/validation_datasets/positive_validation.fasta
    validation_pos_labels = CONFIG["validation_positives_labels"] # data/validation_datasets/validation_labels.txt
    validation_neg_fasta = CONFIG["validation_negatives_fasta"] # data/validation_datasets/negative_validation.fasta
    validation_neg_labels = CONFIG["validation_negatives_labels"] # data/validation_datasets/negative_labels.txt
    swissprot_fasta = CONFIG["swissprot_fasta"] # data/swissprot_database/uniprot_sprot.fasta
    output_dir = CONFIG["output_dir"]
    e_value_cutoff = CONFIG["e_value_cutoff"]

    print("\n--- 1. Data File Checks ---")
    all_ok = True
    if not check_stockholm(sto_file): all_ok = False
    if not check_fasta(validation_pos_fasta): all_ok = False
    if not check_label_txt(validation_pos_labels): all_ok = False
    if not check_fasta(validation_neg_fasta): all_ok = False
    if not check_label_txt(validation_neg_labels): all_ok = False
    # Swissprot fasta can be downloaded by a separate script if not present
    # if not check_fasta(swissprot_fasta): all_ok = False # Might be downloaded by other script

    if not all_ok:
        print("\nERROR: Some required input files are missing or invalid. Please check the 'data/' directory and run previous scripts.", file=sys.stderr)
        sys.exit(1)

    print("\n--- 2. Building HMM ---")
    hmm_file = output_dir / "kunitz.hmm"
    run_hmmbuild(sto_file, hmm_file)

    print("\n--- 3. Running HMMER Search on Validation Sets ---")
    # Run hmmsearch on positive validation sequences
    val_pos_tbl = run_hmmsearch(hmm_file, validation_pos_fasta, output_dir, tag="validation_pos", e_value=e_value_cutoff)
    predicted_pos_hits = parse_tblout(val_pos_tbl, e_value_cutoff)

    # Run hmmsearch on negative validation sequences
    val_neg_tbl = run_hmmsearch(hmm_file, validation_neg_fasta, output_dir, tag="validation_neg", e_value=e_value_cutoff)
    predicted_neg_hits = parse_tblout(val_neg_tbl, e_value_cutoff)

    # Combine all sequence IDs from both positive and negative validation files
    # to form the comprehensive set of sequences that were *tested*.
    all_validation_seq_ids = set()
    for record in SeqIO.parse(validation_pos_fasta, "fasta"):
        all_validation_seq_ids.add(record.id)
    for record in SeqIO.parse(validation_neg_fasta, "fasta"):
        all_validation_seq_ids.add(record.id)

    # Load ALL true labels for the validation set (positives and negatives)
    true_labels = {}
    true_labels.update(load_labels(validation_pos_labels))
    true_labels.update(load_labels(validation_neg_labels))
    
    # Filter predicted hits to only include those that were actually in the validation set
    # (hmmsearch might sometimes find spurious hits from a database-wide search
    # if the FASTA file contained sequences not in your labels for some reason)
    all_predicted_hits_in_validation = (predicted_pos_hits.union(predicted_neg_hits)).intersection(all_validation_seq_ids)

    print("\n--- 4. Evaluating Model Performance ---")
    metrics = evaluate_performance(all_predicted_hits_in_validation, true_labels)

    print("\n--- Combined Validation Performance ---")
    with open(output_dir / "validation_metrics.txt", "w") as f_metrics:
        for key, val in metrics.items():
            line = f"  {key}: {val:.4f}" if isinstance(val, float) else f"  {key}: {val}"
            print(line)
            f_metrics.write(line + "\n")
    print(f"Metrics saved to {output_dir / 'validation_metrics.txt'}")

      # Save raw confusion matrix counts to a JSON file for plotting
    import json # Ensure json is imported at the top of the file, or here if not already
    confusion_counts_path = output_dir / "confusion_matrix_counts.json"
    confusion_counts = {
        "TP": metrics["TP"],
        "FP": metrics["FP"],
        "FN": metrics["FN"],
        "TN": metrics["TN"]
    }
    with open(confusion_counts_path, "w") as f:
        json.dump(confusion_counts, f, indent=4)
    print(f"Confusion matrix counts saved to {confusion_counts_path}")

    # --- The calls to plot_confusion_bar and plot_metrics_summary should come AFTER these two blocks ---
    print("\n--- Generating Metric Visualizations ---")
    
    # Define paths for the new plots
    confusion_bar_plot_path = output_dir / "confusion_matrix_bar.png"
    metrics_summary_plot_path = output_dir / "performance_metrics_summary.png"

    try:
        # Call the functions from visualize_metrics.py
        plot_confusion_bar(metrics, confusion_bar_plot_path)
        print(f"Confusion matrix bar plot saved to {confusion_bar_plot_path}")
        
        plot_metrics_summary(metrics, metrics_summary_plot_path)
        print(f"Performance metrics summary plot saved to {metrics_summary_plot_path}")
    except Exception as e:
        print(f"WARNING: Could not generate additional metric plots: {e}", file=sys.stderr)

    print("\n--- 5. Analyzing False Positives and False Negatives ---")
    # Identify False Positives and False Negatives based on the combined set
    false_positives = []
    false_negatives = []

    for seq_id, true_label in true_labels.items():
        is_predicted = seq_id in all_predicted_hits_in_validation
        is_positive_true = (true_label == "1")

        if is_predicted and not is_positive_true:
            false_positives.append(seq_id)
        elif not is_predicted and is_positive_true:
            false_negatives.append(seq_id)

    fp_path = output_dir / "false_positives.txt"
    fn_path = output_dir / "false_negatives.txt"

    with open(fp_path, "w") as f:
        for seq_id in sorted(false_positives):
            f.write(seq_id + "\n")

    with open(fn_path, "w") as f:
        for seq_id in sorted(false_negatives):
            f.write(seq_id + "\n")

    print(f"False Positives ({len(false_positives)}): saved to {fp_path.name}")
    print(f"False Negatives ({len(false_negatives)}): saved to {fn_path.name}")
    
    if false_positives:
        print("Sample False Positives:", ", ".join(false_positives[:5]))
    if false_negatives:
        print("Sample False Negatives:", ", ".join(false_negatives[:5]))


    print("\n--- 6. Annotating Full SwissProt Database ---")
    if not swissprot_fasta.exists():
        print(f"WARNING: SwissProt FASTA file '{swissprot_fasta}' not found. Skipping full SwissProt scan.", file=sys.stderr)
        print("Please download it (e.g., from UniProt FTP) and place it in `data/swissprot_database/`.", file=sys.stderr)
    else:
        swiss_tbl = run_hmmsearch(hmm_file, swissprot_fasta, output_dir, tag="swissprot_scan", e_value=e_value_cutoff)
        swiss_hits = parse_tblout(swiss_tbl, e_value_cutoff)
        print(f"Total Kunitz domains predicted in SwissProt: {len(swiss_hits)}")
        
        swissprot_hits_file = output_dir / "predicted_kunitz_domains_swissprot.txt"
        with open(swissprot_hits_file, "w") as f:
            for hit_id in sorted(swiss_hits):
                f.write(hit_id + "\n")
        print(f"List of predicted Kunitz domains in SwissProt saved to {swissprot_hits_file}")

    print("\n--- 7. Generating HMM Logo ---")
    run_hmmlogo(hmm_file, output_dir)

    print(f"\n✅ Pipeline finished. All results are in: {output_dir}")
