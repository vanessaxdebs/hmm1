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
import json
# Import plotting functions from the local module
from visualize_metrics import plot_confusion_bar, plot_metrics_summary

# ---------- Biopython-based utilities and Checks ----------

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
        print(f"ERROR: {path} is not a valid Stockholm file or has parsing issues.", file=sys.stderr)
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
        sys.exit(1) # Exit if essential FASTA is malformed
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
                if line and len(line.split('\t')) != 2: # Expecting "ID\tLABEL"
                    print(f"ERROR: {path} has a line with wrong format: '{line}' (expected ID\\tLABEL)", file=sys.stderr)
                    return False
        return True
    except Exception as e:
        print(f"ERROR: Error reading label file {path}: {e}", file=sys.stderr)
        sys.exit(1)
        return False

# ---------- Config loading & output dir creation ----------

def find_config() -> Path:
    """Finds the config.yaml file relative to the script's location."""
    script_dir = Path(__file__).parent # This is 'scripts/'
    config_path = script_dir.parent / "config" / "config.yaml"
    if config_path.exists():
        return config_path.resolve()
        
    raise FileNotFoundError("config.yaml not found! Ensure it's in 'config/' relative to the project root.")

def load_config() -> Dict:
    """Loads and validates configuration from config.yaml."""
    try:
        config_file = find_config()
    except FileNotFoundError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        sys.exit(1)

    required_fields = {
        'output_dir': str,
        'seed_alignment': str,
        'positive_validation_fasta': str,
        'non_kunitz_validation_fasta': str, # NEW: This is now required
        'validation_labels_txt': str,
        'swissprot_fasta': str,
        'e_value_cutoff': float,
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

        # Resolve all paths relative to the project root (assuming script in 'scripts/')
        project_root = Path(__file__).resolve().parent.parent
        for key in ['seed_alignment', 'positive_validation_fasta', 'non_kunitz_validation_fasta',
                    'validation_labels_txt', 'swissprot_fasta']:
            config[key] = project_root / Path(config[key])
        
        # Create unique run output directory
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_output_dir = project_root / Path(config['output_dir'])
        output_dir = base_output_dir / f"run_{run_id}"
        output_dir.mkdir(parents=True, exist_ok=True)
        config['output_dir'] = output_dir

        print(f"Configuration loaded. Results will be saved to: {output_dir}")
        return config
    except Exception as e:
        print(f"Configuration error in {config_file}: {e}", file=sys.stderr)
        sys.exit(1)

# ---------- HMM and metrics functions ----------

def run_hmmbuild(seed_alignment: Path, hmm_file: Path):
    """Trains a Profile HMM using HMMER's hmmbuild."""
    cmd = ["hmmbuild", str(hmm_file), str(seed_alignment)]
    print(f"Building HMM: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
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
        "--noali",
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
                if line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 13:
                    try:
                        target_name = parts[0]
                        e_value = float(parts[4])
                        if e_value < e_value_cutoff:
                            hits.add(target_name)
                    except ValueError:
                        continue # Skip lines where E-value isn't a float
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
                    parts = line.split('\t')
                    if len(parts) == 2:
                        labels[parts[0]] = parts[1]
                    else:
                        print(f"WARNING: Skipping malformed line in {label_file}: '{line}' (expected ID\\tLABEL)", file=sys.stderr)
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
            tn += 1 # A true negative: correctly not predicted and is truly negative
        elif not is_predicted and is_positive: # This condition should logically be: not is_predicted and is_positive
            fn += 1 # A false negative: not predicted but is truly positive
        
    # Recalculate tn to avoid double counting or logical errors
    tn = len(all_true_labels) - tp - fp - fn

    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return dict(TP=tp, FP=fp, FN=fn, TN=tn, accuracy=acc, precision=prec, recall=rec, f1=f1, specificity=specificity)

def run_hmmlogo(hmm_file: Path, output_dir: Path):
    """Generates an HMM logo using hmmlogo."""
    logo_file = output_dir / "hmm_logo.png"
    cmd = ["hmmlogo", "-o", str(logo_file), str(hmm_file)]
    print(f"Generating HMM logo: {' '.join(cmd)}")
    try:
        # hmmlogo prints to stderr by default, capture for cleaner output
        result = subprocess.run(cmd, check=True, capture_output=True)
        if logo_file.exists() and logo_file.stat().st_size > 0:
            print(f"HMM logo saved to {logo_file}")
        else:
            print(f"WARNING: hmmlogo ran but {logo_file} was not created or is empty. Stderr:\n{result.stderr.decode()}", file=sys.stderr)
    except FileNotFoundError:
        print("ERROR: hmmlogo command not found. Please install HMMER's easel tools or ensure it's in your PATH.", file=sys.stderr)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to generate HMM logo. Stderr:\n{e.stderr.decode()}", file=sys.stderr)


# ---------- Main pipeline ----------

if __name__ == "__main__":
    CONFIG = load_config()

    # --- Paths from config ---
    sto_file = CONFIG["seed_alignment"]
    validation_pos_fasta = CONFIG["positive_validation_fasta"]
    validation_neg_fasta = CONFIG["non_kunitz_validation_fasta"] # Updated key
    validation_labels_txt = CONFIG["validation_labels_txt"]
    swissprot_fasta = CONFIG["swissprot_fasta"]
    output_dir = CONFIG["output_dir"]
    e_value_cutoff = CONFIG["e_value_cutoff"]

    print("\n--- 1. Data File Checks ---")
    all_ok = True
    if not check_stockholm(sto_file): all_ok = False
    if not check_fasta(validation_pos_fasta): all_ok = False
    if not check_fasta(validation_neg_fasta): all_ok = False # Now this file is auto-generated
    if not check_label_txt(validation_labels_txt): all_ok = False
    
    # Swissprot fasta can be checked for existence, but not critical for validation steps
    if not swissprot_fasta.exists():
        print(f"WARNING: SwissProt FASTA file '{swissprot_fasta}' not found. Full SwissProt scan will be skipped.", file=sys.stderr)

    if not all_ok:
        print("\nERROR: Some required input files for validation are missing or invalid. Please ensure you've run 'data_prep.py' and checked manual files (like non_kunitz_proteins.fasta if it was manual).", file=sys.stderr)
        sys.exit(1)

    print("\n--- 2. Building HMM ---")
    hmm_file = output_dir / "kunitz.hmm"
    run_hmmbuild(sto_file, hmm_file)

    print("\n--- 3. Running HMMER Search on Validation Sets ---")
    # Search against positive validation set
    val_pos_tbl = run_hmmsearch(hmm_file, validation_pos_fasta, output_dir, tag="validation_pos", e_value=e_value_cutoff)
    # Search against negative validation set
    val_neg_tbl = run_hmmsearch(hmm_file, validation_neg_fasta, output_dir, tag="validation_neg", e_value=e_value_cutoff)

    # Parse hits from both searches
    predicted_pos_hits_from_tbl = parse_tblout(val_pos_tbl, e_value_cutoff)
    predicted_neg_hits_from_tbl = parse_tblout(val_neg_tbl, e_value_cutoff)

    # Combine all predicted hits. Note: If a non-Kunitz sequence accidentally gets a hit, it's a FP.
    predicted_overall_hits = predicted_pos_hits_from_tbl.union(predicted_neg_hits_from_tbl)

    # Load ALL true labels for the entire validation set (positives and negatives)
    true_labels = load_labels(validation_labels_txt)

    # Filter predicted_overall_hits to only include IDs present in our true_labels (validation set)
    # This prevents issues if hmmsearch finds hits in sequences not meant for validation.
    final_predicted_hits_for_eval = {
        seq_id for seq_id in predicted_overall_hits if seq_id in true_labels
    }

    print("\n--- 4. Evaluating Model Performance ---")
    metrics = evaluate_performance(final_predicted_hits_for_eval, true_labels)

    print("\n--- Combined Validation Performance ---")
    with open(output_dir / "validation_metrics.txt", "w") as f_metrics:
        for key, val in metrics.items():
            line = f"  {key}: {val:.4f}" if isinstance(val, float) else f"  {key}: {val}"
            print(line)
            f_metrics.write(line + "\n")
    print(f"Metrics saved to {output_dir / 'validation_metrics.txt'}")

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

    print("\n--- Generating Metric Visualizations ---")
    confusion_bar_plot_path = output_dir / "confusion_matrix_bar.png"
    metrics_summary_plot_path = output_dir / "performance_metrics_summary.png"

    try:
        plot_confusion_bar(metrics, confusion_bar_plot_path)
        print(f"Confusion matrix bar plot saved to {confusion_bar_plot_path}")
        plot_metrics_summary(metrics, metrics_summary_plot_path)
        print(f"Performance metrics summary plot saved to {metrics_summary_plot_path}")
    except Exception as e:
        print(f"WARNING: Could not generate additional metric plots: {e}", file=sys.stderr)


    print("\n--- 5. Analyzing False Positives and False Negatives ---")
    false_positives = []
    false_negatives = []

    for seq_id, true_label in true_labels.items():
        is_predicted = seq_id in final_predicted_hits_for_eval
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
        print(f"Skipping full SwissProt scan because '{swissprot_fasta}' was not found.")
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

    # Save a copy of the config for reproducibility
    config_copy_path = output_dir / "config.yaml"
    try:
        original_config_file = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
        with open(original_config_file, 'r') as src_file:
            config_content = yaml.safe_load(src_file)
        with open(config_copy_path, "w") as dest_file:
            yaml.safe_dump(config_content, dest_file, indent=4, sort_keys=False)
        print(f"Configuration for this run saved to {config_copy_path}")
    except Exception as e:
        print(f"WARNING: Could not save a copy of config.yaml: {e}", file=sys.stderr)


    print(f"\n✅ Pipeline finished. All results are in: {output_dir}")
