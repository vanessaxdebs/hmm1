#!/usr/bin/env python3
"""
Main pipeline script for building, validating, and applying a
structure-informed Profile Hidden Markov Model (HMM) for the Kunitz domain.
"""

import os
import sys
import subprocess
import gzip
import json
from pathlib import Path
from datetime import datetime
import yaml
from Bio import AlignIO, SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
import matplotlib.pyplot as plt # Import matplotlib for plotting

# --- Helper Functions ---

def load_config(config_file: Path = None) -> dict:
    """Loads configuration from a YAML file."""
    # Define script_dir and project_root_guess unconditionally at the start
    script_dir = Path(__file__).resolve().parent
    project_root_guess = script_dir.parent

    if config_file is None:
        # Assume config.yaml is in a 'config' directory one level up from 'scripts'
        config_file = project_root_guess / "config" / "config.yaml"

    print(f"DEBUG: script_dir (parent of hmm.py): {script_dir}")
    print(f"DEBUG: project_root_guess: {project_root_guess}")
    print(f"DEBUG: config_path that script is looking for: {config_file}")
    print(f"DEBUG: config_path.exists() returned {config_file.exists()}: {config_file}")

    if not config_file.exists():
        print(f"ERROR: Config file not found at {config_file}", file=sys.stderr)
        sys.exit(1)

    # IMPORTANT: 'output_dir' is NOT in config.yaml directly. It's derived from 'results_dir'.
    # So, 'results_dir' should be in required_fields, not 'output_dir'.
    required_fields = {
        'data_dir': str,
        'results_dir': str, # Corrected: Expect results_dir, not output_dir
        'scripts_dir': str,
        'config_dir': str,
        'seed_alignment': str,
        'positive_validation_fasta': str,
        'non_kunitz_validation_fasta': str,
        'validation_labels_txt': str,
        'swissprot_fasta': str,
        'e_value_cutoff': float,
        'negative_set_strategy': str,
        'clustering_identity_threshold': float, # Added for consistency
        'clustering_length_difference_cutoff': float, # Added for consistency
    }

    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        # Validate required fields
        for field, expected_type in required_fields.items():
            if field not in config or not isinstance(config[field], expected_type):
                print(f"Configuration error in {config_file}: Missing or invalid required config field: {field}. Expected type: {expected_type.__name__}", file=sys.stderr)
                sys.exit(1)

        # Validate strategy-specific fields (copied from data_prep.py for consistency)
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

        # Convert path strings to Path objects for easier handling AFTER all validation
        # This is done here to ensure all config values are validated before path conversion
        config['data_dir'] = Path(config['data_dir'])
        config['results_dir'] = Path(config['results_dir'])
        config['scripts_dir'] = Path(config['scripts_dir'])
        config['config_dir'] = Path(config['config_dir'])
        config['seed_alignment'] = Path(config['seed_alignment'])
        config['positive_validation_fasta'] = Path(config['positive_validation_fasta'])
        config['non_kunitz_validation_fasta'] = Path(config['non_kunitz_validation_fasta'])
        config['validation_labels_txt'] = Path(config['validation_labels_txt'])
        config['swissprot_fasta'] = Path(config['swissprot_fasta'])

        return config

    except yaml.YAMLError as e:
        print(f"ERROR: Error parsing config.yaml: {e}", file=sys.stderr)
        sys.exit(1)

def get_full_path(relative_path: Path, project_root: Path) -> Path:
    """Converts a relative Path object to an absolute Path relative to project root."""
    return project_root / relative_path

# --- File Check Helper Functions (Copied from data_prep.py and now needed here) ---
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
# --- End of File Check Helper Functions ---


def run_hmmbuild(seed_alignment_file: Path, output_hmm_file: Path) -> None:
    """Builds an HMM from a Stockholm alignment file."""
    print(f"Building HMM: hmmbuild {output_hmm_file} {seed_alignment_file}")
    cmd = ["hmmbuild", str(output_hmm_file), str(seed_alignment_file)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("HMM built successfully.")
    except FileNotFoundError:
        print("ERROR: hmmbuild command not found. Please install HMMER or ensure it's in your PATH.", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: hmmbuild failed: {e.stderr}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during hmmbuild: {e}", file=sys.stderr)
        sys.exit(1)

def run_hmmsearch(hmm_file: Path, fasta_file: Path, output_dir: Path, tag: str, e_value: float) -> Path:
    """Runs hmmsearch against a FASTA file."""
    tblout_file = output_dir / f"hmmsearch_{tag}.tbl"
    print(f"Running hmmsearch: hmmsearch --tblout {tblout_file} -E {e_value} {hmm_file} {fasta_file}")
    cmd = ["hmmsearch", "--tblout", str(tblout_file), "-E", str(e_value), str(hmm_file), str(fasta_file)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"hmmsearch for '{tag}' completed successfully.")
        return tblout_file
    except FileNotFoundError:
        print("ERROR: hmmsearch command not found. Please install HMMER or ensure it's in your PATH.", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: hmmsearch failed for '{tag}': {e.stderr}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during hmmsearch for '{tag}': {e}", file=sys.stderr)
        sys.exit(1)

def parse_tblout(tblout_file: Path, e_value_cutoff: float) -> set:
    """Parses hmmsearch --tblout file and returns a set of hit sequence IDs."""
    hits = set()
    try:
        with open(tblout_file, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) > 4: # Ensure enough columns for parsing
                    seq_id = parts[0]
                    e_value = float(parts[4])
                    if e_value <= e_value_cutoff:
                        hits.add(seq_id)
    except FileNotFoundError:
        print(f"ERROR: tblout file '{tblout_file}' not found.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to parse tblout file '{tblout_file}': {e}", file=sys.stderr)
        sys.exit(1)
    return hits

def load_labels(label_file: Path) -> dict:
    """Loads true labels from a text file (ID TAB Label)."""
    labels = {}
    try:
        with open(label_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                parts = line.split('\t')
                if len(parts) == 2:
                    labels[parts[0]] = parts[1]
                else:
                    print(f"WARNING: Skipping malformed label line: '{line}' in {label_file}", file=sys.stderr)
    except FileNotFoundError:
        print(f"ERROR: Label file '{label_file}' not found.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to load labels from '{label_file}': {e}", file=sys.stderr)
        sys.exit(1)
    return labels

def evaluate_performance(predicted_hits: set, true_labels: dict) -> dict:
    """
    Evaluates classification performance based on predicted hits and true labels.
    Returns a dictionary of metrics.
    """
    tp = 0 # True Positives: Actual Kunitz, Predicted Kunitz
    fp = 0 # False Positives: Actual Non-Kunitz, Predicted Kunitz
    fn = 0 # False Negatives: Actual Kunitz, Predicted Non-Kunitz
    tn = 0 # True Negatives: Actual Non-Kunitz, Predicted Non-Kunitz

    all_validation_ids = set(true_labels.keys())

    for seq_id in all_validation_ids:
        is_true_positive = (true_labels.get(seq_id) == "1")
        is_predicted_positive = (seq_id in predicted_hits)

        if is_true_positive and is_predicted_positive:
            tp += 1
        elif not is_true_positive and is_predicted_positive:
            fp += 1
        elif is_true_positive and not is_predicted_positive:
            fn += 1
        elif not is_true_positive and not is_predicted_positive:
            tn += 1

    total_samples = tp + fp + fn + tn

    accuracy = (tp + tn) / total_samples if total_samples > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0 # Specificity: True Neg Rate

    metrics = {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity
    }
    return metrics

def run_hmmlogo(hmm_file: Path, output_dir: Path) -> None:
    """Generates an HMM logo from the HMM file."""
    logo_output_path = output_dir / "hmm_logo.png"
    print(f"Generating HMM logo: hmmlogo -o {logo_output_path} {hmm_file}")
    cmd = ["hmmlogo", "-o", str(logo_output_path), str(hmm_file)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"HMM logo saved to {logo_output_path}")
    except FileNotFoundError:
        print("ERROR: hmmlogo command not found. Please install HMMER (which includes hmmlogo) or ensure it's in your PATH.", file=sys.stderr)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: hmmlogo failed: {e.stderr}", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred during hmmlogo generation: {e}", file=sys.stderr)

# --- Plotting Functions ---
def plot_confusion_bar(metrics, output_path):
    """Plot confusion matrix as a bar chart."""
    labels = ['TP', 'FP', 'FN', 'TN']
    values = [metrics['TP'], metrics['FP'], metrics['FN'], metrics['TN']]
    plt.figure(figsize=(6,4))
    bars = plt.bar(labels, values, color=['green', 'red', 'orange', 'blue'])
    plt.title("Confusion Matrix Counts")
    plt.ylabel("Count")
    plt.xlabel("Prediction Outcome")
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, int(yval), ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_metrics_summary(metrics, output_path):
    """Plot precision, recall, accuracy, F1, specificity as a bar chart."""
    labels = ['Accuracy', 'Precision', 'Recall', 'F1', 'Specificity']
    values = [metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1'], metrics['specificity']]
    plt.figure(figsize=(8,4)) # Slightly wider to accommodate Specificity
    bars = plt.bar(labels, values, color=['skyblue', 'lightgreen', 'gold', 'violet', 'lightcoral']) # Added color for specificity
    plt.ylim(0,1.05)
    plt.title("Performance Metrics")
    plt.ylabel("Score")
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.2f}", ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


# ---------- Main pipeline ----------

if __name__ == "__main__":
    # Get project root from script location
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    # Load configuration
    CONFIG = load_config(project_root / "config" / "config.yaml")

    # --- Paths from config ---
    # Use get_full_path to ensure all paths are absolute
    DATA_DIR = get_full_path(CONFIG["data_dir"], project_root)
    RESULTS_BASE_DIR = get_full_path(CONFIG["results_dir"], project_root) # Base results dir
    SEED_ALIGNMENT_FILE = get_full_path(CONFIG["seed_alignment"], project_root)
    POSITIVE_VALIDATION_FASTA = get_full_path(CONFIG["positive_validation_fasta"], project_root)
    NON_KUNITZ_VALIDATION_FASTA = get_full_path(CONFIG["non_kunitz_validation_fasta"], project_root)
    VALIDATION_LABELS_TXT = get_full_path(CONFIG["validation_labels_txt"], project_root)
    SWISSPROT_FASTA = get_full_path(CONFIG["swissprot_fasta"], project_root)
    E_VALUE_CUTOFF = CONFIG["e_value_cutoff"]

    # Create a unique output directory for this run
    run_output_dir = RESULTS_BASE_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(run_output_dir, exist_ok=True)
    print(f"Configuration loaded. Results will be saved to: {run_output_dir}")

    print("\n--- 1. Data File Checks ---")
    all_ok = True
    if not check_stockholm(SEED_ALIGNMENT_FILE): all_ok = False
    if not check_fasta(POSITIVE_VALIDATION_FASTA): all_ok = False
    if not check_fasta(NON_KUNITZ_VALIDATION_FASTA): all_ok = False
    if not check_label_txt(VALIDATION_LABELS_TXT): all_ok = False
    
    # Swissprot fasta can be checked for existence, but not critical for validation steps
    if not SWISSPROT_FASTA.exists():
        print(f"WARNING: SwissProt FASTA file '{SWISSPROT_FASTA}' not found. Full SwissProt scan will be skipped.", file=sys.stderr)

    if not all_ok:
        print("\nERROR: Some required input files for validation are missing or invalid. Please ensure you've run 'data_prep.py' and checked manual files (like non_kunitz_proteins.fasta if it was manual).", file=sys.stderr)
        sys.exit(1)

    print("\n--- 2. Building HMM ---")
    hmm_file = run_output_dir / "kunitz.hmm"
    run_hmmbuild(SEED_ALIGNMENT_FILE, hmm_file)

    print("\n--- 3. Running HMMER Search on Validation Sets ---")
    # Search against positive validation set
    val_pos_tbl = run_hmmsearch(hmm_file, POSITIVE_VALIDATION_FASTA, run_output_dir, tag="validation_pos", e_value=E_VALUE_CUTOFF)
    # Search against negative validation set
    val_neg_tbl = run_hmmsearch(hmm_file, NON_KUNITZ_VALIDATION_FASTA, run_output_dir, tag="validation_neg", e_value=E_VALUE_CUTOFF)

    # Parse hits from both searches
    predicted_pos_hits_from_tbl = parse_tblout(val_pos_tbl, E_VALUE_CUTOFF)
    predicted_neg_hits_from_tbl = parse_tblout(val_neg_tbl, E_VALUE_CUTOFF)

    # Combine all predicted hits. Note: If a non-Kunitz sequence accidentally gets a hit, it's a FP.
    predicted_overall_hits = predicted_pos_hits_from_tbl.union(predicted_neg_hits_from_tbl)

    # Load ALL true labels for the entire validation set (positives and negatives)
    true_labels = load_labels(VALIDATION_LABELS_TXT)

    # Filter predicted_overall_hits to only include IDs present in our true_labels (validation set)
    # This prevents issues if hmmsearch finds hits in sequences not meant for validation.
    final_predicted_hits_for_eval = {
        seq_id for seq_id in predicted_overall_hits if seq_id in true_labels
    }

    print("\n--- 4. Evaluating Model Performance ---")
    metrics = evaluate_performance(final_predicted_hits_for_eval, true_labels)

    print("\n--- Combined Validation Performance ---")
    with open(run_output_dir / "validation_metrics.txt", "w") as f_metrics:
        for key, val in metrics.items():
            line = f"  {key}: {val:.4f}" if isinstance(val, float) else f"  {key}: {val}"
            print(line)
            f_metrics.write(line + "\n")
    print(f"Metrics saved to {run_output_dir / 'validation_metrics.txt'}")

    confusion_counts_path = run_output_dir / "confusion_matrix_counts.json"
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
    confusion_bar_plot_path = run_output_dir / "confusion_matrix_bar.png"
    metrics_summary_plot_path = run_output_dir / "performance_metrics_summary.png"

    try:
        plot_confusion_bar(metrics, confusion_bar_plot_path)
        print(f"Confusion matrix bar plot saved to {confusion_bar_plot_path}")
        plot_metrics_summary(metrics, metrics_summary_plot_path)
        print(f"Performance metrics bar chart saved to {metrics_summary_plot_path}")
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

    fp_path = run_output_dir / "false_positives.txt"
    fn_path = run_output_dir / "false_negatives.txt"

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
    if not SWISSPROT_FASTA.exists():
        print(f"Skipping full SwissProt scan because '{SWISSPROT_FASTA}' was not found.")
        print("Please download it (e.g., from UniProt FTP) and place it in `data/swissprot_database/`.", file=sys.stderr)
    else:
        swiss_tbl = run_hmmsearch(hmm_file, SWISSPROT_FASTA, run_output_dir, tag="swissprot_scan", e_value=E_VALUE_CUTOFF)
        swiss_hits = parse_tblout(swiss_tbl, E_VALUE_CUTOFF)
        print(f"Total Kunitz domains predicted in SwissProt: {len(swiss_hits)}")

        swissprot_hits_file = run_output_dir / "predicted_kunitz_domains_swissprot.txt"
        with open(swissprot_hits_file, "w") as f:
            for hit_id in sorted(swiss_hits):
                f.write(hit_id + "\n")
        print(f"List of predicted Kunitz domains in SwissProt saved to {swissprot_hits_file}")

    print("\n--- 7. Generating HMM Logo ---")
    run_hmmlogo(hmm_file, run_output_dir)

    # Save a copy of the config for reproducibility
    config_copy_path = run_output_dir / "config.yaml"
    try:
        original_config_file = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
        with open(original_config_file, 'r') as src_file:
            config_content = yaml.safe_load(src_file)
        with open(config_copy_path, "w") as dest_file:
            yaml.safe_dump(config_content, dest_file, indent=4, sort_keys=False)
        print(f"Configuration for this run saved to {config_copy_path}")
    except Exception as e:
        print(f"WARNING: Could not save a copy of config.yaml: {e}", file=sys.stderr)


    print(f"\n✅ Pipeline finished. All results are in: {run_output_dir}")
