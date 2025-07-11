import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
from datetime import datetime

def plot_confusion_matrix(counts: dict, output_dir: Path, run_id: str):
    """
    Plots a confusion matrix heatmap using TP, FP, FN, TN counts.

    Args:
        counts (dict): A dictionary containing 'TP', 'FP', 'FN', 'TN' integer counts.
        output_dir (Path): The base output directory (e.g., 'images/').
        run_id (str): The specific run ID (e.g., 'run_YYYYMMDD_HHMMSS') to include in the filename.
    """
    # Ensure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract counts
    tp = counts.get('TP', 0)
    fp = counts.get('FP', 0)
    fn = counts.get('FN', 0)
    tn = counts.get('TN', 0)

    # Create the matrix
    # Row 1: Actual Positives (TP, FN)
    # Row 2: Actual Negatives (FP, TN)
    # Note: Traditionally, rows are actual, columns are predicted.
    # Our HMM predicts 'positive' if it hits (TP, FP) and 'negative' if it doesn't (FN, TN).
    # So:
    #             Predicted Positive   Predicted Negative
    # Actual Positive     TP               FN
    # Actual Negative     FP               TN
    confusion_matrix = [[tp, fn],
                        [fp, tn]]

    plt.figure(figsize=(7, 6)) # Adjusted figure size
    
    # Use seaborn for a nice heatmap
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=['Predicted Positive', 'Predicted Negative'],
                yticklabels=['Actual Positive', 'Actual Negative'],
                linewidths=.5, linecolor='black', annot_kws={"size": 14}) # Add lines, adjust font size

    plt.title(f"Confusion Matrix for Run: {run_id}", fontsize=16)
    plt.ylabel("Actual Class", fontsize=12)
    plt.xlabel("Predicted Class", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10, rotation=0) # Ensure Y-labels are not rotated

    plot_filename = output_dir / f"confusion_matrix_kunitz_{run_id}.png"
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=300) # Save with higher DPI
    plt.close()
    print(f"Confusion matrix plot saved to {plot_filename}")

if __name__ == "__main__":
    # Define the base results directory where HMM runs are stored
    results_base_dir = Path("results")
    images_output_dir = Path("images") # Directory for saving plots

    if not results_base_dir.exists():
        print(f"ERROR: Results directory '{results_base_dir}' not found.", file=sys.stderr)
        sys.exit(1)

    # Find the most recent run directory
    run_dirs = sorted([d for d in results_base_dir.iterdir() if d.is_dir() and d.name.startswith("run_")],
                      key=lambda d: d.name, reverse=True)

    if not run_dirs:
        print(f"No HMM run directories found in '{results_base_dir}'. Please run hmm.py first.", file=sys.stderr)
        sys.exit(1)

    latest_run_dir = run_dirs[0]
    latest_run_id = latest_run_dir.name
    json_file_path = latest_run_dir / "confusion_matrix_counts.json"

    if not json_file_path.exists():
        print(f"ERROR: 'confusion_matrix_counts.json' not found in the latest run directory: {latest_run_dir}", file=sys.stderr)
        print("Please ensure your hmm.py script successfully generates this file.", file=sys.stderr)
        sys.exit(1)

    try:
        with open(json_file_path, 'r') as f:
            confusion_counts = json.load(f)
        
        print(f"Loaded confusion counts from: {json_file_path}")
        plot_confusion_matrix(confusion_counts, images_output_dir, latest_run_id)

    except json.JSONDecodeError as e:
        print(f"ERROR: Could not parse JSON from {json_file_path}: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)
