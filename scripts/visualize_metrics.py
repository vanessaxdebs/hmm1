import matplotlib.pyplot as plt
import numpy as np # Added for potential future use, good practice for plotting scripts

def plot_confusion_bar(metrics, output_path):
    """
    Plot confusion matrix counts (TP, FP, FN, TN) as a bar chart.

    Args:
        metrics (dict): A dictionary containing 'TP', 'FP', 'FN', 'TN' counts.
        output_path (Path): The file path (including filename and extension, e.g., .png)
                            where the plot will be saved.
    """
    labels = ['True Positives (TP)', 'False Positives (FP)',
              'False Negatives (FN)', 'True Negatives (TN)']
    
    # Use .get() with a default of 0 in case a key is missing, for robustness
    values = [
        metrics.get('TP', 0),
        metrics.get('FP', 0),
        metrics.get('FN', 0),
        metrics.get('TN', 0)
    ]

    # Define colors for better distinction
    colors = ['#28a745', '#dc3545', '#ffc107', '#007bff'] # Green, Red, Orange, Blue

    plt.figure(figsize=(8, 6)) # Increased figure size for better readability
    bars = plt.bar(labels, values, color=colors)

    plt.title("Confusion Matrix Counts", fontsize=16)
    plt.ylabel("Count", fontsize=12)
    plt.xlabel("Classification Outcome", fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10) # Rotate labels for better fit
    plt.yticks(fontsize=10)

    # Add value labels on top of the bars
    for bar in bars:
        yval = bar.get_height()
        if yval > 0: # Only label bars with a count greater than 0
            plt.text(bar.get_x() + bar.get_width() / 2, yval + (max(values) * 0.02), # Adjust offset based on max value
                     int(yval), ha='center', va='bottom', fontsize=10)

    plt.tight_layout() # Adjust layout to prevent labels from overlapping
    plt.savefig(output_path, dpi=300) # Save with higher DPI for better quality
    plt.close()


def plot_metrics_summary(metrics, output_path):
    """
    Plot precision, recall, accuracy, and F1-score as a bar chart.

    Args:
        metrics (dict): A dictionary containing 'accuracy', 'precision',
                        'recall', and 'f1' scores.
        output_path (Path): The file path (including filename and extension, e.g., .png)
                            where the plot will be saved.
    """
    labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    # Use .get() with a default of 0.0 in case a key is missing
    values = [
        metrics.get('accuracy', 0.0),
        metrics.get('precision', 0.0),
        metrics.get('recall', 0.0),
        metrics.get('f1', 0.0)
    ]

    # Define colors
    colors = ['#17a2b8', '#28a745', '#ffc107', '#6f42c1'] # Skyblue, Lightgreen, Gold, Violet

    plt.figure(figsize=(7, 5)) # Adjusted figure size
    bars = plt.bar(labels, values, color=colors)

    plt.ylim(0, 1.05) # Set Y-axis limits from 0 to slightly above 1 for text clarity
    plt.title("HMM Performance Metrics", fontsize=16)
    plt.ylabel("Score", fontsize=12)
    plt.xlabel("Metric", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # Add value labels on top of the bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.02, # Adjust offset
                 f"{yval:.2f}", ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
