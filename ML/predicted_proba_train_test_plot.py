import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_predicted_proba(train_probs: list, test_probs: list, model_threshold: float = None, output_full_path: str = None):
    """
    Plots the distribution of predicted probabilities for train and test sets.

    Args:
        train_probs (list): List of predicted probabilities for the training set.
        test_probs (list): List of predicted probabilities for the test set.
        model_threshold (float, optional): Threshold value to plot as a vertical line. Defaults to None.
        output_full_path (str, optional): Full path (including filename) to save the plot. Defaults to None.
    """
    # Plot distributions
    plt.figure(figsize=(10, 6))
    sns.kdeplot(x=train_probs, label='Train', fill=True, alpha=0.3,clip=(0, 1))
    sns.kdeplot(x=test_probs, label='Test', fill=True, alpha=0.3,clip=(0, 1))
    
    # Add threshold line if provided
    if model_threshold is not None:
        plt.axvline(x=model_threshold, color='red', linestyle='--', label=f'Threshold: {model_threshold}')
    
    plt.xlim(0, 1)
    plt.title('Predicted Probabilities Distribution (Train vs Test)')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Density')
    plt.legend()
    
    if output_full_path:
        plt.savefig(output_full_path)
        print(f"Plot saved to {output_full_path}")
        plt.close()
    else:
        plt.show()
    
    plt.close()
