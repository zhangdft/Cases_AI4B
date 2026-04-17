
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
from sklearn.metrics import r2_score
import os

def parse_simple_gnn_log(log_file):
    epochs = []
    train_losses = []
    val_maes = []
    
    if not os.path.exists(log_file):
        print(f"Log file {log_file} not found.")
        return [], [], []

    with open(log_file, 'r') as f:
        lines = f.readlines()
        
    # Pattern: Epoch 1/10: Train Loss=0.5444, Val MAE=1.3650
    for line in lines:
        match = re.search(r'Epoch (\d+)/\d+: Train Loss=(\d+\.\d+), Val MAE=(\d+\.\d+)', line)
        if match:
            epochs.append(int(match.group(1)))
            train_losses.append(float(match.group(2)))
            val_maes.append(float(match.group(3)))
                
    return epochs, train_losses, val_maes

def parse_simple_gnn_test_results(csv_file):
    if not os.path.exists(csv_file):
        print(f"CSV file {csv_file} not found.")
        return np.array([]), np.array([])
        
    df = pd.read_csv(csv_file, header=None)
    actuals = df[0].values
    predicts = df[1].values
    return actuals, predicts

def plot_simple_gnn_results():
    log_file = 'simple_gnn.out'
    csv_file = 'simple_gnn_test_results.csv'
    output_img = 'simple_gnn_performance.png'
    
    print(f"Parsing log file: {log_file}")
    epochs, losses, val_maes = parse_simple_gnn_log(log_file)
    
    print(f"Parsing test results: {csv_file}")
    actuals, predicts = parse_simple_gnn_test_results(csv_file)
    
    if len(actuals) == 0:
        print("No test results found to plot parity plot.")
        # Plot only loss if possible
        if len(epochs) > 0:
            plt.figure(figsize=(8, 6))
            plt.plot(epochs, losses, 'b-o', label='Train Loss (MSE)')
            plt.plot(epochs, val_maes, 'r-s', label='Val MAE')
            plt.xlabel('Epoch')
            plt.ylabel('Value')
            plt.title('Simple GNN Training Process')
            plt.legend()
            plt.savefig(output_img)
            print(f"Partial results plot saved as '{output_img}'")
        return

    # Calculate R2
    r2 = r2_score(actuals, predicts)
    print(f"Calculated R2: {r2:.4f}")
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Loss and MAE curves
    ax1.plot(epochs, losses, 'b-o', label='Train Loss (MSE)')
    ax1_mae = ax1.twinx()
    ax1_mae.plot(epochs, val_maes, 'r-s', label='Val MAE')
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)', color='b')
    ax1_mae.set_ylabel('MAE', color='r')
    ax1.set_title('Simple GNN Training Process')
    
    # Combine legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_mae.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper right')
    
    # Plot 2: Parity plot (Actual vs Predicted)
    min_val = min(min(actuals), min(predicts))
    max_val = max(max(actuals), max(predicts))
    ax2.scatter(actuals, predicts, alpha=0.5, color='green', edgecolors='k')
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    ax2.set_xlabel('Actual Shear Modulus (GPa)')
    ax2.set_ylabel('Predicted Shear Modulus (GPa)')
    ax2.set_title(f'Simple GNN Performance (Test Set)\n$R^2 = {r2:.4f}$')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_img, dpi=300)
    print(f"Results plot saved as '{output_img}'")

if __name__ == "__main__":
    plot_simple_gnn_results()
