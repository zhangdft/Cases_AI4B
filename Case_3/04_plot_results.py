
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
from sklearn.metrics import r2_score

def parse_log(log_file):
    epochs = []
    train_losses = []
    train_maes = []
    test_maes = []
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
        
    # Pattern for training lines: Epoch: [0][20/30] ... Loss 1.1368 (1.1741) MAE 33.381 (33.176)
    # Pattern for test lines:  * MAE 57.502
    
    current_epoch = -1
    for line in lines:
        if 'Epoch:' in line:
            match = re.search(r'Epoch: \[(\d+)\]\[\d+/\d+\].*Loss \d+\.\d+ \((\d+\.\d+)\).*MAE \d+\.\d+ \((\d+\.\d+)\)', line)
            if match:
                epoch = int(match.group(1))
                loss = float(match.group(2))
                mae = float(match.group(3))
                
                if epoch > current_epoch:
                    epochs.append(epoch)
                    train_losses.append(loss)
                    train_maes.append(mae)
                    current_epoch = epoch
                else:
                    # Update with the latest average for this epoch
                    train_losses[-1] = loss
                    train_maes[-1] = mae
        
        if ' * MAE' in line:
            match = re.search(r' \* MAE (\d+\.\d+)', line)
            if match:
                test_maes.append(float(match.group(1)))
                
    return epochs, train_losses, train_maes, test_maes

def parse_test_results(csv_file):
    # Format: mp-27505,[57.06089401245117],59.834556579589844
    actuals = []
    predicts = []
    
    with open(csv_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 3:
                # Actual value is in brackets like [57.06]
                actual_str = parts[1].strip('[]')
                actuals.append(float(actual_str))
                predicts.append(float(parts[2]))
                
    return np.array(actuals), np.array(predicts)

def plot_results():
    log_file = 'finetune_gpu.out'
    csv_file = 'test_results.csv'
    
    print(f"Parsing log file: {log_file}")
    epochs, losses, train_maes, test_maes = parse_log(log_file)
    
    print(f"Parsing test results: {csv_file}")
    actuals, predicts = parse_test_results(csv_file)
    
    # Calculate R2
    r2 = r2_score(actuals, predicts)
    print(f"Calculated R2: {r2:.4f}")
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Loss and MAE curves
    ax1.plot(epochs, losses, 'b-o', label='Train Loss (MSE)')
    ax1_mae = ax1.twinx()
    ax1_mae.plot(epochs, train_maes, 'r-s', label='Train MAE')
    # Use min of lengths for test_maes since it might be shorter if training was interrupted
    n_test = min(len(epochs), len(test_maes))
    ax1_mae.plot(epochs[:n_test], test_maes[:n_test], 'g-^', label='Val MAE')
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)', color='b')
    ax1_mae.set_ylabel('MAE', color='r')
    ax1.set_title('Training Process: Loss and MAE')
    
    # Combine legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_mae.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper right')
    
    # Plot 2: Parity plot (Actual vs Predicted)
    min_val = min(min(actuals), min(predicts))
    max_val = max(max(actuals), max(predicts))
    ax2.scatter(actuals, predicts, alpha=0.5, color='blue', edgecolors='k')
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    ax2.set_xlabel('Actual Shear Modulus (GPa)')
    ax2.set_ylabel('Predicted Shear Modulus (GPa)')
    ax2.set_title(f'Model Performance (Test Set)\n$R^2 = {r2:.4f}$')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('model_performance.png', dpi=300)
    print("Results plot saved as 'model_performance.png'")
    
    # Final metrics
    print("-" * 30)
    print(f"Final Test MAE: {test_maes[-1]:.4f}")
    print(f"Final Test R2: {r2:.4f}")
    print("-" * 30)

if __name__ == "__main__":
    plot_results()
