
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

# Add CGCNN path to reuse its data loading if needed, or implement a simple one
CGCNN_PATH = "/data/home/5240019/class_AI4Bat/cgcnn"
sys.path.append(CGCNN_PATH)
from cgcnn.data import CIFData, collate_pool, get_train_val_test_loader

class SimpleGNNLayer(nn.Module):
    """
    A simple message passing layer for crystal graphs.
    h_i' = h_i + sum_j ( sigmoid(W_1 [h_i, h_j, e_ij]) * softplus(W_2 [h_i, h_j, e_ij]) )
    """
    def __init__(self, atom_fea_len, nbr_fea_len):
        super(SimpleGNNLayer, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.fc_full = nn.Linear(2 * atom_fea_len + nbr_fea_len, 2 * atom_fea_len)
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(2 * atom_fea_len)
        self.bn2 = nn.BatchNorm1d(atom_fea_len)

    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        N, M = nbr_fea_idx.shape
        # atom_in_fea: [N, atom_fea_len]
        # nbr_fea: [N, M, nbr_fea_len]
        # nbr_fea_idx: [N, M]
        
        # Get neighbor features
        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]
        # Concatenate: [N, M, 2*atom_fea_len + nbr_fea_len]
        total_nbr_fea = torch.cat([
            atom_in_fea.unsqueeze(1).expand(N, M, self.atom_fea_len),
            atom_nbr_fea,
            nbr_fea
        ], dim=2)
        
        total_gated_fea = self.fc_full(total_nbr_fea.view(-1, 2 * self.atom_fea_len + self.nbr_fea_len))
        total_gated_fea = self.bn1(total_gated_fea).view(N, M, 2 * self.atom_fea_len)
        
        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus(nbr_core)
        
        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)
        nbr_sumed = self.bn2(nbr_sumed)
        out = self.softplus(atom_in_fea + nbr_sumed)
        return out

class SimpleCrystalGNN(nn.Module):
    def __init__(self, orig_atom_fea_len, nbr_fea_len, atom_fea_len=64, n_conv=3, h_fea_len=128, n_h=1):
        super(SimpleCrystalGNN, self).__init__()
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.convs = nn.ModuleList([SimpleGNNLayer(atom_fea_len, nbr_fea_len) for _ in range(n_conv)])
        self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        self.fcs = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len) for _ in range(n_h-1)])
        self.fc_out = nn.Linear(h_fea_len, 1)
        self.softplus = nn.Softplus()
        self.dropout = nn.Dropout(p=0.3)  

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx):
        # Initial embedding
        atom_fea = self.embedding(atom_fea)
        # Message passing
        for conv in self.convs:
            atom_fea = conv(atom_fea, nbr_fea, nbr_fea_idx)
        # Pooling (average pooling for each crystal)
        crys_fea = []
        for idx in crystal_atom_idx:
            crys_fea.append(torch.mean(atom_fea[idx], dim=0, keepdim=True))
        crys_fea = torch.cat(crys_fea, dim=0)
        # Prediction
        crys_fea = self.softplus(self.conv_to_fc(crys_fea))
        crys_fea = self.dropout(crys_fea)  
        for fc in self.fcs:
            crys_fea = self.softplus(fc(crys_fea))
            crys_fea = self.dropout(crys_fea)
        out = self.fc_out(crys_fea)
        return out

# Training helper classes (AverageMeter, Normalizer, mae) - re-implementing briefly
class Normalizer(object):
    def __init__(self, tensor):
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)
    def norm(self, tensor):
        return (tensor - self.mean) / self.std
    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

def train(model, loader, criterion, optimizer, normalizer, device):
    model.train()
    total_loss = 0
    for input, target, _ in loader:
        atom_fea, nbr_fea, nbr_fea_idx = input[0].to(device), input[1].to(device), input[2].to(device)
        crystal_atom_idx = [idx.to(device) for idx in input[3]]
        target = target.to(device)
        
        target_normed = normalizer.norm(target)
        optimizer.zero_grad()
        output = model(atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)
        loss = criterion(output, target_normed)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * target.size(0)
    return total_loss / len(loader.dataset)

def evaluate(model, loader, normalizer, device, return_preds=False):
    model.eval()
    mae_sum = 0
    actuals = []
    predicts = []
    with torch.no_grad():
        for input, target, _ in loader:
            atom_fea, nbr_fea, nbr_fea_idx = input[0].to(device), input[1].to(device), input[2].to(device)
            crystal_atom_idx = [idx.to(device) for idx in input[3]]
            target = target.to(device)
            
            output = model(atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)
            prediction = normalizer.denorm(output)
            mae_sum += torch.sum(torch.abs(prediction - target)).item()
            if return_preds:
                actuals.extend(target.view(-1).tolist())
                predicts.extend(prediction.view(-1).tolist())
    if return_preds:
        return mae_sum / len(loader.dataset), actuals, predicts
    return mae_sum / len(loader.dataset)

def main():
    data_path = "/data/home/5240019/class_AI4Bat/Case_3/data/jarvis_dft_3d"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    dataset = CIFData(data_path)
    train_loader, val_loader, test_loader = get_train_val_test_loader(
        dataset, collate_fn=collate_pool, batch_size=32, 
        train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, 
        return_test=True, train_size=None, val_size=None, test_size=None
    )
    
    # Normalizer
    sample_target = torch.Tensor([dataset[i][1] for i in range(min(len(dataset), 500))])
    normalizer = Normalizer(sample_target)
    
    # Build model
    structures, _, _ = dataset[0]
    orig_atom_fea_len = structures[0].shape[-1]
    nbr_fea_len = structures[1].shape[-1]
    model = SimpleCrystalGNN(orig_atom_fea_len, nbr_fea_len).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    # Train
    print("Starting training from scratch...")
    best_val_mae = 1e10
    for epoch in range(20):
        train_loss = train(model, train_loader, criterion, optimizer, normalizer, device)
        val_mae = evaluate(model, val_loader, normalizer, device)
        scheduler.step(val_mae)
        print(f"Epoch {epoch+1}/20: Train Loss={train_loss:.4f}, Val MAE={val_mae:.4f}, LR={optimizer.param_groups[0]['lr']:.6f}")
        
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), "simple_gnn_best.pth")
            print(f"  Best model saved at epoch {epoch+1}")
    
    # Final test using the best model
    print("Loading best model for final testing...")
    model.load_state_dict(torch.load("simple_gnn_best.pth"))
    test_mae, actuals, predicts = evaluate(model, test_loader, normalizer, device, return_preds=True)
    print(f"Final Test MAE: {test_mae:.4f}")
    
    # Save test results for plotting
    import csv
    with open('simple_gnn_test_results.csv', 'w') as f:
        writer = csv.writer(f)
        for a, p in zip(actuals, predicts):
            writer.writerow([a, p])
    
    # Save model
    torch.save(model.state_dict(), "simple_gnn_jarvis.pth")
    print("Model saved to simple_gnn_jarvis.pth")

if __name__ == "__main__":
    main()
