import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import time
from torch_geometric.nn import GCNConv
import sys
import json
from scipy import sparse
from Utils.layers import CITRUS  # Import CITRUS from Utils.layers

# Set GPU device explicitly to use GPU 1 (which has more free memory)
if torch.cuda.is_available():
    torch.cuda.set_device(1)  # Using GPU 1 instead of default GPU 0
    print(f"Using GPU: {torch.cuda.get_device_name(1)}")

# Configuration
DATA_DIR = "data/TCGA/processed"
OUTPUT_DIR = "data/TCGA/results"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 100
LEARNING_RATE = 1e-3
HIDDEN_DIM = 64
WEIGHT_DECAY = 1e-5
# Configuration for CITRUS
CITRUS_WIDTH = 64
N_BLOCKS = 4

class SimpleGCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(SimpleGCN, self).__init__()
        self.conv1 = GCNConv(in_features, hidden_features)
        self.conv2 = GCNConv(hidden_features, hidden_features)
        self.fc = nn.Linear(hidden_features, out_features)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.fc(x)
        return x

class SimpleMLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, hidden_features)
        self.fc3 = nn.Linear(hidden_features, out_features)
        
    def forward(self, x, *args):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train(model, optimizer, data, target, train_idx, edge_index, criterion):
    model.train()
    optimizer.zero_grad()
    
    # Apply the model (either GCN or MLP)
    if isinstance(model, SimpleMLP):
        out = model(data)
    else:  # GCN
        out = model(data, edge_index)
    
    # Evaluate loss
    loss = criterion(out[train_idx], target[train_idx])
    
    loss.backward()
    optimizer.step()
    
    return loss.item()

@torch.no_grad()
def evaluate(model, data, target, idx, edge_index, criterion):
    model.eval()
    
    if isinstance(model, SimpleMLP):
        out = model(data)
    else:  # GCN
        out = model(data, edge_index)
    
    loss = criterion(out[idx], target[idx])
    
    # Calculate additional metrics for classification
    pred_probs = torch.softmax(out[idx], dim=1).cpu().numpy()
    pred_labels = np.argmax(pred_probs, axis=1)
    true_labels = np.argmax(target[idx].cpu().numpy(), axis=1)
    
    # Check if all samples have same label (single class)
    n_classes = len(np.unique(true_labels))
    
    accuracy = accuracy_score(true_labels, pred_labels)
    
    # Handle case with only one class
    if n_classes < 2:
        precision = 1.0 if np.all(pred_labels == true_labels) else 0.0
        recall = 1.0 if np.all(pred_labels == true_labels) else 0.0
        f1 = 1.0 if np.all(pred_labels == true_labels) else 0.0
        auc = 0.5  # AUC is undefined for single class, use 0.5 as default
    else:
        precision = precision_score(true_labels, pred_labels, average='macro')
        recall = recall_score(true_labels, pred_labels, average='macro')
        f1 = f1_score(true_labels, pred_labels, average='macro')
        
        # Handle AUC calculation
        try:
            auc = roc_auc_score(target[idx].cpu().numpy(), pred_probs, average='macro')
        except ValueError:
            # If AUC calculation fails, use 0.5 as default
            auc = 0.5
            
    return loss.item(), accuracy, precision, recall, f1, auc

def train_and_evaluate():
    """Main function to train and evaluate models on TCGA data"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load preprocessed data
    data_file = os.path.join(DATA_DIR, 'citrus_ready_data.pt')
    if not os.path.exists(data_file):
        raise FileNotFoundError("Processed data not found. Run preprocess_tcga.py first.")
    
    data = torch.load(data_file)
    
    # Extract data
    X = data['X'].to(DEVICE)
    Y = data['Y'].to(DEVICE)
    train_idx = data['train_idx'].to(DEVICE)
    val_idx = data['val_idx'].to(DEVICE)
    test_idx = data['test_idx'].to(DEVICE)
    patient_L = data['patient_L']
    gene_L = data['gene_L']
    patient_evals = data['patient_evals'].to(DEVICE)
    patient_evecs = data['patient_evecs'].to(DEVICE)
    gene_evals = data['gene_evals'].to(DEVICE)
    gene_evecs = data['gene_evecs'].to(DEVICE)
    N_list = data['N_list']  # [num_patients, num_genes]
    
    print(f"X shape: {X.shape}")
    print(f"Y shape: {Y.shape}")
    print(f"Training samples: {len(train_idx)}")
    print(f"Validation samples: {len(val_idx)}")
    print(f"Test samples: {len(test_idx)}")
    print(f"N_list (patients, genes): {N_list}")
    
    # Create edge_index for GCN
    from torch_geometric.utils import from_scipy_sparse_matrix
    edge_index, _ = from_scipy_sparse_matrix(patient_L)
    edge_index = edge_index.to(DEVICE)
    
    # Model configurations
    in_features = X.shape[1]
    out_features = Y.shape[1]
    criterion = nn.CrossEntropyLoss()
    
    # Initialize models 
    gcn_model = SimpleGCN(
        in_features=in_features,
        hidden_features=HIDDEN_DIM,
        out_features=out_features
    ).to(DEVICE)
    
    mlp_model = SimpleMLP(
        in_features=in_features,
        hidden_features=HIDDEN_DIM,
        out_features=out_features
    ).to(DEVICE)
    
    # Initialize CITRUS model
    # Calculate the product of eigenvalues' dimensions for the k parameter
    k_patient = patient_evecs.shape[1]  # Number of eigenvectors for patient graph
    k_gene = gene_evecs.shape[1]  # Number of eigenvectors for gene graph
    k_total = k_patient * k_gene  # Total number for the tensor product graph
    
    print(f"CITRUS input dimensions - k_patient: {k_patient}, k_gene: {k_gene}, k_total: {k_total}")
    
    # Prepare combined eigenvectors using Kronecker product
    # This will be used to efficiently compute on the tensor product graph
    L_normalized_sparse_list = [patient_L, gene_L]
    k_list = [k_patient, k_gene]
    
    # Get mass for the Laplacian eigenvectors (uniform weights for all nodes)
    mass = torch.ones(np.prod(N_list)).to(DEVICE)
    
    # Combine the eigenvectors using Kronecker product
    evecs_kron = torch.kron(patient_evecs, gene_evecs).to(DEVICE)
    evals_list = [patient_evals, gene_evals]
    
    citrus_model = CITRUS(
        k=k_total,
        C_in=in_features,
        C_out=out_features,
        C_width=CITRUS_WIDTH,
        N_block=N_BLOCKS,
        num_nodes=N_list,
        last_activation=lambda x: x,
        diffusion_method='spectral',
        with_MLP=True,
        dropout=True,
        device=DEVICE
    ).to(DEVICE)
    
    # New train function for CITRUS
    def train_citrus(model, optimizer, data, target, train_idx, criterion, epoch, mass, L_list, evals_list, evecs):
        model.train()
        optimizer.zero_grad()
        
        # Apply the CITRUS model
        out = model(epoch, data, [], mass=mass, L=L_list, evals=evals_list, evecs=evecs)
        
        # Evaluate loss
        loss = criterion(out[train_idx], target[train_idx])
        
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    # New evaluate function for CITRUS
    @torch.no_grad()
    def evaluate_citrus(model, data, target, idx, criterion, epoch, mass, L_list, evals_list, evecs):
        model.eval()
        
        out = model(epoch, data, [], mass=mass, L=L_list, evals=evals_list, evecs=evecs)
        
        loss = criterion(out[idx], target[idx])
        
        # Calculate additional metrics for classification
        pred_probs = torch.softmax(out[idx], dim=1).cpu().numpy()
        pred_labels = np.argmax(pred_probs, axis=1)
        true_labels = np.argmax(target[idx].cpu().numpy(), axis=1)
        
        # Check if all samples have same label (single class)
        n_classes = len(np.unique(true_labels))
        
        accuracy = accuracy_score(true_labels, pred_labels)
        
        # Handle case with only one class
        if n_classes < 2:
            precision = 1.0 if np.all(pred_labels == true_labels) else 0.0
            recall = 1.0 if np.all(pred_labels == true_labels) else 0.0
            f1 = 1.0 if np.all(pred_labels == true_labels) else 0.0
            auc = 0.5  # AUC is undefined for single class, use 0.5 as default
        else:
            precision = precision_score(true_labels, pred_labels, average='macro')
            recall = recall_score(true_labels, pred_labels, average='macro')
            f1 = f1_score(true_labels, pred_labels, average='macro')
            
            # Handle AUC calculation
            try:
                auc = roc_auc_score(target[idx].cpu().numpy(), pred_probs, average='macro')
            except ValueError:
                # If AUC calculation fails, use 0.5 as default
                auc = 0.5
                
        return loss.item(), accuracy, precision, recall, f1, auc
    
    # Optimizers
    gcn_optimizer = optim.Adam(gcn_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    mlp_optimizer = optim.Adam(mlp_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    citrus_optimizer = optim.Adam(citrus_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Training loop
    models = {
        'GCN': (gcn_model, gcn_optimizer, False),  # False indicates not a CITRUS model
        'MLP': (mlp_model, mlp_optimizer, False),
        'CITRUS': (citrus_model, citrus_optimizer, True)  # True indicates CITRUS model
    }
    
    results = {name: {'train_loss': [], 'val_loss': [], 'test_loss': [], 
                     'train_acc': [], 'val_acc': [], 'test_acc': [],
                     'train_auc': [], 'val_auc': [], 'test_auc': []} 
              for name in models.keys()}
    
    best_val_metrics = {name: {'loss': float('inf'), 'acc': 0, 'epoch': 0} for name in models.keys()}
    
    for name, (model, optimizer, is_citrus) in models.items():
        print(f"\nTraining {name} model...")
        t_start = time.time()
        
        for epoch in range(1, EPOCHS + 1):
            # Train
            if is_citrus:
                train_loss = train_citrus(
                    model, optimizer, X, Y, train_idx, criterion, epoch, 
                    mass, L_normalized_sparse_list, evals_list, evecs_kron
                )
                
                # Evaluate
                train_metrics = evaluate_citrus(
                    model, X, Y, train_idx, criterion, epoch,
                    mass, L_normalized_sparse_list, evals_list, evecs_kron
                )
                val_metrics = evaluate_citrus(
                    model, X, Y, val_idx, criterion, epoch,
                    mass, L_normalized_sparse_list, evals_list, evecs_kron
                )
                test_metrics = evaluate_citrus(
                    model, X, Y, test_idx, criterion, epoch,
                    mass, L_normalized_sparse_list, evals_list, evecs_kron
                )
            else:
                train_loss = train(model, optimizer, X, Y, train_idx, edge_index, criterion)
                
                # Evaluate
                train_metrics = evaluate(model, X, Y, train_idx, edge_index, criterion)
                val_metrics = evaluate(model, X, Y, val_idx, edge_index, criterion)
                test_metrics = evaluate(model, X, Y, test_idx, edge_index, criterion)
            
            # Unpack metrics
            _, train_acc, _, _, _, train_auc = train_metrics
            val_loss, val_acc, _, _, _, val_auc = val_metrics
            test_loss, test_acc, test_precision, test_recall, test_f1, test_auc = test_metrics
            
            # Track results
            results[name]['train_loss'].append(train_loss)
            results[name]['val_loss'].append(val_loss)
            results[name]['test_loss'].append(test_loss)
            results[name]['train_acc'].append(train_acc)
            results[name]['val_acc'].append(val_acc)
            results[name]['test_acc'].append(test_acc)
            results[name]['train_auc'].append(train_auc)
            results[name]['val_auc'].append(val_auc)
            results[name]['test_auc'].append(test_auc)
            
            # Track best model
            if val_acc > best_val_metrics[name]['acc']:
                best_val_metrics[name]['acc'] = val_acc
                best_val_metrics[name]['loss'] = val_loss
                best_val_metrics[name]['epoch'] = epoch
                
                # Save best model
                torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f'best_{name.lower()}_model.pt'))
                
                print(f"Epoch {epoch}: {name} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Test Loss: {test_loss:.4f}")
                print(f"          Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}")
                print(f"          Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}, Test F1: {test_f1:.4f}, Test AUC: {test_auc:.4f}")
        
        t_end = time.time()
        print(f"{name} training completed in {(t_end - t_start)/60:.2f} minutes.")
        print(f"Best validation accuracy: {best_val_metrics[name]['acc']:.4f} at epoch {best_val_metrics[name]['epoch']}")
    
    # Plot results
    plt.figure(figsize=(15, 12))
    
    # Plot losses
    plt.subplot(3, 2, 1)
    for name in models.keys():
        plt.plot(results[name]['train_loss'], label=f'{name} Train')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    
    plt.subplot(3, 2, 2)
    for name in models.keys():
        plt.plot(results[name]['val_loss'], label=f'{name} Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss')
    plt.legend()
    
    # Plot accuracies
    plt.subplot(3, 2, 3)
    for name in models.keys():
        plt.plot(results[name]['train_acc'], label=f'{name} Train')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.legend()
    
    plt.subplot(3, 2, 4)
    for name in models.keys():
        plt.plot(results[name]['val_acc'], label=f'{name} Val')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()
    
    # Plot AUC
    plt.subplot(3, 2, 5)
    for name in models.keys():
        plt.plot(results[name]['train_auc'], label=f'{name} Train')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.title('Training AUC')
    plt.legend()
    
    plt.subplot(3, 2, 6)
    for name in models.keys():
        plt.plot(results[name]['test_auc'], label=f'{name} Test')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.title('Test AUC')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_results.png'))
    
    # Print final comparison
    print("\n----- Final Performance Comparison -----")
    for name in models.keys():
        best_epoch = best_val_metrics[name]['epoch'] - 1  # 0-indexed
        print(f"{name}:")
        print(f"  Test Accuracy: {results[name]['test_acc'][best_epoch]:.4f}")
        print(f"  Test AUC: {results[name]['test_auc'][best_epoch]:.4f}")
    
    # Save results
    with open(os.path.join(OUTPUT_DIR, 'results.json'), 'w') as f:
        # Convert numpy arrays to lists
        for name in results:
            for metric in results[name]:
                results[name][metric] = [float(x) for x in results[name][metric]]
        
        json.dump({
            'results': results,
            'best_val_metrics': best_val_metrics
        }, f, indent=2)

if __name__ == "__main__":
    train_and_evaluate() 