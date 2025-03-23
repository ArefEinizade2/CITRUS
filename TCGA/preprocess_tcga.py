import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import sparse
from scipy.spatial.distance import pdist, squareform
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import traceback
from tqdm import tqdm
import time
import psutil

# Configuration
DATA_DIR = "data/TCGA"
OUTPUT_DIR = "data/TCGA/processed"
CANCER_TYPE = "BRCA"
TEST_SIZE = 0.15
VAL_SIZE = 0.15
# Max number of genes to use (set to None to use all)
MAX_GENES = 5000  # Limit number of genes to make computation feasible
# Max number of patients for graph construction (None to use all)
MAX_PATIENTS = None

def read_gene_expression(file_path):
    """Read gene expression data from file"""
    print(f"Reading gene expression data from {file_path}")
    
    try:
        # The file has a special format. Read the header first to get column names
        with open(file_path, 'r') as f:
            # Skip comment line
            line = f.readline()
            while line.startswith('#'):
                line = f.readline()
            header = line.strip().split('\t')
        
        print(f"Found headers: {header}")
        
        # Now read the file with pandas, skipping rows with non-gene data
        df = pd.read_csv(file_path, sep='\t', comment='#', header=None, names=header, skiprows=1)
        
        print(f"Initial dataframe shape: {df.shape}")
        
        # Keep only rows that start with ENSG (gene IDs)
        df = df[df[header[0]].str.startswith('ENSG', na=False)]
        
        print(f"After filtering ENSG genes: {df.shape}")
        
        # Extract gene_id and gene_name columns
        gene_id_col = header[0]  # First column should be gene_id
        gene_name_col = header[1]  # Second column should be gene_name
        expression_col = header[3]  # Fourth column should be unstranded counts
        
        # Select relevant columns
        df = df[[gene_id_col, gene_name_col, expression_col]]
        df = df.set_index(gene_id_col)
        
        print(f"Columns used: gene_id={gene_id_col}, gene_name={gene_name_col}, expression={expression_col}")
        print(f"After selecting columns: {df.shape}")
        
        # Extract sample ID from filename
        sample_id = os.path.basename(file_path).split('.')[0]
        
        # Rename the expression column to the sample ID
        df = df.rename(columns={expression_col: sample_id})
        
        print(f"Final dataframe shape: {df.shape}")
        print(f"Sample columns: {df.columns.tolist()}")
        
        return df
    except Exception as e:
        print(f"Error reading gene expression file: {e}")
        traceback.print_exc()
        raise e

def combine_gene_expression_files(file_paths):
    """Combine multiple gene expression files into a single matrix"""
    print(f"Combining {len(file_paths)} gene expression files")
    
    try:
        all_dfs = []
        for file_path in tqdm(file_paths, desc="Reading gene expression files"):
            df = read_gene_expression(file_path)
            all_dfs.append(df)
            print(f"Added file with shape: {df.shape}")
        
        # Merge all dataframes on gene_id
        combined_df = pd.concat(all_dfs, axis=1)
        print(f"After concatenation: {combined_df.shape}")
        
        # If there are duplicate columns, keep the first occurrence
        combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]
        print(f"After removing duplicates: {combined_df.shape}")
        
        # Ensure gene_name is in the final dataframe
        # It might be duplicated across files, so we need to ensure it's only present once
        gene_name_cols = [col for col in combined_df.columns if col == 'gene_name']
        if gene_name_cols:
            gene_name_col = gene_name_cols[0]
            # Remove all gene_name columns except the first one
            for col in gene_name_cols[1:]:
                combined_df = combined_df.drop(columns=[col])
            
            # Move gene_name to the first column
            cols = combined_df.columns.tolist()
            cols.remove(gene_name_col)
            cols.insert(0, gene_name_col)
            combined_df = combined_df[cols]
        
        print(f"Final combined dataframe shape: {combined_df.shape}")
        print(f"Final columns: {combined_df.columns.tolist()}")
        
        return combined_df
    except Exception as e:
        print(f"Error combining gene expression files: {e}")
        traceback.print_exc()
        raise e

def read_mirna_expression(file_path):
    """Read miRNA expression data from file"""
    print(f"Reading miRNA expression data from {file_path}")
    df = pd.read_csv(file_path, sep='\t', index_col=0)
    return df

def read_methylation(file_path):
    """Read methylation data from file"""
    print(f"Reading methylation data from {file_path}")
    df = pd.read_csv(file_path, sep='\t', index_col=0)
    return df

def create_similarity_graph(data_matrix, n_neighbors=10, threshold=None, batch_size=1000):
    """Create a similarity graph based on correlation or euclidean distance
    
    Uses batched computation to reduce memory usage for large matrices.
    """
    print(f"Creating similarity graph for matrix of shape {data_matrix.shape} with {n_neighbors} neighbors")
    start_time = time.time()
    
    try:
        n_samples = data_matrix.shape[0]
        
        # Create graph
        G = nx.Graph()
        
        # Add nodes
        for i in range(n_samples):
            G.add_node(i)
        
        # Process correlation matrix in batches to reduce memory usage
        if threshold:
            print("Using correlation threshold method")
            # We need to compute correlation and add edges based on threshold
            n_batches = max(1, n_samples // batch_size)
            with tqdm(total=n_batches, desc="Building graph (threshold)") as pbar:
                for i in range(0, n_samples, batch_size):
                    batch_end = min(i + batch_size, n_samples)
                    # Compute correlation of this batch with all other samples
                    batch_corr = np.corrcoef(data_matrix[i:batch_end], data_matrix)[0:batch_end-i, batch_end:]
                    
                    # Add edges where correlation exceeds threshold
                    for bi, global_i in enumerate(range(i, batch_end)):
                        for global_j in range(n_samples):
                            if global_j <= global_i:  # Skip lower triangle (already processed) and self-correlations
                                continue
                            corr_idx = global_j - batch_end if global_j >= batch_end else bi
                            if batch_corr[bi, corr_idx] > threshold:
                                G.add_edge(global_i, global_j, weight=batch_corr[bi, corr_idx])
                    
                    pbar.update(1)
        else:
            print("Using k-nearest neighbors method")
            # Compute k-nearest neighbors using batch processing
            with tqdm(total=n_samples, desc="Building graph (knn)") as pbar:
                for i in range(n_samples):
                    # Compute correlation of this sample with all others
                    corr_vec = np.array([np.corrcoef(data_matrix[i], data_matrix[j])[0, 1] 
                                        for j in range(n_samples)])
                    corr_vec[i] = -np.inf  # Exclude self
                    
                    # Get indices of k highest correlations
                    indices = np.argsort(corr_vec)[-n_neighbors:]
                    
                    # Add edges to k-nearest neighbors
                    for j in indices:
                        G.add_edge(i, j, weight=corr_vec[j])
                    
                    pbar.update(1)
                    
                    # Print progress every 1000 nodes
                    if i % 1000 == 0 and i > 0:
                        elapsed = time.time() - start_time
                        estimated_total = elapsed * n_samples / i
                        print(f"Progress: {i}/{n_samples} nodes processed ({i/n_samples*100:.1f}%), "
                              f"Time: {elapsed/60:.1f} min, ETA: {(estimated_total-elapsed)/60:.1f} min")
        
        print(f"Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        # Ensure graph is connected
        if not nx.is_connected(G):
            print("Graph is not connected, connecting components...")
            components = list(nx.connected_components(G))
            print(f"Found {len(components)} connected components")
            
            # Compute a sparse representation of the correlation matrix
            sparse_corr = {}
            
            with tqdm(total=len(components)-1, desc="Connecting components") as pbar:
                for i in range(len(components) - 1):
                    # Find nodes with highest correlation between components
                    comp1 = list(components[i])
                    comp2 = list(components[i+1])
                    max_corr = -np.inf
                    max_i, max_j = -1, -1
                    
                    # Process in a memory-efficient way
                    for i_idx in tqdm(comp1, desc=f"Component {i} connections", leave=False):
                        for j_idx in comp2:
                            # Only compute correlation if we haven't already
                            if (i_idx, j_idx) not in sparse_corr and (j_idx, i_idx) not in sparse_corr:
                                corr = np.corrcoef(data_matrix[i_idx], data_matrix[j_idx])[0, 1]
                                sparse_corr[(i_idx, j_idx)] = corr
                            else:
                                corr = sparse_corr.get((i_idx, j_idx), sparse_corr.get((j_idx, i_idx)))
                                
                            if corr > max_corr:
                                max_corr = corr
                                max_i, max_j = i_idx, j_idx
                    
                    G.add_edge(max_i, max_j, weight=max_corr)
                    pbar.update(1)
                
            print(f"After connecting components: {G.number_of_edges()} edges")
        
        return G
    except Exception as e:
        print(f"Error creating similarity graph: {e}")
        traceback.print_exc()
        raise e

def get_laplacian_eigenvectors(G, k, use_approx=True, max_nodes=5000):
    """Compute the Laplacian eigenvectors of a graph
    
    Args:
        G: networkx graph
        k: number of eigenvectors to compute
        use_approx: whether to use Lanczos algorithm (faster for large graphs)
        max_nodes: maximum number of nodes to use (subsample if larger)
    """
    n_nodes = G.number_of_nodes()
    print(f"Computing Laplacian eigenvectors for graph with {n_nodes} nodes, k={k}")
    
    # Report memory usage
    memory_info = psutil.Process().memory_info()
    print(f"Memory usage: {memory_info.rss / (1024 * 1024):.1f} MB")
    
    start_time = time.time()
    
    try:
        # For very large graphs, subsample nodes to make computation feasible
        if max_nodes and n_nodes > max_nodes:
            print(f"Graph is very large ({n_nodes} nodes). Subsampling to {max_nodes} nodes...")
            # Select a random subset of nodes
            node_subset = sorted(np.random.choice(n_nodes, max_nodes, replace=False))
            # Create subgraph
            G_sub = G.subgraph(node_subset)
            # Create a mapping from subgraph nodes to original nodes
            node_mapping = {i: node for i, node in enumerate(node_subset)}
            print(f"Created subgraph with {G_sub.number_of_nodes()} nodes and {G_sub.number_of_edges()} edges")
            G = G_sub
        else:
            node_mapping = None  # No subsampling
        
        # Get adjacency matrix - use different method based on graph size
        print("Computing adjacency matrix...")
        if n_nodes < 1000:
            # For smaller graphs, dense representation is fine
            A = nx.to_numpy_array(G)
            # Get node degrees
            degrees = np.sum(A, axis=1)
            
            # Compute Laplacian directly
            print("Computing Laplacian matrix...")
            D = np.diag(degrees)
            L = D - A
            
            # Compute normalized Laplacian
            print("Computing normalized Laplacian...")
            with np.errstate(divide='ignore', invalid='ignore'):
                D_sqrt_inv = np.diag(1.0 / np.sqrt(np.maximum(degrees, 1e-10)))
            L_normalized = D_sqrt_inv @ L @ D_sqrt_inv
            
            # Convert to sparse for eigendecomposition (more efficient)
            L_normalized_sparse = sparse.csr_matrix(L_normalized)
        else:
            # For larger graphs, use sparse representation
            try:
                A = nx.to_scipy_sparse_array(G, format='csr')
                
                # Get node degrees
                degrees = A.sum(axis=1)
                # Convert to flat array - handle both sparse and dense arrays
                if hasattr(degrees, 'A1'):
                    # Sparse matrix
                    degrees = degrees.A1
                elif isinstance(degrees, np.matrix):
                    # numpy matrix
                    degrees = np.asarray(degrees).flatten()
                else:
                    # Already a flat array
                    degrees = np.asarray(degrees).flatten()
                
                # Compute Laplacian using sparse matrices
                print("Computing Laplacian matrix...")
                D = sparse.diags(degrees)
                L = D - A
                
                # Compute normalized Laplacian
                print("Computing normalized Laplacian...")
                D_sqrt_inv = sparse.diags(1.0 / np.sqrt(np.maximum(degrees, 1e-10)))
                L_normalized_sparse = D_sqrt_inv @ L @ D_sqrt_inv
            except Exception as e:
                print(f"Error with sparse computation, falling back to dense: {e}")
                # Fallback to dense representation
                A = nx.to_numpy_array(G)
                degrees = np.sum(A, axis=1)
                D = np.diag(degrees)
                L = D - A
                with np.errstate(divide='ignore', invalid='ignore'):
                    D_sqrt_inv = np.diag(1.0 / np.sqrt(np.maximum(degrees, 1e-10)))
                L_normalized = D_sqrt_inv @ L @ D_sqrt_inv
                L_normalized_sparse = sparse.csr_matrix(L_normalized)
        
        # Compute eigendecomposition
        print(f"Computing {k} eigenvectors using {'Lanczos algorithm' if use_approx else 'standard method'}...")
        if use_approx:
            # Use Lanczos algorithm (faster for large sparse matrices)
            evals, evecs = sparse.linalg.eigsh(L_normalized_sparse, k=k, which='SM', tol=1e-4)
        else:
            # Standard eigendecomposition
            evals, evecs = sparse.linalg.eigs(L_normalized_sparse, k=k, which='SM', return_eigenvectors=True)
        
        # Convert to PyTorch tensors (real part only)
        evals = torch.tensor(evals.real).to(torch.float32)
        evecs = torch.tensor(evecs.real).to(torch.float32)
        
        elapsed = time.time() - start_time
        print(f"Computed {len(evals)} eigenvalues and eigenvectors of shape {evecs.shape} in {elapsed/60:.1f} minutes")
        
        # If we subsampled, we need to adjust the eigenvectors
        if node_mapping:
            print("Adjusting eigenvectors to account for subsampling...")
            # Create a mapping tensor to map back to full space
            full_evecs = torch.zeros((n_nodes, k), dtype=torch.float32)
            for i, node in node_mapping.items():
                full_evecs[node] = evecs[i]
            evecs = full_evecs
        
        return evals, evecs, L_normalized_sparse
    except Exception as e:
        print(f"Error computing Laplacian eigenvectors: {e}")
        traceback.print_exc()
        raise e

def select_significant_genes(df, n_top=5000):
    """Select most informative genes based on variance across samples"""
    print(f"Selecting top {n_top} most variable genes from {df.shape[0]} genes")
    
    # Calculate variance for each gene across samples
    gene_variance = df.iloc[:, 1:].var(axis=1)  # Skip gene_name column
    
    # Sort genes by variance and select top n
    top_genes = gene_variance.sort_values(ascending=False).head(n_top).index
    
    # Filter dataframe to keep only top genes
    filtered_df = df.loc[top_genes]
    
    print(f"Selected {len(top_genes)} genes with highest variance")
    print(f"Filtered dataframe shape: {filtered_df.shape}")
    
    return filtered_df

def prepare_data_for_citrus():
    """Prepare multi-omics data for CITRUS model"""
    try:
        start_time = time.time()
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Read metadata file
        metadata_file = os.path.join(DATA_DIR, "files_metadata.csv")
        if not os.path.exists(metadata_file):
            raise FileNotFoundError("Metadata file not found. Run download_tcga.py first.")
        
        metadata = pd.read_csv(metadata_file)
        print(f"Loaded metadata with {len(metadata)} files")
        
        # Separate files by data type
        gene_exp_files = metadata[metadata['data_type'] == 'Gene Expression Quantification']['local_path'].tolist()
        mirna_files = metadata[metadata['data_type'] == 'miRNA Expression Quantification']['local_path'].tolist()
        methylation_files = metadata[metadata['data_type'] == 'Methylation Beta Value']['local_path'].tolist()
        
        print(f"Found: {len(gene_exp_files)} gene expression files, {len(mirna_files)} miRNA files, {len(methylation_files)} methylation files")
        
        # Combine gene expression data from multiple files
        if gene_exp_files:
            gene_exp_data = combine_gene_expression_files(gene_exp_files)
        else:
            raise ValueError("No gene expression files found")
        
        # Convert the gene expression data to numeric (might be strings from file reading)
        print("Converting gene expression data to numeric values...")
        for col in gene_exp_data.columns:
            if col != 'gene_name':
                gene_exp_data[col] = pd.to_numeric(gene_exp_data[col], errors='coerce')
        
        # Drop rows with missing values
        gene_exp_data = gene_exp_data.dropna()
        print(f"After dropping NAs: {gene_exp_data.shape}")
        
        # Select most informative genes if MAX_GENES is set
        if MAX_GENES and gene_exp_data.shape[0] > MAX_GENES:
            gene_exp_data = select_significant_genes(gene_exp_data, MAX_GENES)
        
        if mirna_files:
            mirna_data = read_mirna_expression(mirna_files[0])
        else:
            mirna_data = None
            print("No miRNA expression files found")
        
        if methylation_files:
            methylation_data = read_methylation(methylation_files[0])
        else:
            methylation_data = None
            print("No methylation files found")
        
        # Extract patient and gene information
        patients = gene_exp_data.columns[1:]  # Skip gene_name column
        genes = gene_exp_data.index.tolist()
        
        print(f"Raw data shape: Patients: {len(patients)}, Genes: {len(genes)}")
        
        # Limit number of patients if MAX_PATIENTS is set
        if MAX_PATIENTS and len(patients) > MAX_PATIENTS:
            print(f"Limiting to {MAX_PATIENTS} patients for analysis...")
            selected_patients = patients[:MAX_PATIENTS]
            gene_exp_data = gene_exp_data[['gene_name'] + list(selected_patients)]
            patients = selected_patients
            print(f"Using {len(patients)} patients")
        
        # Prepare feature matrix (patients × genes)
        X = gene_exp_data.iloc[:, 1:].values.T  # Transpose to get (patients × genes)
        
        print(f"Feature matrix shape: {X.shape}")
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        print(f"Standardized feature matrix")
        
        # For demo purposes, create a simple binary classification task
        # Here we arbitrarily split patients into two groups based on the median expression of a random gene
        random_gene_idx = np.random.randint(0, X.shape[1])
        print(f"Using random gene index {random_gene_idx} for classification task")
        
        # Use numpy array directly
        feature_values = X[:, random_gene_idx]
        threshold = np.median(feature_values)
        target_values = (feature_values > threshold).astype(int)
        print(f"Target distribution: {np.bincount(target_values)}")
        
        Y = np.zeros((len(patients), 2))
        Y[np.arange(len(patients)), target_values] = 1  # One-hot encoding
        
        print(f"Created binary classification task")
        
        # Create the two factor graphs
        # 1. Patient similarity graph based on gene expression
        n_neighbors_patient = min(3, X_scaled.shape[0]-1)  # Reduce neighbors for small datasets
        print(f"Using {n_neighbors_patient} neighbors for patient graph")
        patient_graph = create_similarity_graph(X_scaled, n_neighbors=n_neighbors_patient)
        
        # 2. Gene similarity graph based on expression patterns across patients
        n_neighbors_gene = min(10, X_scaled.shape[1]-1)
        print(f"Using {n_neighbors_gene} neighbors for gene graph")
        gene_graph = create_similarity_graph(X_scaled.T, n_neighbors=n_neighbors_gene)
        
        # Get Laplacian eigenvectors for both graphs
        k_patient = min(2, len(patients) - 1)  # Reduce k for small datasets
        k_gene = min(20, len(genes) - 1)
        
        print(f"Computing Laplacian eigenvectors with k_patient={k_patient}, k_gene={k_gene}")
        
        # Use approximation for large graphs
        patient_evals, patient_evecs, patient_L = get_laplacian_eigenvectors(
            patient_graph, k_patient, use_approx=True)
        gene_evals, gene_evecs, gene_L = get_laplacian_eigenvectors(
            gene_graph, k_gene, use_approx=True, max_nodes=5000)
        
        # Package the data for CITRUS
        # Convert to PyTorch tensors
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        Y_tensor = torch.tensor(Y, dtype=torch.float32)
        
        print(f"Created PyTorch tensors for features and targets")
        
        # Split into train/val/test sets - ensure we have at least 1 sample in each set
        if len(patients) < 5:
            # For very small datasets, use all data for everything
            train_idx = val_idx = test_idx = np.arange(len(patients))
        else:
            # Regular split
            train_idx, test_idx = train_test_split(np.arange(len(patients)), test_size=min(TEST_SIZE, 0.4), random_state=42)
            if len(train_idx) > 2:
                train_idx, val_idx = train_test_split(train_idx, test_size=min(VAL_SIZE/(1-TEST_SIZE), 0.5), random_state=42)
            else:
                val_idx = train_idx  # Use training data for validation if too few samples
        
        train_idx = torch.tensor(train_idx, dtype=torch.long)
        val_idx = torch.tensor(val_idx, dtype=torch.long)
        test_idx = torch.tensor(test_idx, dtype=torch.long)
        
        print(f"Split data into train/val/test sets: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")
        
        # Save processed data
        save_path = os.path.join(OUTPUT_DIR, 'citrus_ready_data.pt')
        print(f"Saving processed data to {save_path}")
        
        torch.save({
            'X': X_tensor,
            'Y': Y_tensor,
            'train_idx': train_idx,
            'val_idx': val_idx,
            'test_idx': test_idx,
            'patient_evals': patient_evals,
            'patient_evecs': patient_evecs,
            'gene_evals': gene_evals,
            'gene_evecs': gene_evecs,
            'patient_L': patient_L,
            'gene_L': gene_L,
            'N_list': [len(patients), len(genes)]
        }, save_path)
        
        # Save metadata for reference
        metadata_path = os.path.join(OUTPUT_DIR, 'patient_metadata.csv')
        print(f"Saving patient metadata to {metadata_path}")
        
        pd.DataFrame({
            'patient_id': list(patients),
            'target': target_values
        }).to_csv(metadata_path, index=False)
        
        elapsed = time.time() - start_time
        print(f"Data processed and saved to {OUTPUT_DIR}/citrus_ready_data.pt")
        print(f"Number of patients: {len(patients)}")
        print(f"Number of genes: {len(genes)}")
        print(f"X shape: {X_tensor.shape}")
        print(f"Y shape: {Y_tensor.shape}")
        print(f"Train/Val/Test split: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")
        print(f"Total processing time: {elapsed/60:.1f} minutes")
    except Exception as e:
        print(f"Error in prepare_data_for_citrus: {e}")
        traceback.print_exc()
        raise e

if __name__ == "__main__":
    prepare_data_for_citrus() 