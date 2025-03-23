# TCGA Data Analysis Pipeline with Graph Neural Networks

## Dataset Overview

The Cancer Genome Atlas (TCGA) is a landmark cancer genomics program that molecularly characterized over 20,000 primary cancer and matched normal samples spanning 33 cancer types. This project focuses on the Breast Cancer (BRCA) dataset from TCGA, which includes multi-omics data for breast cancer patients.

### What is TCGA?

TCGA (The Cancer Genome Atlas) is a large-scale collaborative effort launched in 2006 by the National Cancer Institute (NCI) and the National Human Genome Research Institute (NHGRI). Its goal was to create a comprehensive atlas of the genomic changes that occur in different cancer types. By analyzing thousands of tumor samples, TCGA has helped researchers better understand how cancer develops at the molecular level.

## The CITRUS Model

### What is CITRUS?

CITRUS (Coupled Tensor-Matrix Factorization Using the Schur Decomposition) is an advanced graph neural network model specifically designed for multi-modal biological data. It leverages both patient similarity and gene similarity information through tensor product graphs.

### How CITRUS Works in This Pipeline

CITRUS operates on two complementary graph structures:

1. **Patient Similarity Graph**: Captures relationships between patients based on their gene expression profiles
2. **Gene Similarity Graph**: Captures relationships between genes based on their expression patterns across patients

The key innovation of CITRUS is that it works with the tensor product of these two graphs, effectively utilizing all pairwise interactions between patient-gene pairs. This allows the model to:

- Simultaneously model patient similarity and gene similarity
- Capture complex dependencies between genes across patients
- Learn more robust representations by leveraging both data modalities
- Better handle sparse or noisy biological data

In mathematical terms, CITRUS uses the Laplacian eigenvectors of both graphs and combines them using the Kronecker product, enabling efficient computation on the tensor product graph without explicitly constructing it (which would be computationally prohibitive).

## Data Types and Biological Background

The pipeline processes three types of omics data. Below is an explanation of each for readers without a biology background:

### 1. Gene Expression Quantification (RNA-Seq)

#### What is Gene Expression?
Every cell in your body contains the same DNA (your genome), which has instructions for making all possible proteins. However, not all genes are active (expressed) in every cell. Gene expression is the process by which the information encoded in a gene is used to create a functional product, typically a protein.

#### What is RNA-Seq?
RNA sequencing (RNA-Seq) is a technology that measures which genes are active in a cell and how active they are:

- **DNA to RNA**: When a gene is active, the cell makes a temporary copy of that gene in the form of RNA (Ribonucleic Acid)
- **Counting RNA**: RNA-Seq counts how many RNA copies are being made from each gene
- **Higher counts = more active**: The more RNA copies detected, the more active that gene is

RNA-Seq data helps researchers understand which genes are turned on or off in cancer cells compared to normal cells. Each file in our dataset contains expression values for approximately 60,000 genes for a single patient.

### 2. miRNA Expression Quantification

#### What are microRNAs?
MicroRNAs (miRNAs) are tiny RNA molecules (about 22 nucleotides long) that don't code for proteins but instead regulate other genes:

- **Gene Silencers**: miRNAs act like "off switches" for genes by binding to messenger RNAs and preventing them from being translated into proteins
- **Fine-Tuning**: A single miRNA can regulate hundreds of different target genes
- **Cancer Connection**: Abnormal miRNA levels have been linked to various cancers, as they can affect genes involved in cell growth, division, and death

miRNA expression data tells us which of these regulatory molecules are present in cancer cells and in what amounts, providing insight into the regulation of gene networks in cancer.

### 3. Methylation Beta Value

#### What is DNA Methylation?
DNA methylation is a process that adds a chemical mark (a methyl group) to DNA without changing its sequence:

- **Chemical Modification**: Methyl groups (CH₃) are attached to specific sites in the DNA
- **Gene Silencing**: Methylation typically turns genes off by preventing the cellular machinery from accessing the DNA
- **Reversible Changes**: Unlike mutations, methylation changes can be reversed

#### What are Beta Values?
Beta values represent the level of methylation at specific DNA sites:

- **Scale**: Values range from 0 to 1
- **Interpretation**: 0 means completely unmethylated (no methyl groups), 1 means completely methylated (methyl groups present)
- **Cancer Relevance**: Cancer cells often show abnormal methylation patterns, with some regions becoming highly methylated (silencing tumor suppressor genes) and others becoming unmethylated (activating cancer-promoting genes)

The primary analysis in this project focuses on gene expression data, while miRNA and methylation data are included for potential future multi-omics integration.

## Cancer Biology Simplified

For those unfamiliar with cancer biology:

- **Normal Cells vs. Cancer Cells**: Normal cells grow, divide, and die in a controlled manner. Cancer cells grow uncontrollably and don't die when they should.

- **Genetic and Epigenetic Changes**: Cancer develops due to changes in the DNA sequence (genetic) and in how genes are regulated (epigenetic, like methylation).

- **Multiple Factors**: Many genes and regulatory mechanisms work together in cancer development, which is why we analyze different types of data simultaneously.

- **Personalized Medicine**: By understanding the molecular signatures of each patient's cancer, we hope to develop more targeted, effective treatments.

## Prediction Task

For this demonstration, we create a binary classification task based on gene expression data:

- Patients are divided into two groups based on the median expression value of a randomly selected gene
- The task is to predict which group a patient belongs to based on their overall gene expression profile
- This serves as a proof of concept for the pipeline and models, demonstrating their ability to identify patterns in high-dimensional genomic data

In real applications, we might predict more clinically relevant outcomes such as:
- Patient survival
- Response to specific treatments
- Cancer subtype classification
- Risk of recurrence

## Pipeline Steps

The pipeline consists of three main steps:

### 1. Data Download (`download_tcga.py`)
- Downloads the specified number of patient samples (currently set to 100) from the GDC API
- Saves the files locally and creates a metadata file
- The GDC (Genomic Data Commons) is a database that stores TCGA data and other cancer genomics datasets

### 2. Data Preprocessing (`preprocess_tcga.py`)
- Reads and parses the gene expression files
- Combines data across patients
- Selects the top 5,000 most variable genes to reduce dimensionality
  - Why? Human cells have about 20,000 protein-coding genes, but not all of them vary significantly between patients or are relevant to cancer
  - By focusing on the most variable genes, we focus on those likely to be informative
- Creates patient similarity graphs and gene similarity graphs
- Computes graph Laplacian eigenvectors for both graphs
- Splits the data into training, validation, and test sets
- Saves the processed data for model training

### 3. Model Training (`train_tcga_citrus.py`)
- Trains multiple models on the preprocessed data
- Evaluates model performance
- Saves trained models and performance metrics

## What are Graphs in This Context?

In this project, "graphs" refer to mathematical structures consisting of nodes (vertices) and edges, not charts or plots:

- **Nodes**: Individual entities (patients or genes)
- **Edges**: Connections between nodes, indicating similarity or relationship
- **Edge Weights**: Numbers indicating how strong the relationship is

Think of it like a social network where friends (nodes) are connected (edges), and the strength of friendship is the edge weight.

## Graph Construction

Two types of graphs are constructed during preprocessing:

### 1. Patient Similarity Graph
- **Nodes**: Individual patients
- **Edges**: Connect patients with similar gene expression profiles
- **How similarity is measured**: Correlation between gene expression patterns
- **Construction method**: K-nearest neighbors approach with k=3
  - Each patient is connected to the 3 most similar other patients
  - This creates a sparse but informative graph

### 2. Gene Similarity Graph
- **Nodes**: Individual genes
- **Edges**: Connect genes with similar expression patterns across patients
- **How similarity is measured**: Correlation between expression patterns
- **Construction method**: K-nearest neighbors approach with k=10
  - Each gene is connected to the 10 most similar other genes
  - This captures gene co-expression networks

### Similarity Measurement Details

The similarity between nodes (patients or genes) is measured using Pearson correlation coefficient. A higher correlation indicates greater similarity. Here's how it works:

For patients:
- We compute the correlation between the gene expression profiles of each pair of patients
- Each patient is connected to their k most correlated neighbors
- The correlation value becomes the weight of the edge

For genes:
- We compute the correlation between the expression patterns of each pair of genes across patients
- Each gene is connected to their k most correlated neighbors
- The correlation value becomes the weight of the edge

#### Code Implementation

The core similarity graph construction is implemented as follows:

```python
# For patient similarity (X_scaled has shape [n_patients, n_genes])
def create_similarity_graph(data_matrix, n_neighbors=10):
    """Create a similarity graph based on correlation"""
    n_samples = data_matrix.shape[0]
    
    # Create graph
    G = nx.Graph()
    
    # Add nodes
    for i in range(n_samples):
        G.add_node(i)
    
    # Using k-nearest neighbors method
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
    
    return G
```

This code calculates the correlation between each pair of samples (patients or genes) and then connects each node to its k most similar neighbors, where similarity is defined by Pearson correlation.

### From Graphs to Laplacian Eigenvectors

After constructing the patient and gene similarity graphs, the pipeline computes their graph Laplacian matrices and the associated eigenvectors:

1. **Graph Laplacian Matrix**:  
   L = D - A, where D is the degree matrix and A is the adjacency matrix of the graph

2. **Normalized Laplacian**:  
   L_normalized = D^(-1/2) × L × D^(-1/2)

3. **Eigenvector Decomposition**:  
   We compute the smallest k eigenvectors of the normalized Laplacian

These eigenvectors capture the intrinsic geometry of the patient and gene spaces and are used by CITRUS to learn from the tensor product graph structure.

## What is a Graph Neural Network?

A Graph Neural Network (GNN) is a type of artificial intelligence that can learn from data organized as graphs:

- **Beyond Regular Neural Networks**: Traditional neural networks treat each sample independently, whereas GNNs can use information about how samples are related to each other
- **Message Passing**: GNNs work by passing information between connected nodes, allowing them to learn patterns based on both individual features and the network structure
- **Advantage for Biological Data**: Biological systems are inherently interconnected (genes work in pathways, patients share disease characteristics), making graphs a natural way to represent them

## CITRUS vs. Traditional GNNs

Traditional GNNs like Graph Convolutional Networks (GCNs) operate on a single graph structure. In contrast, CITRUS can effectively operate on two coupled graphs simultaneously via their tensor product.

The advantages of CITRUS include:

- **Multi-modal integration**: Can naturally incorporate multiple data types (gene expression, miRNA, methylation)
- **Interpretability**: The model structure reflects the biological reality where both patient similarities and gene interactions matter
- **Efficiency**: Uses mathematical properties of tensor products to avoid explicitly forming the full product graph
- **Regularization**: The tensor product structure provides an implicit regularization effect, which is beneficial for high-dimensional genomic data

## Models and Parameters

The pipeline trains and compares multiple models:

### 1. Graph Convolutional Network (GCN)
- Leverages the patient similarity graph to capture relationships between patients
- Hidden dimension: 64
- Two convolutional layers followed by a fully connected layer
- **Intuition**: GCNs can learn that "patients similar to this one tended to be in group X, so this patient is likely in group X too"

### 2. Multi-Layer Perceptron (MLP)
- Standard neural network that doesn't use graph structure
- Hidden dimension: 64
- Three fully connected layers
- **Intuition**: MLPs look only at a patient's gene expression values without considering relationships between patients
- Serves as a baseline for comparison

### Training parameters
- **Epochs**: 100 (the number of complete passes through the training dataset)
- **Learning rate**: 0.001 (controls how quickly the model updates its internal parameters)
- **Weight decay**: 0.00001 (a regularization technique to prevent overfitting)
- **Loss function**: Cross-entropy (measures the performance of the classification model)
- **Optimizer**: Adam (an algorithm that updates the model parameters based on the training data)

## Performance Evaluation Explained

Models are evaluated based on several metrics:

### Accuracy
- The proportion of predictions that are correct
- Example: If 80 out of 100 patients are classified correctly, accuracy is 80%

### Precision
- The proportion of positive identifications that are actually correct
- Example: Of all patients predicted to be in group A, what percentage are actually in group A?

### Recall
- The proportion of actual positives that are identified correctly
- Example: Of all patients actually in group A, what percentage are correctly identified as group A?

### F1 Score
- The harmonic mean of precision and recall
- Provides a balance between precision and recall

### ROC AUC Score
- Area Under the Receiver Operating Characteristic Curve
- Measures how well the model can distinguish between classes
- Ranges from 0.5 (no better than random guessing) to 1.0 (perfect classification)
- Example: An AUC of 0.8 means there's an 80% chance that the model will rank a randomly chosen positive example higher than a randomly chosen negative example

Results are tracked separately for training, validation, and test sets. The best model is selected based on validation accuracy.

## Usage

To run the full pipeline:
```
python run_tcga_pipeline.py
```

To skip specific steps:
```
python run_tcga_pipeline.py --skip-download  # Skip downloading data
python run_tcga_pipeline.py --skip-preprocess  # Skip preprocessing
```

## Hardware Requirements

- The preprocessing step can be memory-intensive, particularly for the graph construction and Laplacian eigenvector computation
- For datasets with many patients or genes, a machine with at least 16GB RAM is recommended
- GPU acceleration is supported for model training but not required
  - **What's a GPU?**: A Graphics Processing Unit, originally designed for rendering images but now widely used to accelerate machine learning because of its ability to perform many calculations simultaneously

## Data Size Considerations

- By default, the pipeline uses the top 5,000 most variable genes to reduce dimensionality
- The number of patients can be controlled in the download script (currently set to 100)
- For larger datasets, the preprocessing script includes optimizations like batch processing and subsampling for graph construction

## Results and Output

After running the pipeline, results are available in `data/TCGA/results/`:
- **Model performance metrics**: `results.json` (a structured file containing all evaluation metrics)
- **Training curves**: `training_results.png` (plots showing how performance metrics change during training)
- **Best models**: `best_gcn_model.pt`, `best_mlp_model.pt` (saved model weights that can be loaded for future use)

## Understanding the Results

When interpreting the results, consider:

1. **Comparative performance**: Does the GCN outperform the MLP? This would suggest that patient relationships (graph structure) contain useful information.

2. **Overfitting**: Are training metrics much better than validation metrics? This suggests the model has memorized the training data rather than learning generalizable patterns.

3. **Generalization**: How well do the models perform on the test set? This indicates how well they might work on new, unseen patients.

## Future Work

- **Integration of multi-omics data**: Combining gene expression with miRNA and methylation to get a more complete picture of cancer biology
- **More sophisticated prediction tasks**: Using real clinical outcomes like survival time or treatment response
- **Implementation of the full CITRUS model**: A more advanced graph neural network for tensor product graphs
- **Evaluation on larger patient cohorts**: Testing the models on more patients to improve statistical power and generalizability

## Glossary of Technical Terms

### Biological Terms
- **Genome**: The complete set of genetic information (DNA) in an organism
- **Transcriptome**: The complete set of RNA transcripts in a cell
- **Epigenome**: Chemical modifications to DNA and histones that regulate gene expression
- **Gene**: A segment of DNA that contains the instructions for making a specific protein or RNA
- **Expression**: The process by which information from a gene is used to create a functional product

### Machine Learning Terms
- **Feature**: An individual measurable property (in our case, gene expression values)
- **Label**: The category we're trying to predict (binary group assignment)
- **Training/Validation/Test Sets**: Different subsets of data used for model training, tuning, and evaluation
- **Epoch**: One complete pass through the training dataset
- **Overfitting**: When a model learns the training data too well but performs poorly on new data
- **Hyperparameter**: A configuration setting for the model or training process 