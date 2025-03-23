import os
import subprocess
import time
import argparse

def run_command(command, description):
    """Run a command and print timing information"""
    print(f"\n{'=' * 80}")
    print(f"  {description}")
    print(f"{'=' * 80}")
    
    start_time = time.time()
    result = subprocess.run(command, shell=True)
    end_time = time.time()
    
    if result.returncode != 0:
        print(f"Error running command: {command}")
        exit(1)
    
    print(f"\nCompleted in {(end_time - start_time)/60:.2f} minutes")
    return result

def main():
    parser = argparse.ArgumentParser(description='Run the TCGA analysis pipeline with CITRUS')
    parser.add_argument('--skip-download', action='store_true', help='Skip the data download step')
    parser.add_argument('--skip-preprocess', action='store_true', help='Skip the preprocessing step')
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs("data/TCGA", exist_ok=True)
    os.makedirs("data/TCGA/processed", exist_ok=True)
    os.makedirs("data/TCGA/results", exist_ok=True)
    
    # Step 1: Download data
    if not args.skip_download:
        run_command("python download_tcga.py", "Downloading TCGA data")
    else:
        print("Skipping download step...")
    
    # Step 2: Preprocess data
    if not args.skip_preprocess:
        run_command("python preprocess_tcga.py", "Preprocessing TCGA data for CITRUS")
    else:
        print("Skipping preprocessing step...")
    
    # Step 3: Train and evaluate models
    run_command("python train_tcga_citrus.py", 
                "Training and evaluating GCN and MLP models")
    
    print("\n" + "=" * 80)
    print("  Pipeline completed successfully!")
    print("=" * 80)
    print("\nResults are available in data/TCGA/results/")
    print("- Model performance metrics: results.json")
    print("- Training curves: training_results.png")
    print("- Best models saved as: best_gcn_model.pt, best_mlp_model.pt")

if __name__ == "__main__":
    main() 