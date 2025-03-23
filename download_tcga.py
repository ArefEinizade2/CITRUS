import os
import requests
import json
import pandas as pd
import numpy as np
import time

# Configuration
CANCER_TYPE = "BRCA"  # Breast cancer
DATA_TYPES = ["Gene Expression Quantification", "miRNA Expression Quantification", "Methylation Beta Value"]
OUTPUT_DIR = "data/TCGA"

# GDC API endpoints
GDC_API_BASE = "https://api.gdc.cancer.gov/"
FILES_ENDPOINT = GDC_API_BASE + "files"
CASES_ENDPOINT = GDC_API_BASE + "cases"

def get_file_data(file_id, output_dir):
    """Download a file from GDC API given its UUID"""
    data_endpt = GDC_API_BASE + "data/" + file_id
    response = requests.get(data_endpt, headers={"Content-Type": "application/json"})
    
    # Get file name from content-disposition header
    content_disp = response.headers.get("Content-Disposition")
    if content_disp:
        filename = content_disp.split("filename=")[1].strip("\"")
    else:
        filename = file_id
    
    output_file = os.path.join(output_dir, filename)
    with open(output_file, "wb") as f:
        f.write(response.content)
    
    return output_file

def search_files(data_type, cancer_type, file_extension=".txt"):
    """Search for files of a specific data type for a specific cancer"""
    filters = {
        "op": "and",
        "content": [
            {
                "op": "in",
                "content": {
                    "field": "cases.project.project_id",
                    "value": [f"TCGA-{cancer_type}"]
                }
            },
            {
                "op": "in",
                "content": {
                    "field": "data_type",
                    "value": [data_type]
                }
            },
            {
                "op": "in",
                "content": {
                    "field": "access",
                    "value": ["open"]
                }
            }
        ]
    }
    
    params = {
        "filters": json.dumps(filters),
        "fields": "file_id,file_name,cases.submitter_id,cases.case_id,data_category,data_type",
        "format": "JSON",
        "size": "100"  # Adjust size as needed
    }
    
    response = requests.get(FILES_ENDPOINT, params=params)
    files_data = json.loads(response.content.decode("utf-8"))
    
    return files_data["data"]["hits"]

def download_and_process_data():
    """Main function to download and process TCGA data"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    all_files_info = []
    
    for data_type in DATA_TYPES:
        print(f"Searching for {data_type} files...")
        files = search_files(data_type, CANCER_TYPE)
        print(f"Found {len(files)} files.")
        
        # Download a subset of files (adjust as needed)
        download_count = min(100, len(files))
        
        for i in range(download_count):
            file_info = files[i]
            file_id = file_info["file_id"]
            print(f"Downloading file {i+1}/{download_count}: {file_id}")
            
            try:
                output_file = get_file_data(file_id, OUTPUT_DIR)
                file_info["local_path"] = output_file
                all_files_info.append(file_info)
                print(f"  Saved to {output_file}")
                
                # Sleep to avoid overwhelming the API
                time.sleep(1)
            except Exception as e:
                print(f"  Error downloading: {e}")
    
    # Save file information
    files_df = pd.DataFrame(all_files_info)
    files_df.to_csv(os.path.join(OUTPUT_DIR, "files_metadata.csv"), index=False)
    print(f"Saved metadata for {len(files_df)} files.")

if __name__ == "__main__":
    download_and_process_data() 