import pandas as pd
import numpy as np
import os
from scipy.stats import pearsonr

# --- Configuration ---
# List your stations here
stations = ["Luqa", "Dingli", "Bengħajsa", "Birkirkara", "Selmun", "Valletta", "Xagħra", "Xewkija", "Giordan"]
data_dir = "../Data/"
output_dir = "../analysis/corr_analysis/"

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# --- Helper Function to Read Data ---
def read_station_data(station_name):
    filename = f"{data_dir}Daily_{station_name}.csv"
    
    # Check if file exists
    if not os.path.exists(filename):
        print(f"Warning: File {filename} not found. Skipping.")
        return None
    
    # Read CSV
    # 'na_values' handles the "-" entries automatically
    try:
        df = pd.read_csv(filename, na_values=['-', ' -', '- ']) 
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None

    # Parse Date
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    else:
        print(f"Error: No 'Date' column in {filename}")
        return None
    
    # Convert all other columns to numeric
    for col in df.columns:
        if col != 'Date':
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    return df

# --- Main Logic ---

# 1. Load all data and identify all unique variables
station_dfs = {}
all_vars = set()

print("Loading data...")
for stn in stations:
    df = read_station_data(stn)
    if df is not None:
        station_dfs[stn] = df
        # Add all columns except Date to our list of variables
        all_vars.update([c for c in df.columns if c != 'Date'])

print(f"Variables found: {all_vars}")

# 2. Loop through each variable
for var in all_vars:
    print(f"Processing variable: {var}...")
    
    # Create a combined DataFrame for this variable
    combined_data = {}
    
    for stn in stations:
        if stn in station_dfs:
            df = station_dfs[stn]
            if var in df.columns:
                # Set date as index to align rows
                series = df.set_index('Date')[var]
                # Remove any duplicate dates
                series = series[~series.index.duplicated(keep='first')]
                combined_data[stn] = series
    
    # Skip if no data found for this variable
    if not combined_data:
        continue
        
    # Create the matrix (aligns dates automatically)
    df_matrix = pd.DataFrame(combined_data)
    
    # --- Calculate Correlation ---
    # min_periods=30 ensures we don't correlate if we have too few overlapping days
    corr_matrix = df_matrix.corr(method='pearson', min_periods=30)
    
    # --- Calculate P-values ---
    # Pandas doesn't have a built-in p-value matrix, so we calculate it manually
    pval_matrix = pd.DataFrame(index=df_matrix.columns, columns=df_matrix.columns, dtype=float)
    
    cols = df_matrix.columns
    for col1 in cols:
        for col2 in cols:
            if col1 == col2:
                pval_matrix.loc[col1, col2] = 0.0 # p-value with self is 0
                continue
            
            # Get valid paired data (drop rows where either station is NaN)
            valid_data = df_matrix[[col1, col2]].dropna()
            
            if len(valid_data) > 30: # Only calculate if we have enough data points
                stat, p = pearsonr(valid_data[col1], valid_data[col2])
                pval_matrix.loc[col1, col2] = p
            else:
                pval_matrix.loc[col1, col2] = np.nan

    # --- Save Results ---
    # Create a safe filename (replace spaces with underscores)
    safe_var = var.replace(" ", "_").replace("/", "-")
    
    corr_path = os.path.join(output_dir, f"corr_{safe_var}.csv")
    pval_path = os.path.join(output_dir, f"pval_{safe_var}.csv")
    
    corr_matrix.to_csv(corr_path)
    pval_matrix.to_csv(pval_path)

print("Processing complete.")
