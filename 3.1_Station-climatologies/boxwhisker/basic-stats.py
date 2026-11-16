import pandas as pd
import numpy as np
import os

# --- Configuration ---
workdir = "/mnt/d/Storage/James/Dropbox/PALEOSIM/Research/WP2/climate assessment/Maltese Islands Climate"
mia_dir = f"{workdir}/Data/"
output_dir = f"{workdir}/analysis/basic-stats/"

# Station Configurations
mia_stations = ["Luqa", "Dingli", "Bengħajsa", "Birkirkara", "Selmun", "Valletta", "Xagħra", "Xewkija", "San-Lawrenz", "Zokrija", "Giordan"]

# Variable Mappings for Comparison (Manikata Name : MIA Name)
# Note: Pressure is handled specifically for Luqa below
variables = ["Precipitation", "Temperature", "Max Temperature", "Min Temperature", "Relative Humidity", "Solar", "Pressure", "Wind Speed", "Wind Direction", "Wind Gust"]

# Ensure output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# --- Helper Functions ---

def load_mia(station_name):
    filepath = os.path.join(mia_dir, f"Daily_{station_name}.csv")
    if not os.path.exists(filepath):
        return None
    
    try:
        # Load and handle missing values
        df = pd.read_csv(filepath, na_values=['-', ' -', '- '])
    except:
        return None

    # Remove hidden BOM characters and whitespace
    df.columns = [c.replace('\ufeff', '').strip() for c in df.columns]
    
    # Parse 'Date' and Normalize (ensure it is 00:00:00)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce').dt.normalize()       
        df = df[df['Date'] >= pd.Timestamp("2007-01-01")]
   
    return df

def get_basic_stats(series, mode="basic"):
    """
    Calculates the list of statistics requested.
    mode='manikata': n, %N, avg, max, p90, p75, med, p25, p10, min
    mode='compare': avg, max, p75, med, p25, min
    """
    # Drop NaNs for calculation
    clean_series = series.dropna()
    
    if len(clean_series) == 0:
        return {k: np.nan for k in ["n", "%N", "Average", "Max", "p90", "p75", "Median", "p25", "p10", "Min"]}

    stats = {}
    
    if mode == "basic":
        stats["n"] = len(series) # Total points including NaNs
        stats["%N"] = (len(clean_series) / len(series)) * 100
        stats["Average"] = clean_series.mean()
        stats["Max"] = clean_series.max()
        stats["p99"] = np.percentile(clean_series, 99)
        stats["p95"] = np.percentile(clean_series, 95)
        stats["p90"] = np.percentile(clean_series, 90)
        stats["p75"] = np.percentile(clean_series, 75)
        stats["Median"] = clean_series.median()
        stats["p25"] = np.percentile(clean_series, 25)
        stats["p10"] = np.percentile(clean_series, 10)
        stats["p05"] = np.percentile(clean_series,  5)
        stats["p01"] = np.percentile(clean_series,  1)
        stats["Min"] = clean_series.min()
        
    elif mode == "compare":
        stats["Average"] = clean_series.mean()
        stats["Max"] = clean_series.max()
        stats["p75"] = np.percentile(clean_series, 75)
        stats["Median"] = clean_series.median()
        stats["p25"] = np.percentile(clean_series, 25)
        stats["Min"] = clean_series.min()
        
    return stats

def get_rainy_day_stats(series):
    """
    Wet-day precipitation statistics (rainy days only, >= 0.1 mm).

    NOTE ON n / %N — these differ in MEANING from the all-day stats CSV:
      - In the standard stats_*.csv, n = all valid days and %N = record
        completeness (valid days / total days).
      - Here, n = number of WET days (>= 0.1 mm) and %N = wet-day FREQUENCY
        (wet days / valid days * 100), i.e. how often it rains — NOT a
        data-completeness metric. Do not feed this %N into completeness
        figures (e.g. the spider plot).
    All percentiles/mean/max are computed over wet days only, matching the
    wet-day box-whisker definition.
    """
    valid = series.dropna()
    n_valid = len(valid)
    if n_valid == 0:
        return {k: np.nan for k in ["n", "%N", "Average", "Max", "p99", "p95",
                "p90", "p75", "Median", "p25", "p10", "p05", "p01", "Min"]}

    wet = valid[valid >= 0.1]          # wet-day mask — keep in sync with plot & title
    if len(wet) == 0:
        return {k: np.nan for k in ["n", "%N", "Average", "Max", "p99", "p95",
                "p90", "p75", "Median", "p25", "p10", "p05", "p01", "Min"]}

    return {
        "n":       len(wet),                        # wet-day COUNT (see note above)
        "%N":      len(wet) / n_valid * 100,        # wet-day FREQUENCY, not completeness
        "Average": wet.mean(),
        "Max":     wet.max(),
        "p99":     np.percentile(wet, 99),
        "p95":     np.percentile(wet, 95),
        "p90":     np.percentile(wet, 90),
        "p75":     np.percentile(wet, 75),
        "Median":  wet.median(),
        "p25":     np.percentile(wet, 25),
        "p10":     np.percentile(wet, 10),
        "p05":     np.percentile(wet,  5),
        "p01":     np.percentile(wet,  1),
        "Min":     wet.min(),
    }

# --- Main Execution ---

# 1. Load Data
print("Loading MIA Data...")
mia_data = {}
for stn in mia_stations:
    mia_data[stn] = load_mia(stn)

# 2. Comparison with MIA Stations
print("Processing Comparisons...")

for target_var in variables:
    print(f"Analyzing {target_var}...")
    
    # We will build a DataFrame where:
    # Rows = Stats (Max, Average, Min)
    # Columns = Stations
    stats_results = {}
    
    for stn in mia_stations:
        df = mia_data.get(stn)
        
        # Default empty stats
        stats_out = {"n": np.nan, "%N": np.nan, "Average": np.nan, "Max": np.nan, "p99":np.nan, 
            "p95":np.nan, "p90":np.nan, "p75":np.nan, "Median": np.nan, "p25": np.nan, "p10": np.nan, 
            "p05":np.nan, "p01":np.nan, "Min": np.nan}
        
        if df is not None and not df.empty:
            # Find the column (Case insensitive check)
            col_match = None
            if target_var in df.columns:
                col_match = target_var
            else:
                matches = [c for c in df.columns if target_var.lower() in c.lower()]
                # Avoid confusing "Temperature" with "Max Temperature"
                if target_var == "Temperature":
                     matches = [c for c in matches if "max" not in c.lower() and "min" not in c.lower()]
                if matches: col_match = matches[0]
            
            if col_match:
                series = pd.to_numeric(df[col_match], errors='coerce')
                # Calculate basic stats
                stats_out = get_basic_stats(series)

        stats_results[stn] = stats_out

    # Save Results for this Variable
    if stats_results:
        # Create DataFrame from dict (Keys=Stations, Values=Stats)
        res_df = pd.DataFrame(stats_results)
        
        # Enforce row order
        row_order = ["n", "%N", "Average", "Max", "p99", "p95", "p90", "p75", "Median", 
            "p25", "p10", "p05", "p01", "Min"]
        res_df = res_df.reindex(row_order)
        
        # Clean filename
        safe_var_name = target_var.replace(" ", "_")
        output_filename = f"stats_{safe_var_name}.csv"
        output_path = os.path.join(output_dir, output_filename)
        
        res_df.to_csv(output_path)
        print(f"  > Saved {output_filename}")

    # --- Rainy-day precipitation CSV (precipitation only) ---
    # Separate artefact so the all-day stats_Precipitation.csv (used by the
    # completeness figures) is left untouched. See get_rainy_day_stats for
    # the different meaning of n / %N here.
    if target_var == "Precipitation":
         rainy_results = {}
         for stn in mia_stations:
             df = mia_data.get(stn)
             rainy_out = {k: np.nan for k in row_order}
             if df is not None and not df.empty:
                 col_match = None
                 if target_var in df.columns:
                     col_match = target_var
                 else:
                     matches = [c for c in df.columns if target_var.lower() in c.lower()]
                     if matches:
                         col_match = matches[0]
                 if col_match:
                     series = pd.to_numeric(df[col_match], errors='coerce')
                     rainy_out = get_rainy_day_stats(series)
             rainy_results[stn] = rainy_out

         rainy_df = pd.DataFrame(rainy_results).reindex(row_order)
         rainy_path = os.path.join(output_dir, "stats_Precipitation_rainydays.csv")
         rainy_df.to_csv(rainy_path)
         print(f"  > Saved stats_Precipitation_rainydays.csv")

print("Done.")
