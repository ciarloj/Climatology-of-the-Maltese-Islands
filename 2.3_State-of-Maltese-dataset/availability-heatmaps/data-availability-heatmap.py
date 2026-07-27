import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

# --- Configuration ---
workdir = "/mnt/d/Storage/James/Dropbox/PALEOSIM/Research/WP2/climate assessment/Maltese Islands Climate"
mia_dir = f"{workdir}/Data/"
output_dir = f"{workdir}/analysis/availability-heatmaps/"

stations = [
    "Luqa", "Dingli", "Bengħajsa", "Birkirkara", "Selmun", 
    "Valletta", "Xagħra", "Xewkija", "San-Lawrenz", "Zokrija", "Giordan"
]
DISPLAY_NAME_OVERRIDES = {"Birkirkara": "Msida"}

variables = [
    "Precipitation", "Temperature", "Max Temperature", "Min Temperature", 
    "Relative Humidity", "Solar", "Pressure", "Wind Speed", 
    "Wind Direction", "Wind Gust"
]

# Ensure output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# --- Helper Functions ---

def load_station_data(station_name):
    """Loads and cleans data for a specific station."""
    filepath = os.path.join(mia_dir, f"Daily_{station_name}.csv")
    
    if not os.path.exists(filepath):
        print(f"  [!] Missing file: {filepath}")
        return None
    
    try:
        # Load, treating dashes as NaN immediately
        df = pd.read_csv(filepath, na_values=['-', ' -', '- ', 'nan', 'NaN'])
    except Exception as e:
        print(f"  [!] Error reading {station_name}: {e}")
        return None

    # Clean Column Names (remove hidden BOM or spaces)
    df.columns = [c.replace('\ufeff', '').strip() for c in df.columns]

    # Parse Date
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        # Drop rows with invalid dates
        df = df.dropna(subset=['Date'])
    else:
        print(f"  [!] 'Date' column missing in {station_name}")
        return None

    return df

def create_custom_cmap():
    """Creates a custom White -> Brown colormap."""
    # Define colors: Start White (0%), End Brown (100%)
    colors = ["#ffffff", "#8B4513"] # White to SaddleBrown
    return mcolors.LinearSegmentedColormap.from_list("custom_white_brown", colors)

# --- Main Processing Loop ---

print(f"Starting Heatmap Generation...")
print(f"Output Directory: {output_dir}\n")

for stn in stations:
    print(f"Processing Station: {stn}...")
    
    # 1. Load Data
    df = load_station_data(stn)
    if df is None or df.empty:
        continue

    # 2. Prepare Time Columns
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    
    # Calculate days in each month for the denominator (accuracy)
    # We group by Year/Month and count the number of rows (assuming 1 row per day in file)
    # If your file has gaps (missing rows), we should strictly use calendar days, 
    # but using 'size()' is standard for these CSVs.
    monthly_totals = df.groupby(['Year', 'Month']).size()

    for var in variables:
        # Check if variable exists in this station's file
        # Case-insensitive match
        col_match = None
        exact_match = [c for c in df.columns if c.lower() == var.lower()]
        
        # Handle "Max Temperature" vs "Temperature" overlap
        if len(exact_match) > 1:
            # If multiple, try to find exact, otherwise careful filter
            col_match = var if var in df.columns else exact_match[0]
        elif len(exact_match) == 1:
            col_match = exact_match[0]
        else:
            # Fuzzy match attempt
            matches = [c for c in df.columns if var.lower() in c.lower()]
            if var == "Temperature":
                matches = [c for c in matches if "max" not in c.lower() and "min" not in c.lower()]
            if matches:
                col_match = matches[0]

        if not col_match:
            continue

        # 3. Calculate Availability Percentage
        # Force numeric (non-numeric becomes NaN)
        valid_series = pd.to_numeric(df[col_match], errors='coerce')
        
        # Combine with time data
        analysis_df = pd.DataFrame({
            'Year': df['Year'],
            'Month': df['Month'],
            'Valid': valid_series.notna() # True if data exists, False if NaN
        })

        # Count Valid entries per month
        monthly_valid = analysis_df.groupby(['Year', 'Month'])['Valid'].sum()

        # Calculate Percentage: (Valid Count / Total Rows in Month) * 100
        # Reindex to ensure we match the specific months present
        percentage_df = (monthly_valid / monthly_totals * 100).reset_index(name='Percentage')
        
        # 4. Pivot for Heatmap (Y=Month, X=Year)
        heatmap_data = percentage_df.pivot(index='Month', columns='Year', values='Percentage')
        
        # Fill missing months in the grid with NaN (or 0 if you prefer)
        # Reindex y-axis to ensure 1-12 sort order
        heatmap_data = heatmap_data.reindex(range(1, 13))
        
        # Stop if map is empty (e.g. column existed but was all empty)
        if heatmap_data.isna().all().all():
            continue

        # 5. Plotting
        plt.figure(figsize=(12, 8))
        
        # Create Heatmap
        # annot=True shows numbers
        # fmt=".1f" shows 1 decimal place
        # cbar=False hides the sidebar
        ax = sns.heatmap(
            heatmap_data, 
            annot=True, 
            fmt=".1f", 
            cmap=create_custom_cmap(),
            vmin=0, vmax=100,
            cbar=False,
            linewidths=0.5,
            linecolor='gray'
        )

        # Formatting
        display_stn = DISPLAY_NAME_OVERRIDES.get(stn, stn)
        plt.title(f"{display_stn} - {var} Data Availability (%)", fontsize=16, pad=20)
        plt.ylabel("Month", fontsize=12)
        plt.xlabel("Year", fontsize=12)
        
        # Set Y-ticks to Month Names
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ax.set_yticklabels(month_names, rotation=0)

        # Save Plot
        safe_var = var.replace(" ", "_")
        filename = f"{display_stn}_{safe_var}_Availability.png"
        save_path = os.path.join(output_dir, filename)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close() # Close memory to prevent overflow
        
print("\nProcessing Complete.")
