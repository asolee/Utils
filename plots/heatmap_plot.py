import pandas as pd
import matplotlib.pyplot as plt
import PyComplexHeatmap as pch
from sklearn.preprocessing import MinMaxScaler
import os
import matplotlib.cm as cm # For default colormaps
import matplotlib.colors as mcolors # For more detailed color mapping
import numpy as np # Import numpy for checking all zeros

def create_combined_heatmap_from_dataframe(df, value_columns, metadata_columns, rescale_values=False, min_relative_abundance=0.0, min_sample_percentage=0.0, row_cluster=True, remove_all_zero=False, output="output/", metadata_colors_mapping=None):
    """
    Generates a heatmap from a pandas DataFrame, displaying multiple
    value columns along with multiple metadata columns.

    Args:
        df (pd.DataFrame): The input pandas DataFrame.
        value_columns (list of str): A list of column names in `df` to use for the heatmap values.
        metadata_columns (list of str): A list of column names in `df` to use as
            metadata. These columns will form part of the x and y axes.
        rescale_values (bool, optional): If True, values in `value_columns` will be
            rescaled to a 0-1 range using Min-Max scaling before plotting.
            Defaults to False.
        min_relative_abundance (float, optional): The minimum relative abundance (between 0 and 1)
            a cell type must have in a sample to be considered. Defaults to 0.0.
        min_sample_percentage (float, optional): The minimum percentage of samples (between 0 and 1)
            in which a cell type must meet `min_relative_abundance` to be included. Defaults to 0.0.
        row_cluster (bool, optional): If True, rows will be clustered. Defaults to True.
        remove_all_zero (bool, optional): If True, value columns that contain only zero
            values across all rows will be removed. Defaults to False.
        output (str, optional): The base path for saving the heatmap. Defaults to "output/".
        metadata_colors_mapping (dict, optional): A dictionary where keys are metadata column names
            and values are dictionaries mapping specific metadata values to their colors.
            If a color is not provided for a value, it will be automatically assigned.
            Defaults to None.

    Returns:
        A single PyComplexHeatmap heatmap. Returns None if the input DataFrame is
            empty or if the specified columns are not found, or if no value columns
            remain after filtering.
    """

    # --- Basic input validation ---
    if not isinstance(df, pd.DataFrame):
        print("Error: df must be a pandas DataFrame.")
        return None
    if df.empty:
        print("Error: DataFrame is empty.")
        return None
    if not isinstance(value_columns, list) or not all(isinstance(col, str) for col in value_columns):
        print("Error: value_columns must be a list of strings.")
        return None
    if not isinstance(metadata_columns, list) or not all(isinstance(col, str) for col in metadata_columns):
        print("Error: metadata_columns must be a list of strings.")
        return None
    if not isinstance(min_relative_abundance, (int, float)) or not (0.0 <= min_relative_abundance <= 1.0):
        print("Error: min_relative_abundance must be a float between 0 and 1.")
        return None
    if not isinstance(min_sample_percentage, (int, float)) or not (0.0 <= min_sample_percentage <= 1.0):
        print("Error: min_sample_percentage must be a float between 0 and 1.")
        return None
    if not isinstance(row_cluster, bool):
        print("Error: row_cluster must be a boolean.")
        return None
    if not isinstance(remove_all_zero, bool):
        print("Error: remove_all_zero must be a boolean.")
        return None
    if metadata_colors_mapping is not None and not isinstance(metadata_colors_mapping, dict):
        print("Error: metadata_colors_mapping must be a dictionary or None.")
        return None

    # --- Check for the existence of columns in the DataFrame ---
    all_columns = value_columns + metadata_columns
    missing_columns_in_df = [col for col in all_columns if col not in df.columns]
    if missing_columns_in_df:
        print(f"Error: The following columns are not found in the DataFrame: {', '.join(missing_columns_in_df)}")
        return None

    # --- Filter value_columns based on min_relative_abundance and min_sample_percentage ---
    filtered_value_columns = []
    num_samples = len(df)
    if num_samples == 0:
        print("Warning: DataFrame is empty, no filtering possible for value columns.")
        return None

    for col in value_columns:
        samples_meeting_abundance = (df[col] >= min_relative_abundance).sum()
        percentage_of_samples = samples_meeting_abundance / num_samples

        if percentage_of_samples >= min_sample_percentage:
            filtered_value_columns.append(col)
    
    if not filtered_value_columns:
        print(f"Warning: No value columns remain after filtering with min_relative_abundance={min_relative_abundance} and min_sample_percentage={min_sample_percentage}. Heatmap cannot be generated.")
        return None
    
    value_columns = filtered_value_columns # Update value_columns after initial filtering

    # --- Filter out columns with all zero values if remove_all_zero is True ---
    if remove_all_zero:
        non_zero_value_columns = []
        removed_zero_columns = []
        for col in value_columns:
            # Check if all values in the column are zero
            # Ensure the column is numeric before checking for all zeros
            if pd.api.types.is_numeric_dtype(df[col]):
                if not (df[col] == 0).all():
                    non_zero_value_columns.append(col)
                else:
                    removed_zero_columns.append(col)
            else:
                # If it's not numeric, we can't reliably check for all zeros,
                # so we keep it. Or, you might choose to raise an error/warning
                # if a non-numeric column is in value_columns.
                non_zero_value_columns.append(col)

        if removed_zero_columns:
            print(f"Info: Removed the following value columns as they contain all zero values: {', '.join(removed_zero_columns)}")
        
        value_columns = non_zero_value_columns # Update value_columns after all-zero filtering

        if not value_columns:
            print("Warning: No value columns remain after removing all-zero columns. Heatmap cannot be generated.")
            return None


    # --- Handling missing metadata for annotation and assigning colors ---
    df_for_metadata_annotation = df[metadata_columns].copy()
    
    colors_for_annotation = {} 

    default_cmaps = [plt.cm.tab10, plt.cm.Set3, plt.cm.Pastel1, plt.cm.Dark2, plt.cm.Accent]
    cmap_idx = 0

    for col_idx, col in enumerate(metadata_columns):
        # 1. Fill actual NaN values first with "Missing Information"
        df_for_metadata_annotation[col] = df_for_metadata_annotation[col].fillna("Missing Information")

        # 2. Then ensure the entire column is of string type.
        df_for_metadata_annotation[col] = df_for_metadata_annotation[col].astype(str)
        
        unique_values = df_for_metadata_annotation[col].unique()
        
        col_color_dict = {}
        
        # Always assign gray to "Missing Information"
        if "Missing Information" in unique_values:
            col_color_dict["Missing Information"] = "#BEBEBE" # Gray color for missing data

        # Values to assign colors to (excluding "Missing Information")
        values_to_color = sorted([val for val in unique_values if val != "Missing Information"])
        
        # Apply user-defined colors first
        if metadata_colors_mapping and col in metadata_colors_mapping:
            for val in values_to_color:
                if val in metadata_colors_mapping[col]:
                    col_color_dict[val] = metadata_colors_mapping[col][val]
        
        # Assign automatic colors for values not in user mapping
        unassigned_values = [val for val in values_to_color if val not in col_color_dict]
        
        if len(unassigned_values) > 0:
            current_cmap = default_cmaps[cmap_idx % len(default_cmaps)]
            cmap_idx += 1 
            
            # Normalize for colormap if there's more than one unassigned value
            norm = mcolors.Normalize(vmin=0, vmax=len(unassigned_values) - 1)
            for i, val in enumerate(unassigned_values):
                color = current_cmap(norm(i))
                col_color_dict[val] = mcolors.to_hex(color) 
        
        colors_for_annotation[col] = col_color_dict
            
    if not metadata_columns:
        print("Warning: No metadata columns provided or remaining after processing. Heatmap annotation will be skipped.")
        col_ha = None
    else:
        valid_metadata_columns = [col for col in metadata_columns if not df_for_metadata_annotation[col].empty]
        
        if not valid_metadata_columns:
            print("Warning: All metadata columns are empty or became effectively empty after processing. Heatmap annotation will be skipped.")
            col_ha = None
        else:
            final_colors_for_annotation = {
                col: colors_for_annotation.get(col, {}) for col in valid_metadata_columns
            }

            col_ha = pch.HeatmapAnnotation(df=df_for_metadata_annotation[valid_metadata_columns],
                                           colors=final_colors_for_annotation,
                                           plot=False, legend=True, legend_gap=5, hgap=0.5, axis=1)

    # --- Apply rescaling if requested ---
    if rescale_values:
        df_for_heatmap_data = df[value_columns].copy()
        scaler = MinMaxScaler()
        df_for_heatmap_data[value_columns] = scaler.fit_transform(df_for_heatmap_data[value_columns])
    else:
        df_for_heatmap_data = df[value_columns]

    # --- Calculate dynamic figure height based on number of rows (value_columns) ---.
    base_height = 8  # Minimum height
    height_per_row = 0.3 # Height contribution per row (value column)
    dynamic_height = max(base_height, len(value_columns) * height_per_row)

    # --- Initialize ClusterMapPlotter ---
    plt.figure(figsize=(10, dynamic_height)) # Use dynamic height here
    cm = pch.ClusterMapPlotter(data=df_for_heatmap_data.transpose(),
                               top_annotation=col_ha,
                               col_cluster=True,
                               row_cluster=row_cluster,
                               col_split_gap=0.5,
                               row_split_gap=0.8,
                               label='cell fraction prediction',
                               row_dendrogram=True,
                               col_dendrogram=True,
                               show_rownames=True,
                               show_colnames=False,
                               tree_kws={'row_cmap': 'Set1'},
                               verbose=0,
                               legend_gap=5,
                               cmap='viridis')
    
    # Create the directory if it doesn't exist
    dir_name = os.path.dirname(output)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name)
    
    # Save the plot (PDF)
    filename_pdf = output + ".pdf"
    plt.savefig(filename_pdf, bbox_inches='tight')

    # Save the plot (PNG)
    filename_png = output + ".png"
    plt.savefig(filename_png, bbox_inches='tight')
    
    plt.show()

    print(f"Heatmap saved to {filename_pdf} and {filename_png}")

