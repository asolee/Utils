import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os # Import os for directory operations

def create_dodged_barplot(dataset: pd.DataFrame, meta_column: str, value_columns: list, focus_value: list = None, add_error_bars: bool = False, output: str = None):
    """
    Generates a dodged bar plot with specified columns, collapsing, and optional error bars
    representing the raw standard deviation.

    Args:
        dataset (pd.DataFrame): The input pandas DataFrame.
        meta_column (str): The column to be used on the X-axis (only one).
        value_columns (list): A list of columns that represent the components of the dodged bar.
        focus_value (list, optional): A subset of value_columns. Columns in focus_value will be plotted normally,
                                      while all other columns in value_columns will be collapsed into an "others" category.
                                      If None or an empty list, all value_columns are plotted individually without an "others" category.
        add_error_bars (bool, optional): If True, error bars representing the raw standard deviation
                                         of the counts will be added to the bars. Defaults to False.
        output (str, optional): The base path and filename (without extension) to save the plot.
                                If provided, the plot will be saved as a PDF and PNG.
    """

    # --- Input Validation ---
    if not isinstance(dataset, pd.DataFrame):
        raise TypeError("Input 'dataset' must be a pandas DataFrame.")
    if meta_column not in dataset.columns:
        raise ValueError(f"Column '{meta_column}' not found in the dataset.")
    if not all(col in dataset.columns for col in value_columns):
        raise ValueError("Not all columns in 'value_columns' found in the dataset.")

    # Ensure focus_value is a list, defaulting to empty if None
    if focus_value is None:
        focus_value = []

    if not all(col in value_columns for col in focus_value):
        raise ValueError("All columns in 'focus_value' must also be present in 'value_columns'.")

    # --- Data Preparation ---
    # Create a copy of the relevant columns to avoid modifying the original DataFrame
    df_plot = dataset[[meta_column] + value_columns].copy()

    # Identify columns to be collapsed into "others"
    others_cols = [col for col in value_columns if col not in focus_value]

    colors = [] # Initialize list to hold colors for the plot's categories
    cols_to_plot_y = [] # Initialize list for columns that will be plotted on Y-axis

    if others_cols and len(others_cols) < len(value_columns):
        # Calculate 'others' sum for plotting
        df_plot['others'] = df_plot[others_cols].sum(axis=1) # Still sum for 'others' aggregation
        cols_to_plot_y = focus_value + ['others']
        df_plot_final = df_plot[[meta_column] + cols_to_plot_y]

        # Assign colors
        prop_cycle = plt.rcParams['axes.prop_cycle']
        default_colors = prop_cycle.by_key()['color']
        colors = default_colors[:len(focus_value)] + ['gray']
    else:
        cols_to_plot_y = value_columns
        df_plot_final = df_plot[[meta_column] + cols_to_plot_y]

        # Assign colors
        prop_cycle = plt.rcParams['axes.prop_cycle']
        default_colors = prop_cycle.by_key()['color']
        colors = default_colors[:len(value_columns)]

    # Group by the meta_column and calculate the median of the values for the Y-axis columns
    grouped_df = df_plot_final.groupby(meta_column)[cols_to_plot_y].median()

    # --- Calculate Error Bars (if requested) ---
    yerr_data = None
    if add_error_bars:
        # Calculate standard deviation for each component within each meta_column group
        grouped_std = df_plot_final.groupby(meta_column)[cols_to_plot_y].std()
        grouped_std = grouped_std.fillna(0) # Fill NaN standard deviations with 0
        yerr_data = grouped_std # Use directly for dodged bars

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(12, 7)) # Adjust figure size for better readability of dodged bars

    num_meta_categories = len(grouped_df.index)
    num_value_categories = len(cols_to_plot_y)
    bar_width = 0.8 / num_value_categories # Width of each individual bar

    # Create x-axis positions for each group of bars
    x = np.arange(num_meta_categories)

    for i, col in enumerate(cols_to_plot_y):
        # Calculate offset to place bars side-by-side within each meta_column group
        offset = bar_width * i - (bar_width * (num_value_categories - 1) / 2)
        ax.bar(x + offset, grouped_df[col], bar_width, label=col,
               yerr=yerr_data[col].values if add_error_bars else None, capsize=4, color=colors[i])

    plt.title(f'Dodged Bar Plot of Medians by {meta_column}', fontsize=14)
    plt.xlabel(meta_column, fontsize=12)
    plt.ylabel('Median of Values', fontsize=12) # Y-axis now represents medians
    plt.xticks(x, grouped_df.index, rotation=45, ha='right', fontsize=10) # Set x-ticks explicitly
    plt.yticks(fontsize=10)
    plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, title_fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout() # Adjust plot parameters for a tight layout

    # --- Saving Plot (if output path is provided) ---
    if output:
        dir_name = os.path.dirname(output)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name)

        filename_pdf = output + ".pdf"
        plt.savefig(filename_pdf)
        print(f"Dodged bar plot saved to {filename_pdf}")

        filename_png = output + ".png"
        plt.savefig(filename_png)
        print(f"Dodged bar plot saved to {filename_png}")

    plt.show()
