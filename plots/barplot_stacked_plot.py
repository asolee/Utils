import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os # Import os for directory operations

def create_stacked_barplot(dataset: pd.DataFrame, meta_column: str, value_columns: list, focus_value: list = None, xlabel_fontsize: float = 12,xticks_fontsize: float = 10,yticks_fontsize: float = 10,collapse_focus_values_as: str = None, collapsed_color: str = None, focus_colors_map: dict = None, add_error_bars: bool = False, add_connecting_shades: bool = False, connecting_shades_alpha: float = 0.15, add_category_border: bool = False, category_border_width: float = 0.5, group_by_column: str = None, group_spacing: float = 0, group_label_rotation: float = 0, group_label_fontsize: int = 12, group_label_y_offset: float = 0.0, group_bracket_linewidth: float = 1.0, fig_width: float = 10, fig_height: float = 7, output: str = None, show_xlabel: bool = True, show_ylabel: bool = True, show_title: bool = True, title_fontsize: float = 14, ylabel_fontsize: float = 12):
    """
    Generates a stacked bar plot with specified columns, collapsing, scaling, and optional error bars
    representing the raw standard deviation.

    Args:

        #### BASIC INPUT PARAMETERS ####

        dataset (pd.DataFrame): The input pandas DataFrame.
        meta_column (str): The column to be used on the X-axis (only one).
        value_columns (list): A list of columns that represent the components of the stacked bar.
        focus_value (list, optional): A subset of value_columns.
                                       If `collapse_focus_values_as` is provided, columns in focus_value
                                       will be summed into a single category.
                                       If `collapse_focus_values_as` is None, columns in focus_value
                                       will be plotted individually, while all other columns in value_columns
                                       will be collapsed into an "others" category.
                                       If None or an empty list, all value_columns are plotted individually
                                       without an "others" category.
        output (str, optional): The base path and filename (without extension) to save the plot.
                                 If provided, the plot will be saved as a PDF and PNG.
        fig_width (float, optional): The width of the figure in inches. Defaults to 10.
        fig_height (float, optional): The height of the figure in inches. Defaults to 7.

        #### EDIT FOCUS VALUES FEATURES ####

        collapse_focus_values_as (str, optional): If provided and `focus_value` is not empty,
                                                   all columns listed in `focus_value` will be summed
                                                   and plotted as a single category with this name.
                                                   Cannot be named 'others'.
        collapsed_color (str, optional): The color to use for the collapsed category specified by
                                          `collapse_focus_values_as`. If None, a default color will be used.
        focus_colors_map (dict, optional): A dictionary mapping focus_value category names to specific colors.
                                            If a category is not in the map, a default color will be used.
                                            Only applies if `focus_value` is used for individual plotting
                                            (i.e., `collapse_focus_values_as` is None).

        #### ERROR BARS PARAMETERS ####

        add_error_bars (bool, optional): If True, error bars representing the raw standard deviation
                                          of the counts will be added to the bars. Defaults to False.

        #### CONNECTING SHADES PARAMETERS ####

        add_connecting_shades (bool, optional): If True, adds semi-transparent shaded regions connecting
                                                 the same categories across different stacked bars. Defaults to False.
        connecting_shades_alpha (float, optional): Transparency level for connecting shades. Defaults to 0.15.

        #### CATEGORY BORDER PARAMETERS ####

        add_category_border (bool, optional): If True, adds a thin black line around each category in the stacked bars.
                                               Defaults to False.
        category_border_width (float, optional): The width of the black line around categories if `add_category_border` is True.
                                                 Defaults to 0.5.

        #### METADATA GROUPING ####

        group_by_column (str, optional): The name of a column in `dataset` to use for grouping `meta_column` values.
                                          If provided, bars will be grouped based on this column's values,
                                          and group labels with brackets will be added below the x-axis. Defaults to None.
        group_spacing (float, optional): The extra space to add between groups of bars if `group_by_column` is used.
                                          Defaults to 0.
        group_label_rotation (float, optional): Rotation angle for the group labels in degrees. Defaults to 0.
        group_label_fontsize (int, optional): Font size for the group labels. Defaults to 12.
        group_label_y_offset (float, optional): Vertical offset for the group labels and brackets. A negative value
                                                 moves them further down from the x-axis. Defaults to 0.0.
        group_bracket_linewidth (float, optional): The line width for the square brackets when grouping is enabled.
                                                    Defaults to 1.0.

        #### FONTSIZE AND VISIBILITY PARAMETERS ####

        show_xlabel (bool, optional): If True, the x-axis label will be displayed. Defaults to True.
        xlabel_fontsize (float, optional): The font size for the x-axis label. Defaults to 12.
        show_ylabel (bool, optional): If True, the y-axis label will be displayed. Defaults to True.
        ylabel_fontsize (float, optional): The font size for the y-axis label. Defaults to 12.
        show_title (bool, optional): If True, the plot title will be displayed. Defaults to True.
        title_fontsize (float, optional): The font size for the plot title. Defaults to 14.
        xticks_fontsize (float, optional): The font size for the x-axis tick labels. Defaults to 10.
        yticks_fontsize (float, optional): The font size for the y-axis tick labels. Defaults to 10.
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

    # New validation for collapse_focus_values_as
    if collapse_focus_values_as is not None:
        if not isinstance(collapse_focus_values_as, str):
            raise TypeError("'collapse_focus_values_as' must be a string if provided.")
        if not focus_value:
            raise ValueError("'focus_value' cannot be empty if 'collapse_focus_values_as' is provided, as there would be nothing to collapse.")
        if collapse_focus_values_as == 'others':
            raise ValueError("'collapse_focus_values_as' cannot be named 'others' as it conflicts with an internal category name.")
        # Check for name conflict with existing columns not being melted
        if collapse_focus_values_as in dataset.columns and \
           collapse_focus_values_as not in value_columns and \
           collapse_focus_values_as != meta_column:
            raise ValueError(f"The chosen name '{collapse_focus_values_as}' for the collapsed category already exists as a column in the original dataset and is not part of the 'value_columns' or 'meta_column'. Please choose a different name.")

    # Validate collapsed_color if provided
    if collapsed_color is not None and not isinstance(collapsed_color, str):
        raise TypeError("'collapsed_color' must be a string representing a color if provided.")

    # Validate focus_colors_map if provided
    if focus_colors_map is not None and not isinstance(focus_colors_map, dict):
        raise TypeError("'focus_colors_map' must be a dictionary if provided.")

    # Validate group_by_column if provided
    if group_by_column:
        if not isinstance(group_by_column, str):
            raise TypeError("'group_by_column' must be a string if provided.")
        if group_by_column not in dataset.columns:
            raise ValueError(f"Column '{group_by_column}' not found in the dataset.")
        # Ensure that each meta_column value consistently maps to one group_by_column value
        group_consistency_check = dataset[[meta_column, group_by_column]].drop_duplicates()
        if group_consistency_check.duplicated(subset=[meta_column]).any():
            conflicting_meta_values = group_consistency_check[group_consistency_check.duplicated(subset=[meta_column])][meta_column].tolist()
            raise ValueError(f"Each value in '{meta_column}' must correspond to a single value in '{group_by_column}'. "
                             f"Conflicting '{meta_column}' values: {conflicting_meta_values}")


    # --- Data Preparation ---
    # Create a copy of the relevant columns to avoid modifying the original DataFrame
    df_plot = dataset[[meta_column] + value_columns].copy()
    if group_by_column:
        df_plot = df_plot.merge(dataset[[meta_column, group_by_column]].drop_duplicates(), on=meta_column, how='left')


    # List for columns that will be plotted on Y-axis
    cols_to_plot_y = []
    # Dictionary to map category names to their assigned colors
    category_colors_map = {}

    prop_cycle = plt.rcParams['axes.prop_cycle']
    default_colors = prop_cycle.by_key()['color']
    color_idx = 0 # Index to cycle through default colors

    # Scenario 1: Collapse focus_value into a single new category
    if collapse_focus_values_as is not None and focus_value:
        df_plot[collapse_focus_values_as] = df_plot[focus_value].sum(axis=1)
        cols_to_plot_y.append(collapse_focus_values_as)
        # Assign a color for the collapsed focus_value category
        if collapsed_color:
            category_colors_map[collapse_focus_values_as] = collapsed_color
        else:
            category_colors_map[collapse_focus_values_as] = default_colors[color_idx % len(default_colors)]
        color_idx += 1

        # Determine if there are other columns to collapse into 'others'
        remaining_value_columns_for_others = [col for col in value_columns if col not in focus_value]
        if remaining_value_columns_for_others:
            df_plot['others'] = df_plot[remaining_value_columns_for_others].sum(axis=1)
            cols_to_plot_y.append('others')
            category_colors_map['others'] = 'gray' # 'others' is typically gray

    # Scenario 2: Plot focus_value individually and collapse remaining to 'others'
    elif focus_value:
        cols_to_plot_y.extend(focus_value)
        # Assign individual colors for each focus_value column
        for col in focus_value:
            # Use color from focus_colors_map if available, otherwise use default cycle
            if focus_colors_map and col in focus_colors_map:
                category_colors_map[col] = focus_colors_map[col]
            else:
                category_colors_map[col] = default_colors[color_idx % len(default_colors)]
                color_idx += 1 # Only advance default color index if default color is used

        # Identify columns to be collapsed into "others"
        others_cols = [col for col in value_columns if col not in focus_value]
        if others_cols:
            df_plot['others'] = df_plot[others_cols].sum(axis=1)
            cols_to_plot_y.append('others')
            category_colors_map['others'] = 'gray' # 'others' is typically gray

    # Scenario 3: No focus_value specified, plot all value_columns individually
    else:
        cols_to_plot_y = value_columns
        # Assign individual colors for all value_columns
        for col in value_columns:
            category_colors_map[col] = default_colors[color_idx % len(default_colors)]
            color_idx += 1

    # Select the columns for the final DataFrame used for grouping
    df_plot_final = df_plot[[meta_column, group_by_column] + cols_to_plot_y if group_by_column else [meta_column] + cols_to_plot_y]

    # Group by the meta_column and sum the values for the Y-axis columns
    grouped_df = df_plot_final.groupby(meta_column)[cols_to_plot_y].sum()

    # --- Calculate Error Bars (if requested) ---
    grouped_std = None
    if add_error_bars:
        # Calculate standard deviation for each component within each meta_column group
        # The `cols_to_plot_y` list correctly contains the names of the columns
        # (either original, newly collapsed, or 'others') for which to calculate std.
        grouped_std = df_plot_final.groupby(meta_column)[cols_to_plot_y].std()

        # Fill NaN standard deviations (e.g., if a group has only one value) with 0
        grouped_std = grouped_std.fillna(0)


    # --- Scaling Values to Proportion ---
    # Calculate row sums for normalization. Handle cases where a row sum might be zero to avoid division by zero.
    row_sums = grouped_df.sum(axis=1)
    row_sums[row_sums == 0] = 1 # Replace 0 sums with 1 to avoid division by zero; these rows will result in 0 proportions.
    grouped_df_scaled = grouped_df.div(row_sums, axis=0).fillna(0) # Scale and fill any resulting NaNs with 0

    # --- Determine Plotting Order based on Median Contribution ---
    # Calculate the median contribution for each category across all groups
    median_contributions = grouped_df_scaled.median()

    # Sort categories based on their median contribution in descending order (for biggest at bottom)
    sorted_categories = median_contributions.sort_values(ascending=False).index.tolist()

    # Ensure 'others' is always at the very top (last position in stack)
    if 'others' in sorted_categories:
        sorted_categories.remove('others')
        sorted_categories.append('others')

    # Reorder the columns of the scaled DataFrame for plotting
    grouped_df_scaled = grouped_df_scaled[sorted_categories]

    # Reorder the colors list to match the sorted categories
    sorted_colors = [category_colors_map[cat] for cat in sorted_categories]


    # --- Handle Grouping and X-axis Ordering ---
    _internal_grouping_map = None
    if group_by_column:
        # Create a mapping from meta_column value to group_by_column value
        _internal_grouping_map = dataset.set_index(meta_column)[group_by_column].to_dict()

        # Create a DataFrame to help with sorting and indexing
        group_df_for_sort = pd.DataFrame(index=grouped_df_scaled.index)
        group_df_for_sort['group'] = group_df_for_sort.index.map(_internal_grouping_map)

        # Sort the meta_column index first by group, then by original value
        sorted_meta_values = group_df_for_sort.sort_values(by='group').index.tolist()
        grouped_df_scaled = grouped_df_scaled.loc[sorted_meta_values]

        if add_error_bars and grouped_std is not None:
             grouped_std = grouped_std.loc[sorted_meta_values, sorted_categories]


    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Calculate x-positions for bars and handle grouping
    x_positions = []
    x_labels = []
    current_x = 0
    previous_group = None
    group_label_data = [] # Stores (group_name, start_x, end_x) for brackets and labels
    group_member_counts = {} # To count members per group

    for i, meta_val in enumerate(grouped_df_scaled.index):
        if group_by_column:
            current_group = _internal_grouping_map.get(meta_val)
            # Count members in the current group
            group_member_counts[current_group] = group_member_counts.get(current_group, 0) + 1

            if previous_group is not None and current_group != previous_group:
                # Store end_x for the previous group
                group_label_data[-1] = (group_label_data[-1][0], group_label_data[-1][1], current_x - 1 - group_spacing)
                # Add spacing between groups
                current_x += group_spacing
                # Start new group data
                group_label_data.append((current_group, current_x, None))
            elif not group_label_data: # First item, start the first group
                group_label_data.append((current_group, current_x, None))
            previous_group = current_group

        x_positions.append(current_x)
        x_labels.append(meta_val)
        current_x += 1 # Standard bar width + default matplotlib spacing

    if group_by_column and group_label_data:
        # Update end_x for the last group
        group_label_data[-1] = (group_label_data[-1][0], group_label_data[-1][1], x_positions[-1])


    # Add connecting shades if requested
    if add_connecting_shades:
        cumulative_df_scaled = grouped_df_scaled.cumsum(axis=1)

        for category in sorted_categories:
            y_cumulative = cumulative_df_scaled[category].values

            if category == sorted_categories[0]: # First category, bottom is 0
                y_previous_shades = np.zeros(len(x_positions))
            else:
                prev_category_index = sorted_categories.index(category) - 1
                prev_category_name = sorted_categories[prev_category_index]
                y_previous_shades = cumulative_df_scaled[prev_category_name].values

            ax.fill_between(x_positions, y_previous_shades, y_cumulative,
                            color=category_colors_map[category], alpha=connecting_shades_alpha,
                            edgecolor=None, linewidth=0)


    # Determine border properties
    border_kwargs = {}
    if add_category_border:
        border_kwargs['edgecolor'] = 'black'
        border_kwargs['linewidth'] = category_border_width

    # Plot the stacked bars manually to control x-positions and legend
    plotted_categories_for_legend = set() # To ensure each category appears only once in legend

    for i, meta_val in enumerate(grouped_df_scaled.index):
        bottom_val = 0
        current_x_pos = x_positions[i]
        for category in sorted_categories:
            height = grouped_df_scaled.loc[meta_val, category]
            yerr_val = grouped_std.loc[meta_val, category] if add_error_bars and grouped_std is not None else 0

            # Add label only for the first bar of this category encountered across all bars
            label = category if category not in plotted_categories_for_legend else "_nolegend_"

            ax.bar(current_x_pos, height=height, bottom=bottom_val,
                   color=category_colors_map[category],
                   yerr=yerr_val if add_error_bars else None,
                   capsize=4,
                   label=label, # Assign label for legend
                   **border_kwargs)

            if category not in plotted_categories_for_legend:
                plotted_categories_for_legend.add(category) # Mark as plotted for legend

            bottom_val += height

    ax.set_xticks(x_positions)
    # X-tick labels rotated 90 degrees, centered
    ax.set_xticklabels(x_labels, rotation=90, ha='center', fontsize=xticks_fontsize) # Use xticks_fontsize
    ax.tick_params(axis='x', pad=5) # Add some padding to move labels inside if needed, adjust as desired


    # Add group brackets and labels conditionally if group_by_column is provided
    if group_by_column:

        # Add group brackets and labels
        # Use the new group_label_y_offset
        bracket_y_level = -0.05 + group_label_y_offset # Base offset + user-defined offset
        label_y_level = -0.15 + group_label_y_offset  # Base offset + user-defined offset

        for group_name, start_x, end_x in group_label_data:
            # Only draw if the group has more than one member
            if group_member_counts.get(group_name, 0) > 1: # Check group member count
                if start_x is None or end_x is None: # Skip incomplete group data
                    continue

                # Adjust start and end x to cover the full width of bars in the group
                # Bars are centered on x_positions, so each bar spans x_pos +/- 0.5
                effective_start_x = start_x - 0.5
                effective_end_x = end_x + 0.5

                # Draw left vertical line of bracket
                ax.plot([effective_start_x, effective_start_x], [bracket_y_level, bracket_y_level - 0.05],
                        color='black', transform=ax.get_xaxis_transform(), clip_on=False, linewidth=group_bracket_linewidth)
                # Draw horizontal line of bracket
                ax.plot([effective_start_x, effective_end_x], [bracket_y_level - 0.05, bracket_y_level - 0.05],
                        color='black', transform=ax.get_xaxis_transform(), clip_on=False, linewidth=group_bracket_linewidth)
                # Draw right vertical line of bracket
                ax.plot([effective_end_x, effective_end_x], [bracket_y_level, bracket_y_level - 0.05],
                        color='black', transform=ax.get_xaxis_transform(), clip_on=False, linewidth=group_bracket_linewidth)

                # Add group label
                ax.text((effective_start_x + effective_end_x) / 2, label_y_level,
                        group_name, ha='center', va='top',
                        fontsize=group_label_fontsize, rotation=group_label_rotation,
                        transform=ax.get_xaxis_transform(), clip_on=False)

    if show_title:
        plt.title(f'Stacked Bar Plot of Proportions by {meta_column}', fontsize=title_fontsize)
    if show_xlabel:
        plt.xlabel(meta_column, fontsize=xlabel_fontsize) # Use xlabel_fontsize
    if show_ylabel:
        plt.ylabel('Proportion', fontsize=ylabel_fontsize)
    plt.yticks(fontsize=yticks_fontsize)
    plt.xticks(fontsize=xticks_fontsize)
    # The legend will now correctly display all categories
    plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, title_fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # --- Saving Plot (if output path is provided) ---
    if output:
        dir_name = os.path.dirname(output)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name)

        filename_pdf = output + ".pdf"
        plt.savefig(filename_pdf)
        print(f"Stacked bar plot saved to {filename_pdf}")

        filename_png = output + ".png"
        plt.savefig(filename_png)
        print(f"Stacked bar plot saved to {filename_png}")

    plt.show()
