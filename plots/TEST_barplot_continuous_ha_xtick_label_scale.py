import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os # Import os for directory operations

def create_stacked_barplot(dataset: pd.DataFrame, meta_column: str, value_columns: list, focus_value: list = None, xlabel_fontsize: float = 12,xticks_fontsize: float = 10,yticks_fontsize: float = 10,xlabel_rotation: float = 90,xticks_label_pad: float = 5,collapse_focus_values_as: str = None, collapsed_color: str = None, focus_colors_map: dict = None, add_error_bars: bool = False, add_connecting_shades: bool = False, connecting_shades_alpha: float = 0.15, add_category_border: bool = False, category_border_width: float = 0.5, group_by_column: str = None, group_position: str = 'bottom',group_spacing: float = 0, group_label_rotation: float = 0, group_label_fontsize: int = 12, group_label_y_offset: float = 0.0, group_bracket_linewidth: float = 1.0, group_bracket_vertical_line_length: float = 0.05, fig_width: float = 10, fig_height: float = 7, output: str = None, show_xlabel: bool = True, show_ylabel: bool = True, show_title: bool = True,title: str = None, title_fontsize: float = 14, ylabel_fontsize: float = 12, normalize_data: bool = False, scaling: str = 'none', xlabel: str = None, ylabel: str = None, show_group_label: bool = True, ha_xticks: float = None):
    """
    Generates a stacked bar plot with specified columns, collapsing, scaling, and optional error bars
    representing the raw standard deviation.

    Args:

        #### BASIC INPUT PARAMETERS ####

        dataset (pd.DataFrame): The input pandas DataFrame.
        meta_column (str): The column to be used on the X-axis (only one accepted).
        value_columns (list): A list of columns that represent the values to represent on the Y-axis of the stacked bar.
        focus_value (list, optional): A subset of value_columns to focus on, gouping the excluded ones in a "other" category.
                                       If `collapse_focus_values_as` is None, columns in focus_value
                                       will be plotted individually.
                                       If `collapse_focus_values_as` is provided, columns in focus_value
                                       will be summed into a single category.
                                       If None or an empty list, all value_columns are plotted individually
                                       without an "others" category.
        output (str, optional): The path plus filename (without extension) to save the plot.
                                 If provided, the plot will be saved as a PDF and PNG.
        fig_width (float, optional): The width of the figure in inches. Defaults to 10.
        fig_height (float, optional): The height of the figure in inches. Defaults to 7.

        #### SCALING OPTIONS ####

        scaling (str, optional): Determines how values are aggregated for each bar.
                                 'none': Values are summed for each meta_column group.
                                 'median': The median of each category's values is used for each meta_column group.
                                 Defaults to 'none'.

        #### NORMALIZE DATA ####

        normalize_data (bool, optional): If True, the stacked bars will represent proportions (summing to 1).
                                         If False, the raw counts will be plotted. Defaults to True.

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
                                               Defaults to True.
        category_border_width (float, optional): The width of the black line around categories if `add_category_border` is True.
                                                 Defaults to 0.5.

        #### METADATA GROUPING ####

        group_by_column (str, optional): The name of a column in `dataset` to use for grouping `meta_column` values.
                                          If provided, bars will be grouped based on this column's values,
                                          and group labels with brackets will be added below the x-axis. Defaults to None.
        group_spacing (float, optional): The extra space to add between groups of bars if `group_by_column` is used.
                                          Defaults to 0.
        group_label_rotation (float, optional): Rotation angle for the group labels in degrees. Defaults to 0 (the one below the brackets).
        group_label_fontsize (int, optional): Font size for the group labels. Defaults to 12.
        group_label_y_offset (float, optional): Vertical offset for the group labels and brackets. A negative value
                                                 moves them further down from the x-axis. Defaults to 0.0.
        group_bracket_linewidth (float, optional): The line width for the square brackets when grouping is enabled.
                                                    Defaults to 1.0.
        group_bracket_vertical_line_length (float, optional): Controls the length of the vertical lines of the square brackets.
                                                                This value will be added to or subtracted from the base
                                                                `bracket_y_level` to define the extent of the vertical lines.
                                                                Defaults to 0.05.
        group_position (str, optional): Determines the position of group labels and brackets. default to 'bottom'
                                        'bottom': Below the x-axis.
                                        'middle': between x-ticks and x-label
                                        'top': Above the plot, near the title.
        show_group_label (bool, optional): If True, the group labels will be displayed. Defaults to True.

        #### FONTSIZE AND VISIBILITY PARAMETERS ####

        show_xlabel (bool, optional): If True, the x-axis label will be displayed. Defaults to True.
        xlabel (str, optional): Custom label for the x-axis. If provided, overrides default. Defaults to None.
        xlabel_fontsize (float, optional): The font size for the x-axis label. Defaults to 12.
        xlabel_rotation (float, optional): Rotation angle for the x label in degrees. Defaults to 90.
        xticks_label_pad (float, optional): Distance of x-tick labels from the plot. Defaults to 5.
        ha_xticks (float, optional): Horizontal alignment for x-tick labels. 0.0 for left, 0.5 for center, 1.0 for right.
                                     For values between these, proportional shift will be attempted.
                                     Overrides automatic alignment based on `xlabel_rotation`. Defaults to None.
        show_ylabel (bool, optional): If True, the y-axis label will be displayed. Defaults to True.
        ylabel (str, optional): Custom label for the y-axis. If provided, overrides default. Defaults to None.
        ylabel_fontsize (float, optional): The font size for the y-axis label. Defaults to 12.
        show_title (bool, optional): If True, the plot title will be displayed. Defaults to True.
        title (str, optional): Custom title, if provided, overrides default. Defaults to None.
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

    # Validation for collapse_focus_values_as
    if collapse_focus_values_as is not None:
        if not isinstance(collapse_focus_values_as, str):
            raise TypeError("'collapse_focus_values_as' must be a string if provided.")
        if not focus_value:
            raise ValueError("'focus_value' cannot be empty if 'collapse_focus_values_as' is provided, as there would be nothing to collapse.")
        if collapse_focus_values_as == 'others':
            raise ValueError("'collapse_focus_values_as' cannot be named 'others' as it conflicts with an internal category name (non-focus categories).")
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
    
    if group_position not in ['bottom', 'middle', 'top']:
        raise ValueError("Invalid value for 'group_position'. Choose from 'bottom', 'middle', or 'top'.")

    # Validate scaling parameter
    if scaling not in ['none', 'median']:
        raise ValueError("Invalid value for 'scaling'. Choose from 'none' or 'median'.")

    # Validate ha_xticks (allow values outside 0-1 if we're handling them as custom offsets)
    if ha_xticks is not None and not isinstance(ha_xticks, (int, float)):
        raise TypeError("ha_xticks must be a float or int if provided.")


    # --- Data Preparation, handle focus values scenarios ---
    # Create a copy of the relevant columns to avoid modifying the original DataFrame
    df_plot = dataset[[meta_column] + value_columns].copy()
    if group_by_column:
        df_plot = df_plot.merge(dataset[[meta_column, group_by_column]].drop_duplicates(), on=meta_column, how='left')


    # List for columns that will be plotted on Y-axis
    cols_to_plot_y = []
    # Dictionary to map category names to their assigned colors
    category_colors_map = {}

    # Use default matplotlib color and allow repeating the cicle
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
                color_idx += 1

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
    df_plot_final = df_plot[[meta_column] + cols_to_plot_y]
    if group_by_column:
        df_plot_final = df_plot_final.merge(dataset[[meta_column, group_by_column]].drop_duplicates(), on=meta_column, how='left')


    # --- Calculate Error Bars (if requested) before any scaling/normalization ---
    # The standard deviation should always be calculated from the original "counts" per meta_column,
    # as this represents the variability of the data points contributing to each segment.
    grouped_std = None
    if add_error_bars:
        # Calculate standard deviation for each component within each meta_column group.
        # This will be based on the raw values for each (meta_column, category) pair.
        grouped_std = df_plot_final.groupby(meta_column)[cols_to_plot_y].std()
        grouped_std = grouped_std.fillna(0) # Fill NaN standard deviations with 0


    # --- Apply Scaling (Sum or Median) ---
    if scaling == 'none':
        grouped_df = df_plot_final.groupby(meta_column)[cols_to_plot_y].sum()
    elif scaling == 'median':
        grouped_df = df_plot_final.groupby(meta_column)[cols_to_plot_y].median()


    # --- Apply Normalization (optional) ---
    if normalize_data:
        # Calculate row sums for normalization. Handle cases where a row sum might be zero to avoid division by zero.
        row_sums = grouped_df.sum(axis=1)
        row_sums[row_sums == 0] = 1 # Replace 0 sums with 1 to avoid division by zero, these rows will result in 0 proportions.
        grouped_df_scaled = grouped_df.div(row_sums, axis=0).fillna(0) # Scale and fill NaNs with 0

        # If error bars are enabled, normalize them as well.
        if add_error_bars and grouped_std is not None:
            grouped_std_scaled = grouped_std.div(row_sums, axis=0).fillna(0)
    else:
        grouped_df_scaled = grouped_df.copy() # Use raw counts/medians if not normalizing
        if add_error_bars and grouped_std is not None:
            grouped_std_scaled = grouped_std.copy() # Use raw std if not normalizing


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

    # Reorder the std DataFrame as well if it exists
    if add_error_bars and grouped_std is not None:
        grouped_std_scaled = grouped_std_scaled[sorted_categories]

    # Reorder the colors list to match the sorted categories
    sorted_colors = [category_colors_map[cat] for cat in sorted_categories]


    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Calculate x-positions for bars and handle grouping
    x_positions = []
    x_labels = []
    current_x = 0
    previous_group = None
    group_label_data = [] # Stores (group_name, start_x, end_x) for brackets and labels
    group_member_counts = {} # To count members per group

    if group_by_column:
        # Create a mapping from meta_column value to group_by_column value
        internal_grouping_map = dataset.set_index(meta_column)[group_by_column].to_dict()

        # Create a DataFrame to help with sorting and indexing
        group_df_for_sort = pd.DataFrame(index=grouped_df_scaled.index)
        group_df_for_sort['group'] = group_df_for_sort.index.map(internal_grouping_map)

        # Sort the meta_column index first by group, then by original value
        sorted_meta_values = group_df_for_sort.sort_values(by='group').index.tolist()
        grouped_df_scaled = grouped_df_scaled.loc[sorted_meta_values]

        if add_error_bars and grouped_std is not None:
             grouped_std_scaled = grouped_std_scaled.loc[sorted_meta_values, sorted_categories]

        # Calculate x-positions for bars and handle grouping
        for i, meta_val in enumerate(grouped_df_scaled.index):
            current_group =  internal_grouping_map.get(meta_val)
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

        if group_label_data:
            # Update end_x for the last group
            group_label_data[-1] = (group_label_data[-1][0], group_label_data[-1][1], x_positions[-1])

    else: # No grouping
        x_positions = np.arange(len(grouped_df_scaled.index))
        x_labels = grouped_df_scaled.index.tolist()


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
            yerr_val = grouped_std_scaled.loc[meta_val, category] if add_error_bars and grouped_std_scaled is not None else 0

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

    # --- Custom Horizontal Alignment for X-tick Labels (ha_xticks) ---
    ha_for_set_xticklabels = 'center' # Default for matplotlib's ha parameter
    x_offset_in_data_coords = 0.0 # Additional offset to apply per label

    if ha_xticks is not None:
        if ha_xticks == 0.0:
            ha_for_set_xticklabels = 'left'
        elif ha_xticks == 0.5:
            ha_for_set_xticklabels = 'center'
        elif ha_xticks == 1.0:
            ha_for_set_xticklabels = 'right'
        else:
            # For intermediate values (e.g., 0.1 to 0.49, or 0.51 to 0.9) an offset will be applied
            # TO DO: This is not procise and might need fixing for specific fonts/text lengths.
            if ha_xticks < 0.5:
                ha_for_set_xticklabels = 'left'
                # Calculate a normalized offset (0 to 1 scale)
                normalized_offset = ha_xticks
            else: # ha_xticks > 0.5
                ha_for_set_xticklabels = 'right'
                # If ha_xticks = 0.75, normalized_offset = 0.25 (from right)
                normalized_offset = 1.0 - ha_xticks

    else:
        # automatic alignment based on xlabel_rotation
        if xlabel_rotation == 0:
            ha_for_set_xticklabels = 'center'
        elif xlabel_rotation > 0 and xlabel_rotation < 90:
            ha_for_set_xticklabels = 'right'
        elif xlabel_rotation < 0 and xlabel_rotation > -90:
            ha_for_set_xticklabels = 'left'
        else: # For 90 degrees or other angles
            ha_for_set_xticklabels = 'center' 
            
    # Set the tick labels using the determined string alignment
    # Store the returned Text objects to modify their positions later if needed
    x_tick_labels_objects = ax.set_xticklabels(x_labels, 
                                               rotation=xlabel_rotation, 
                                               ha=ha_for_set_xticklabels, 
                                               fontsize=xticks_fontsize)
    ax.tick_params(axis='x', pad=xticks_label_pad)

    # Apply fine-tuning offset if ha_xticks is not one of the standard exact values (0.0, 0.5, 1.0)
    if ha_xticks is not None and ha_xticks not in [0.0, 0.5, 1.0]:
        # force a draw to get accurate bounding box sizes for labels
        # This is important for precise text width measurement.
        fig.canvas.draw() 
        
        for i, label_obj in enumerate(x_tick_labels_objects):
            # Get the bounding box of the rendered text label in display coordinates
            bbox = label_obj.get_window_extent(renderer=fig.canvas.get_renderer())
            text_width_display = bbox.width

            # Determine the current anchor point (relative to text itself) based on ha_for_set_xticklabels
            if ha_for_set_xticklabels == 'left':
                current_anchor_point_norm = 0.0 # Left edge is at the 'x'
            elif ha_for_set_xticklabels == 'center':
                current_anchor_point_norm = 0.5 # Center is at the 'x'
            elif ha_for_set_xticklabels == 'right':
                current_anchor_point_norm = 1.0 # Right edge is at the 'x'
            else:
                current_anchor_point_norm = 0.5

            # The desired alignment on a 0-1 scale
            desired_align_norm = ha_xticks

            # Calculate the proportional shift needed in display units
            shift_proportion_of_width = desired_align_norm - current_anchor_point_norm
            dx_display_pixels = shift_proportion_of_width * text_width_display
            
            # Get the current position of the label (in data coordinates)
            current_x_data, current_y_data = label_obj.get_position()
            
            # Convert the pixel offset to data coordinates
            offset_data_x, _ = ax.transData.inverted().transform((dx_display_pixels, 0)) - ax.transData.inverted().transform((0, 0))

            # Apply the offset to the label's x position
            label_obj.set_x(current_x_data + offset_data_x)

            # to prevent unintended rotation behavior.
            label_obj.set_rotation_mode('anchor')


    # Add group brackets and labels if group_by_column is provided
    if group_by_column:
        if group_position == 'bottom':
            bracket_horizontal_y_level = -0.05 + group_label_y_offset # Base offset + user-defined offset
            label_y_level = -0.15 + group_label_y_offset  # Base offset + user-defined offset
            bracket_line_offset = -1 * group_bracket_vertical_line_length # Line goes downwards from horizontal line
        elif group_position == 'middle':
            # In 'middle' position, group_label_y_offset controls the horizontal line's position
            bracket_horizontal_y_level = group_label_y_offset 
            # The label's position is then relative to this horizontal line, considering the vertical line length
            label_y_level = bracket_horizontal_y_level - group_bracket_vertical_line_length - 0.05
            
            bracket_line_offset = group_bracket_vertical_line_length # Line goes upwards from horizontal line

        elif group_position == 'top':
            # Position above the plot, near the title.
            # These values are relative to the axis (1.0 is the top of the y-axis)
            # They might need adjustment based on typical y-axis limits and plot size.
            bracket_horizontal_y_level = 1.05 + group_label_y_offset
            label_y_level = 1.15 + group_label_y_offset
            bracket_line_offset = -1 * group_bracket_vertical_line_length # Line goes downwards from horizontal line

        for group_name, start_x, end_x in group_label_data:
            # Only draw if the group has more than one member
            if group_member_counts.get(group_name, 0) > 1: # Check group member count
                if start_x is None or end_x is None: # Skip incomplete group data
                    continue

                # Adjust start and end x to coincide with the x-tick positions
                effective_start_x = start_x - 0.5
                effective_end_x = end_x + 0.5

                # Create bracket
                # Draw left vertical line of bracket
                ax.plot([effective_start_x, effective_start_x], [bracket_horizontal_y_level, bracket_horizontal_y_level + bracket_line_offset],
                        color='black', transform=ax.get_xaxis_transform(), clip_on=False, linewidth=group_bracket_linewidth)
                # Draw horizontal line of bracket
                ax.plot([effective_start_x, effective_end_x], [bracket_horizontal_y_level + bracket_line_offset, bracket_horizontal_y_level + bracket_line_offset],
                        color='black', transform=ax.get_xaxis_transform(), clip_on=False, linewidth=group_bracket_linewidth)
                # Draw right vertical line of bracket
                ax.plot([effective_end_x, effective_end_x], [bracket_horizontal_y_level, bracket_horizontal_y_level + bracket_line_offset],
                        color='black', transform=ax.get_xaxis_transform(), clip_on=False, linewidth=group_bracket_linewidth)

                # Add group label conditionally
                if show_group_label:
                    ax.text((effective_start_x + effective_end_x) / 2, label_y_level,
                            group_name, ha='center', va='top',
                            fontsize=group_label_fontsize, rotation=group_label_rotation,
                            transform=ax.get_xaxis_transform(), clip_on=False)

    # Basic label and font-size setting
    title_suffix = ""
    if scaling == 'none':
        title_suffix += "Counts"
    elif scaling == 'median':
        title_suffix += "Medians"

    if normalize_data:
        title_suffix = "Proportions" # Overwrite if normalized

    if show_title:
        if title is not None:
            plt.title(title, fontsize=title_fontsize)
        else:
            plt.title(f'Stacked Bar Plot of {title_suffix} by {meta_column}', fontsize=title_fontsize)

    # Set x-axis label
    if show_xlabel:
        if xlabel is not None:
            plt.xlabel(xlabel, fontsize=xlabel_fontsize)
        else:
            plt.xlabel(meta_column, fontsize=xlabel_fontsize) # Use xlabel_fontsize

    # Set y-axis label
    if show_ylabel:
        if ylabel is not None:
            plt.ylabel(ylabel, fontsize=ylabel_fontsize)
        else:
            ylabel_text = "Proportion" if normalize_data else ( "Count" if scaling == 'none' else "Median Value" )
            plt.ylabel(ylabel_text, fontsize=ylabel_fontsize)

    plt.yticks(fontsize=yticks_fontsize)
    plt.xticks(fontsize=xticks_fontsize)
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
