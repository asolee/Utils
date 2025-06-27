import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings
import matplotlib.patches as mpatches
import seaborn as sns

def create_boxplot(dataset: pd.DataFrame,
                   meta_column: str,
                   value_column: str or list,
                   hue_column: str = None,
                   meta_order: list = None,
                   hue_order: list = None,
                   color_map: dict = None,
                   add_swarmplot: bool = False,
                   swarm_size: float = 3,
                   swarm_alpha: float = 0.5,
                   swarm_color: str = 'black',
                   box_width: float = 0.6,
                   whisker_capsize: float = 0.05,
                   show_outliers: bool = True,
                   outlier_marker: str = 'o',
                   outlier_size: float = 6,
                   outlier_alpha: float = 0.6,
                   notch: bool = False,
                   median_color: str = 'black',
                   median_linewidth: float = 2,
                   mean_marker: str = 'D',
                   mean_marker_size: float = 8,
                   mean_marker_color: str = 'green',
                   show_mean: bool = False,
                   group_by_column: str = None,
                   group_position: str = 'bottom',
                   group_spacing: float = 0,
                   group_label_rotation: float = 0,
                   group_label_fontsize: int = 12,
                   group_label_y_offset: float = 0.0,
                   group_bracket_linewidth: float = 1.0,
                   group_bracket_vertical_line_length: float = 0.05,
                   fig_width: float = 10,
                   fig_height: float = 7,
                   output: str = None,
                   show_xlabel: bool = True,
                   show_ylabel: bool = True,
                   show_title: bool = True,
                   title: str = None,
                   title_pad: float = 10,
                   title_fontsize: float = 14,
                   xlabel: str = None,
                   ylabel: str = None,
                   xlabel_fontsize: float = 12,
                   xlabel_rotation: float = 90,
                   xticks_label_pad: float = 5,
                   ylabel_fontsize: float = 12,
                   xticks_fontsize: float = 10,
                   yticks_fontsize: float = 10,
                   show_group_label: bool = True,
                   legend_title: str = None,
                   legend_y_pos: float = 0.5,
                   y_upper_pad: float = 0.05):
    """
    Generates box plots

    Args:
        #### BASIC INPUT PARAMETERS ####
        dataset (pd.DataFrame): The input pandas DataFrame.
        meta_column (str): The column to be used on the X-axis for grouping.
        value_column (str or list): The column(s) containing the values for the boxplots.
                                     If a list, the dataset will be melted, and a 'variable' column
                                     will be created holding the original column names, and 'value'
                                     column holding the corresponding values.
        hue_column (str, optional): Column used to separate box plots for each category (hue value)
                                     and for coloring. If provided, overrides internal melting for coloring
                                     if {value_column} is a list.
        meta_order (list, optional): Determines the order of {meta_column} on X-axis. If None, default order
                                      is used. If grouping is used, make sure {meta_order} do not create
                                      conflicts with consistent grouping.
        hue_order (list, optional): Determines the order of hue categories within each meta_column group.
                                     If None, default order from unique values is used.
        color_map (dict, optional): A dictionary mapping category names (from hue_column or melted value_column)
                                    to specific colors. Colors can be matplotlib color names (strings) or RGBA tuples.
                                    If a category is not in the map, a default color will be used.
        output (str, optional): The path plus filename (without extension) to save the plot.
                                 If provided, the plot will be saved as a PDF and PNG.
        fig_width (float, optional): The width of the figure in inches. Defaults to 10.
        fig_height (float, optional): The height of the figure in inches. Defaults to 7.

        #### BOXPLOT STYLE PARAMETERS ####
        add_swarmplot (bool, optional): If True, adds a swarm plot on top of the box plots to show
                                         individual data points. Defaults to False.
        swarm_size (float, optional): Marker size for swarm plot points. Defaults to 3.
        swarm_alpha (float, optional): Transparency of swarm plot points. Defaults to 0.5.
        swarm_color (str, optional): Color of swarm plot points. Defaults to 'black'.
        box_width (float, optional): The width of the boxes. Defaults to 0.6.
        whisker_capsize (float, optional): The length of the whiskers' caps. Defaults to 0.05.
        show_outliers (bool, optional): If True, outliers are shown. Defaults to True.
        outlier_marker (str, optional): Marker style for outliers (according to matplotlib.markers). Defaults to 'o'.
        outlier_size (float, optional): Size of outlier markers. Defaults to 6.
        outlier_alpha (float, optional): Transparency of outlier markers. Defaults to 0.6.
        notch (bool, optional): If True, produces a notched box plot. Defaults to False.
        median_color (str, optional): Color of the median line. Defaults to 'red'.
        median_linewidth (float, optional): Line width of the median line. Defaults to 2.
        mean_marker (str, optional): Marker style for the mean (according to matplotlib.markers). Defaults to 'D' (diamond).
        mean_marker_size (float, optional): Size of the mean marker. Defaults to 8.
        mean_marker_color (str, optional): Color of the mean marker. Defaults to 'green'.
        show_mean (bool, optional): If True, the mean is marked on the box plot. Defaults to False.

        #### METADATA GROUPING ####
        group_by_column (str, optional): The name of a column in the provided dataset to use for grouping {meta_column} values.
                                          If provided, bars will be grouped based on this column's values,
                                          and group labels with brackets will be added below the x-axis. Defaults to None.
        group_spacing (float, optional): The extra space to add between groups of bars if {group_by_column} is used.
                                          Defaults to 0.
        group_label_rotation (float, optional): Rotation angle for the group labels in degrees. Defaults to 0.
        group_label_fontsize (int, optional): Font size for the group labels. Defaults to 12.
        group_label_y_offset (float, optional): Vertical offset for the group labels and brackets. A negative value
                                                 moves them further down from the x-axis. Defaults to 0.0.
        group_bracket_linewidth (float, optional): The line width for the square brackets when grouping is enabled.
                                                    Defaults to 1.0.
        group_bracket_vertical_line_length (float, optional): Controls the length of the vertical lines of the square brackets.
                                                                This value will be added to or subtracted from the base
                                                                {bracket_y_level} to define the extent of the vertical lines.
                                                                Defaults to 0.05.
        group_position (str, optional): Determines the position of group labels and brackets. Default to 'bottom'.
                                        'bottom': Below the x-axis. xlabel_rotation to 90 suggested for better visualization.
                                        'middle': Between x-ticks and x-label.
                                        'top': Above the plot, near the title.
        show_group_label (bool, optional): If True, the group labels will be displayed. Defaults to True.

        #### FONTSIZE AND VISIBILITY PARAMETERS ####
        show_xlabel (bool, optional): If True, the x-axis label will be displayed. Defaults to True.
        xlabel (str, optional): Custom label for the x-axis. If provided, overrides default. Defaults to None.
        xlabel_fontsize (float, optional): The font size for the x-axis label. Defaults to 12.
        xlabel_rotation (float, optional): Rotation angle for the x label in degrees. Defaults to 90.
        xticks_label_pad (float, optional): Distance of x-tick labels from the plot. Defaults to 5.
        show_ylabel (bool, optional): If True, the y-axis label will be displayed. Defaults to True.
        ylabel (str, optional): Custom label for the y-axis. If provided, overrides default. Defaults to None.
        ylabel_fontsize (float, optional): The font size for the y-axis label. Defaults to 12.
        show_title (bool, optional): If True, the plot title will be displayed. Defaults to True.
        title (str, optional): Custom title, if provided, overrides default. Defaults to None.
        title_fontsize (float, optional): The font size for the plot title. Defaults to 14.
        title_pad (float, optional): Distance between title and plot. Default to 10.
        xticks_fontsize (float, optional): The font size for the x-axis tick labels. Defaults to 10.
        yticks_fontsize (float, optional): The font size for the y-axis tick labels. Defaults to 10.
        legend_title (str, optional): Custom name for legend. Default is None.
        legend_y_pos (float, optional): Position of legend in Y-axis. Default 0.5.
        y_upper_pad (float, optional): Size of the space between the end of the bar and the upper plot margin. Default to 0.05 (i.e. 5%).
    """

    # ~ Input Validation ~ #
    if not isinstance(dataset, pd.DataFrame):
        raise TypeError("Input dataset must be a pandas DataFrame.")
    if meta_column not in dataset.columns:
        raise ValueError(f"Column '{meta_column}' not found in the dataset.")

    if isinstance(value_column, list):
        if not all(col in dataset.columns for col in value_column):
            raise ValueError("Not all columns in {value_column} found in the dataset.")
    elif isinstance(value_column, str):
        if value_column not in dataset.columns:
            raise ValueError(f"Column '{value_column}' not found in the dataset.")
    else:
        raise TypeError("{value_column} must be a string or a list of strings.")

    if hue_column:
        if not isinstance(hue_column, str):
            raise TypeError("{hue_column} must be a string if provided.")
        if hue_column not in dataset.columns:
            raise ValueError(f"Column '{hue_column}' not found in the dataset.")

    if color_map is not None and not isinstance(color_map, dict):
        raise TypeError("{color_map} must be a dictionary if provided.")
    if color_map is None:
        color_map = {}

    if group_by_column:
        if not isinstance(group_by_column, str):
            raise TypeError("{group_by_column} must be a string if provided.")
        if group_by_column not in dataset.columns:
            raise ValueError(f"Column '{group_by_column}' not found in the dataset.")
        group_consistency_check = dataset[[meta_column, group_by_column]].drop_duplicates()
        if group_consistency_check.duplicated(subset=[meta_column]).any():
            conflicting_meta_values = group_consistency_check[group_consistency_check.duplicated(subset=[meta_column])][meta_column].tolist()
            raise ValueError(f"Each value in '{meta_column}' must correspond to a single value in '{group_by_column}'. "
                             f"Conflicting '{meta_column}' values: {conflicting_meta_values}")

    if group_position not in ['bottom', 'middle', 'top']:
        raise ValueError("Invalid value for {group_position}. Choose from 'bottom', 'middle', or 'top'.")

    if meta_order is not None and not isinstance(meta_order, list):
        raise TypeError("Meta_order must be a list of meta_column values.")
    if meta_order is not None:
        dataset_meta_values = dataset[meta_column].unique()
        if not set(dataset_meta_values).issubset(set(meta_order)):
            missing_values = set(dataset_meta_values) - set(meta_order)
            raise ValueError(f"Not all unique values from '{meta_column}' are present in {{meta_order}}. Missing: {list(missing_values)}")
        if len(meta_order) != len(set(meta_order)):
            raise ValueError("Meta_order list must not contain duplicate values.")

    if hue_order is not None and not isinstance(hue_order, list):
        raise TypeError("Hue_order must be a list of hue category values.")

    # ~ Data Preparation ~ #
    df_plot = dataset.copy()

    # Melt the DataFrame if value_column is a list
    if isinstance(value_column, list):
        id_vars = [meta_column]
        if group_by_column:
            id_vars.append(group_by_column)
        if hue_column and hue_column not in id_vars:
            id_vars.append(hue_column)
        
        # Determine the name for the new value column and variable column
        melted_value_name = "value"
        melted_variable_name = "category"

        # Check for conflicts with existing columns
        if melted_value_name in df_plot.columns and melted_value_name not in value_column:
            raise ValueError(f"The automatically generated 'value' column name conflicts with an existing column that is not part of {value_column}. Please rename your column or choose a different 'value_column' name if melting manually.")
        if melted_variable_name in df_plot.columns and melted_variable_name not in value_column:
             raise ValueError(f"The automatically generated 'category' column name conflicts with an existing column that is not part of {value_column}. Please rename your column or choose a different 'value_column' name if melting manually.")

        df_plot = pd.melt(df_plot, id_vars=id_vars, value_vars=value_column,
                          var_name=melted_variable_name, value_name=melted_value_name)
        
        # If hue_column is not explicitly set, use the newly created 'category' column for hue
        if hue_column is None:
            hue_column = melted_variable_name
        # Update value_column to the melted value column name
        value_column = melted_value_name
    elif isinstance(value_column, str):
        # If not melting, ensure that the hue_column (if provided) is in the dataframe
        if hue_column and hue_column not in df_plot.columns:
            raise ValueError(f"The specified hue_column '{hue_column}' is not found in the dataset.")


    # Determine the column to use for coloring and legend
    color_by_column = hue_column if hue_column else meta_column

    # Get unique categories for coloring
    if color_by_column:
        unique_color_categories = df_plot[color_by_column].unique()
    else: # If no hue and not melting a list, only one category for coloring
        unique_color_categories = ['_single_category_']

    prop_cycle = plt.rcParams['axes.prop_cycle']
    default_colors = prop_cycle.by_key()['color']
    color_idx = 0
    final_color_map = {}

    for cat in unique_color_categories:
        if cat in color_map:
            final_color_map[cat] = color_map[cat]
        else:
            final_color_map[cat] = default_colors[color_idx % len(default_colors)]
            color_idx += 1

    # Ensure all meta_column values are in meta_order if provided, otherwise create default order
    if meta_order is None:
        final_meta_order = df_plot[meta_column].unique().tolist()
    else:
        # Filter meta_order to include only values present in the data
        final_meta_order = [m for m in meta_order if m in df_plot[meta_column].unique()]
        # Warn if some meta_order values were not found in the data
        if len(final_meta_order) < len(meta_order):
            missing = set(meta_order) - set(final_meta_order)
            warnings.warn(f"The following values in 'meta_order' were not found in '{meta_column}' and will be ignored: {list(missing)}")

    # Ensure hue_order if provided
    if hue_order is None and hue_column:
        final_hue_order = df_plot[hue_column].unique().tolist()
    elif hue_order is not None and hue_column:
        final_hue_order = [h for h in hue_order if h in df_plot[hue_column].unique()]
        if len(final_hue_order) < len(hue_order):
            missing = set(hue_order) - set(final_hue_order)
            warnings.warn(f"The following values in 'hue_order' were not found in '{hue_column}' and will be ignored: {list(missing)}")
    else:
        final_hue_order = [] #no hue order

    # ~ Plotting ~ #
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Calculate x-positions for boxes and handle grouping
    x_positions_meta = [] # Center of each meta group
    x_tick_labels = [] # Labels for x-axis ticks
    group_label_data_raw = [] # Stores (group_name, list_of_x_positions_of_members)
    
    current_x = 0
    box_plots_per_meta_group = len(final_hue_order) if hue_column else 1
    
    # Adjust spacing for hue
    spacing_between_boxes_in_group = 0.1 # A small fixed space between boxes
    total_width_per_meta_group = (box_plots_per_meta_group * box_width) + \
                                 ((box_plots_per_meta_group - 1) * spacing_between_boxes_in_group)
    if box_plots_per_meta_group == 1: # No spacing needed if only one box
        total_width_per_meta_group = box_width

    spacing_between_meta_groups = 1.5

    if group_by_column:
        internal_grouping_map = dataset.set_index(meta_column)[group_by_column].to_dict()
        previous_group = None
        current_group_members_x_pos = []

        if meta_order is not None:
            group_sequence = [internal_grouping_map.get(m) for m in final_meta_order]
            if group_sequence and any(group_sequence[i] != group_sequence[i-1] and group_sequence[i-1] in group_sequence[i+1:]
                                      for i in range(1, len(group_sequence))):
                warnings.warn(f"The provided 'meta_order' results in non-contiguous groups for '{group_by_column}'. "
                              "Grouping brackets might appear broken or inconsistent. "
                              "Consider reordering 'meta_order' to keep groups together.", UserWarning)

        for i, meta_val in enumerate(final_meta_order):
            current_group = internal_grouping_map.get(meta_val)

            if previous_group is not None and current_group != previous_group:
                # End the previous group
                group_label_data_raw.append((previous_group, current_group_members_x_pos))
                current_group_members_x_pos = [] # Reset for new group
                current_x += group_spacing # Add group spacing

            x_positions_meta.append(current_x + total_width_per_meta_group / 2) # Center point for the meta_val tick
            current_group_members_x_pos.append(x_positions_meta[-1])
            x_tick_labels.append(meta_val)
            current_x += total_width_per_meta_group + spacing_between_meta_groups # Always add meta group spacing
            previous_group = current_group

        # Add the last group's data
        if current_group_members_x_pos:
            group_label_data_raw.append((previous_group, current_group_members_x_pos))

    else: # No grouping
        x_positions_meta = []
        current_x = 0
        for i, meta_val in enumerate(final_meta_order):
            x_positions_meta.append(current_x + total_width_per_meta_group / 2) # Center of the meta group
            current_x += total_width_per_meta_group + spacing_between_meta_groups # Always add meta group spacing

        x_tick_labels = final_meta_order

    # Calculate final bracket coordinates from raw group data
    group_label_data = [] # Stores (group_name, start_x_bracket, end_x_bracket, should_draw_bracket)
    for group_name, x_pos_list in group_label_data_raw:
        should_draw_bracket = len(x_pos_list) > 1 # Only draw if more than one meta_column value in group
        
        if len(x_pos_list) > 1:
            # The start of the first group
            start_x_bracket = x_pos_list[0] - (total_width_per_meta_group / 2) - (spacing_between_meta_groups / 2)

            # The end of the last group
            end_x_bracket = x_pos_list[-1] + (total_width_per_meta_group / 2) + (spacing_between_meta_groups / 2)
            
        else:
            # Center it around the single point
            single_point_x = x_pos_list[0]
            start_x_bracket = single_point_x - (total_width_per_meta_group / 2)
            end_x_bracket = single_point_x + (total_width_per_meta_group / 2)
            
        group_label_data.append((group_name, start_x_bracket, end_x_bracket, should_draw_bracket))


    all_box_positions = []
    box_colors_for_legend = {} # Stores colors for legend if hue_column is used

    for i, meta_val in enumerate(final_meta_order):
        if hue_column:
            # Calculate the starting x-position for the first box within this meta_column group
            start_of_hue_group_x = x_positions_meta[i] - total_width_per_meta_group / 2

            for j, hue_val in enumerate(final_hue_order):
                subset_df = df_plot[(df_plot[meta_column] == meta_val) & (df_plot[hue_column] == hue_val)]
                
                # Calculate the exact x-position for this specific boxplot
                box_x_pos = start_of_hue_group_x + j * (box_width + spacing_between_boxes_in_group) + box_width / 2 # Center of current box

                if not subset_df.empty and not subset_df[value_column].dropna().empty: # Ensure there's data to plot
                    bp = ax.boxplot(subset_df[value_column].dropna(),
                                    positions=[box_x_pos],
                                    widths=box_width,
                                    patch_artist=True, # r
                                    notch=notch,
                                    showfliers=show_outliers,
                                    showmeans=show_mean,
                                    boxprops=dict(facecolor=final_color_map.get(hue_val, 'grey')),
                                    whiskerprops=dict(color='black', linewidth=1.5),
                                    capprops=dict(color='black', linewidth=1.5),
                                    medianprops=dict(color=median_color, linewidth=median_linewidth),
                                    meanprops=dict(marker=mean_marker, markerfacecolor=mean_marker_color,
                                                   markeredgecolor=mean_marker_color, markersize=mean_marker_size),
                                    flierprops=dict(marker=outlier_marker, markersize=outlier_size, color='black', markerfacecolor='black', alpha=outlier_alpha)
                                    )
                    # For setting the capsize on the whiskers:
                    for cap in bp['caps']:
                        cap.set_xdata(cap.get_xdata() + np.array([-whisker_capsize/2, whisker_capsize/2]))

                    # Store color for legend
                    box_colors_for_legend[hue_val] = final_color_map.get(hue_val, 'grey')

                    if add_swarmplot:
                        sns.swarmplot(x=np.full(len(subset_df), box_x_pos),
                                      y=subset_df[value_column].dropna(),
                                      ax=ax, color=swarm_color, size=swarm_size, alpha=swarm_alpha, zorder=10) # zorder to ensure points are on top

                    all_box_positions.append(box_x_pos) # For setting x-ticks later
        else: # No hue_column, single box plot per meta_column value
            subset_df = df_plot[df_plot[meta_column] == meta_val]
            box_x_pos = x_positions_meta[i] # Use pre-calculated single box position

            if not subset_df.empty and not subset_df[value_column].dropna().empty: # Ensure there's data to plot
                bp = ax.boxplot(subset_df[value_column].dropna(),
                                positions=[box_x_pos],
                                widths=box_width,
                                patch_artist=True,
                                notch=notch,
                                showfliers=show_outliers,
                                showmeans=show_mean,
                                boxprops=dict(facecolor=final_color_map.get(meta_val, 'grey')),
                                whiskerprops=dict(color='black', linewidth=1.5),
                                capprops=dict(color='black', linewidth=1.5),
                                medianprops=dict(color=median_color, linewidth=median_linewidth),
                                meanprops=dict(marker=mean_marker, markerfacecolor=mean_marker_color,
                                               markeredgecolor=mean_marker_color, markersize=mean_marker_size),
                                flierprops=dict(marker=outlier_marker, markersize=outlier_size, alpha=outlier_alpha)
                                )
                # Apply capsize to whiskers
                for cap in bp['caps']:
                    cap.set_xdata(cap.get_xdata() + np.array([-whisker_capsize/2, whisker_capsize/2]))

                # Store color for legend
                box_colors_for_legend[meta_val] = final_color_map.get(meta_val, 'grey')


                if add_swarmplot:
                    sns.swarmplot(x=np.full(len(subset_df), box_x_pos),
                                  y=subset_df[value_column].dropna(),
                                  ax=ax, color=swarm_color, size=swarm_size, alpha=swarm_alpha, zorder=10)
                all_box_positions.append(box_x_pos)

    ax.set_xticks(x_positions_meta) # Set ticks at the center of each meta_column group
    
    # Set x-tick labels and rotation
    ha_for_set_xticklabels = 'center'
    rotation_mode_for_xticklabels = None

    if xlabel_rotation == 0:
        ha_for_set_xticklabels = 'center'
    elif xlabel_rotation > 0 and xlabel_rotation < 90:
        ha_for_set_xticklabels = 'right'
        rotation_mode_for_xticklabels = 'anchor'
    elif xlabel_rotation < 0 and xlabel_rotation > -90:
        ha_for_set_xticklabels = 'left'
        rotation_mode_for_xticklabels = 'anchor'
    else:
        ha_for_set_xticklabels = 'right'
        rotation_mode_for_xticklabels = 'anchor'

    ax.set_xticklabels(x_tick_labels,
                       rotation=xlabel_rotation,
                       ha=ha_for_set_xticklabels,
                       fontsize=xticks_fontsize,
                       rotation_mode=rotation_mode_for_xticklabels)
    ax.tick_params(axis='x', pad=xticks_label_pad)

    # Add group brackets and labels
    if group_by_column:
        if group_position == 'bottom':
            bracket_horizontal_y_level = -0.05 + group_label_y_offset
            label_y_level = -0.15 + group_label_y_offset
            bracket_line_offset = -1 * group_bracket_vertical_line_length
        elif group_position == 'middle':
            bracket_horizontal_y_level = group_label_y_offset
            label_y_level = bracket_horizontal_y_level - group_bracket_vertical_line_length - 0.05
            bracket_line_offset = group_bracket_vertical_line_length
        elif group_position == 'top':
            bracket_horizontal_y_level = 1.05 + group_label_y_offset
            label_y_level = 1.15 + group_label_y_offset
            bracket_line_offset = -1 * group_bracket_vertical_line_length

        for group_name, start_x_coord, end_x_coord, should_draw_bracket in group_label_data:
            if start_x_coord is None or end_x_coord is None:
                continue

            if should_draw_bracket: # Only draw bracket if there's more than one meta_column value in the group
                ax.plot([start_x_coord, start_x_coord], [bracket_horizontal_y_level, bracket_horizontal_y_level + bracket_line_offset],
                        color='black', transform=ax.get_xaxis_transform(), clip_on=False, linewidth=group_bracket_linewidth)
                ax.plot([start_x_coord, end_x_coord], [bracket_horizontal_y_level + bracket_line_offset, bracket_horizontal_y_level + bracket_line_offset],
                        color='black', transform=ax.get_xaxis_transform(), clip_on=False, linewidth=group_bracket_linewidth)
                ax.plot([end_x_coord, end_x_coord], [bracket_horizontal_y_level, bracket_horizontal_y_level + bracket_line_offset],
                        color='black', transform=ax.get_xaxis_transform(), clip_on=False, linewidth=group_bracket_linewidth)

            if show_group_label: # Always show label if show_group_label is True
                ax.text((start_x_coord + end_x_coord) / 2, label_y_level,
                        group_name, ha='center', va='top',
                        fontsize=group_label_fontsize, rotation=group_label_rotation,
                        transform=ax.get_xaxis_transform(), clip_on=False)

    # Set y-axis limits
    min_y = df_plot[value_column].min()
    max_y = df_plot[value_column].max()
    y_range = max_y - min_y
    ax.set_ylim(min_y - y_range * 0.05, max_y + y_range * y_upper_pad)

    # Set x-axis limits based on the plotted box positions
    if all_box_positions:
        min_x_pos = min(all_box_positions)
        max_x_pos = max(all_box_positions)
        
        # Calculate padding based on a portion of the total width of a meta group
        x_padding = (total_width_per_meta_group + spacing_between_meta_groups) / 2
        ax.set_xlim(min_x_pos - x_padding, max_x_pos + x_padding)


    # Set titles and labels
    if show_title:
        if title is not None:
            plt.title(title, fontsize=title_fontsize, pad=title_pad)
        else:
            title_col = value_column if isinstance(value_column, str) else "Values"
            plt.title(f'Box Plot of {title_col} by {meta_column}', fontsize=title_fontsize, pad=title_pad)

    if show_xlabel:
        if xlabel is not None:
            plt.xlabel(xlabel, fontsize=xlabel_fontsize)
        else:
            plt.xlabel(meta_column, fontsize=xlabel_fontsize)

    if show_ylabel:
        if ylabel is not None:
            plt.ylabel(ylabel, fontsize=ylabel_fontsize)
        else:
            ylabel_text = value_column if isinstance(value_column, str) else "Value"
            plt.ylabel(ylabel_text, fontsize=ylabel_fontsize)

    plt.yticks(fontsize=yticks_fontsize)
    plt.xticks(fontsize=xticks_fontsize)

    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Create legend
    if hue_column:
        legend_handles = [mpatches.Patch(color=color, label=label) for label, color in box_colors_for_legend.items()]
        main_legend_title = legend_title if legend_title is not None else hue_column
        ax.legend(handles=legend_handles, title=main_legend_title, bbox_to_anchor=(1.05, legend_y_pos), loc='center left',
                  fontsize=10, title_fontsize=12)
    elif isinstance(value_column, list): # If melted but no explicit hue, use the melted 'category' column
        legend_handles = [mpatches.Patch(color=color, label=label) for label, color in final_color_map.items()]
        main_legend_title = legend_title if legend_title is not None else 'Category'
        ax.legend(handles=legend_handles, title=main_legend_title, bbox_to_anchor=(1.05, legend_y_pos), loc='center left',
                  fontsize=10, title_fontsize=12)
    
    plt.tight_layout()

    # ~ Saving Plot ~ #
    if output:
        dir_name = os.path.dirname(output)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name)

        filename_pdf = output + ".pdf"
        plt.savefig(filename_pdf)
        print(f"Box plot saved to {filename_pdf}")

        filename_png = output + ".png"
        plt.savefig(filename_png)
        print(f"Box plot saved to {filename_png}")

    plt.show()