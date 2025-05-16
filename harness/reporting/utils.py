import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Union

import matplotlib.pyplot as plt
import pandas as pd
import pandas.io.formats.style
from scipy import stats
import os
import seaborn as sns
import glob
import numpy as np
import matplotlib.patches as mpatches


#### Utility functions to build master dataframe ####


def load_metric_file(file_path: Path) -> dict[str, any]:
    """
    Load a single metric file and return its contents

    Parameters
    ----------
    file_path : Path
        Path to the metric file

    Returns
    -------
    dict[str, any]
        dictionary with metric name as key and test_score as value
    """
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Extract metric names and test scores
    metrics = {}
    for metric_data in data:
        metric_name = metric_data['class_name']
        test_scores = metric_data['test_scores']
        # Calculate mean of test scores if available
        if test_scores:
            metrics[metric_name] = np.mean(test_scores)
        else:
            metrics[metric_name] = None

    return metrics


def find_paired_results(directory: Path) -> dict[dict[str, Path]]:
    """
    Find pairs of top/bottom metric files for each task. Assumes input is a model result dir.

    Parameters
    ----------
    directory : Path
        Directory containing metric files

    Returns
    -------
    dict[ dict[str, Path]]
        dictionary with task identifiers to a dict of (top: top_metric_path, bottom_metric_path)
    """
    paired_metrics = {}

    # Get all metric files
    metric_files = [f for f in directory.iterdir() if f.suffix == '.metrics']

    # for each file establish its task-filter-split-taxa-label scheme
    task_pairs = defaultdict(dict)
    for metric_file in metric_files:
        try:
            model_name, task_spec = metric_file.stem.split('_')
            task_class, domain, split, taxa_level, *label_set = task_spec.split('-')

        except:
            print(f'Cant extract task info from name: {metric_file.stem}')
            continue

        exact_task = '-'.join([task_class, domain, taxa_level])
        for elem in label_set:
            exact_task += '-' + str(elem)
        task_pairs[exact_task][split] = metric_file

    return task_pairs


def calculate_metrics_stats(
    task_pairs: dict[str, dict[str, Path]],
) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    """
    Load metric files and calculate mean and standard deviation for each metric.

    Parameters
    ----------
    task_pairs : dict[str, dict[str, Path]]
        dictionary with task identifiers mapped to top/bottom metric file paths

    Returns
    -------
    dict[str, dict[str, dict[str, dict[str, float]]]]
        dictionary with structure:
        {
            task_id: {
                'top': {
                    metric_name: {
                        'train_mean': float, 'train_std': float,
                        'test_mean': float, 'test_std': float
                    }
                },
                'bottom': {
                    metric_name: {
                        'train_mean': float, 'train_std': float,
                        'test_mean': float, 'test_std': float
                    }
                }
            }
        }
    """
    results = {}

    for task_id, splits in task_pairs.items():
        results[task_id] = {}

        for split_name, metric_file in splits.items():
            try:
                # Load the JSON data
                with open(metric_file, 'r') as f:
                    metrics_data = json.load(f)

                # Process each metric
                split_results = {}
                for metric in metrics_data:
                    metric_name = metric['class_name']

                    # Calculate mean and std for train and test scores
                    train_scores = np.array(metric['train_scores'])
                    test_scores = np.array(metric['test_scores'])

                    split_results[metric_name] = {
                        'train_mean': float(np.mean(train_scores)),
                        'train_std': float(np.std(train_scores)),
                        'test_mean': float(np.mean(test_scores)),
                        'test_std': float(np.std(test_scores)),
                        'is_higher_better': metric['is_higher_better'],
                        'k-fold': train_scores.shape[0],
                        'train_scores': train_scores,
                        'test_scores': test_scores,
                    }

                results[task_id][split_name] = split_results

            except Exception as e:
                print(f'Error processing {metric_file}: {e}')
                continue

    return results


def calculate_representation_gaps(
    task_stats: dict[str, dict[str, dict[str, dict[str, float]]]],
) -> dict[str, dict[str, dict[str, float]]]:
    """
    Calculate representation gaps between top and bottom performance,
    incorporating standard deviation information.

    Parameters
    ----------
    task_stats : dict
        Task statistics from calculate_metrics_stats

    Returns
    -------
    dict[str, dict[str, dict[str, float]]]
        Dictionary with representation gap metrics for each task and metric
    """
    representation_gaps = {}

    for task_id, splits in task_stats.items():
        if 'top' not in splits or 'bottom' not in splits:
            print(f'Missing top or bottom split for task {task_id}')
            continue

        representation_gaps[task_id] = {}

        # Find common metrics between top and bottom
        common_metrics = set(splits['top'].keys()) & set(splits['bottom'].keys())

        for metric_name in common_metrics:
            top_mean = splits['top'][metric_name]['test_mean']
            top_std = splits['top'][metric_name]['test_std']
            bottom_mean = splits['bottom'][metric_name]['test_mean']
            bottom_std = splits['bottom'][metric_name]['test_std']
            is_higher_better = splits['top'][metric_name]['is_higher_better']
            n = splits['top'][metric_name]['k-fold']

            # Calculate various representation gap metrics

            # 1. Raw difference (directionality based on is_higher_better)
            if is_higher_better:
                raw_diff = top_mean - bottom_mean
            else:
                raw_diff = bottom_mean - top_mean

            # 2. Standardized difference (Cohen's d) - effect size
            # Uses pooled standard deviation
            pooled_std = np.sqrt((top_std**2 + bottom_std**2) / 2)
            cohens_d = raw_diff / pooled_std if pooled_std != 0 else 0

            # 3. Percentage difference relative to bottom
            percent_diff = (
                (raw_diff / abs(bottom_mean)) * 100 if bottom_mean != 0 else 0
            )

            # 4. Z-score based on uncertainties (assumes independence)
            combined_variance = top_std**2 + bottom_std**2
            z_score = (
                raw_diff / np.sqrt(combined_variance) if combined_variance > 0 else 0
            )

            # p values and t stat
            # For the degrees of freedom calculation
            if top_std == 0 or bottom_std == 0:
                dof = float('nan')
                p_value = float('nan')
                t_stat = float('nan')
            else:
                # Only calculate these if we have valid standard deviations
                dof = ((top_std**2 / n + bottom_std**2 / n) ** 2) / (
                    (top_std**2 / n) ** 2 / (n - 1) + (bottom_std**2 / n) ** 2 / (n - 1)
                )
                t_stat = raw_diff / np.sqrt((top_std**2 / n) + (bottom_std**2 / n))
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), dof))  # Two-tailed p-value

            # Store all metrics
            representation_gaps[task_id][metric_name] = {
                'raw_difference': raw_diff,
                'cohens_d': cohens_d,
                'percent_difference': percent_diff,
                'z_score': z_score,
                'top_mean': top_mean,
                'top_std': top_std,
                'bottom_mean': bottom_mean,
                'bottom_std': bottom_std,
                'p_value': p_value,
                't_statistic': t_stat,
                'top_train_scores': splits['top'][metric_name]['train_scores'],
                'top_test_scores': splits['top'][metric_name]['test_scores'],
                'bottom_train_scores': splits['bottom'][metric_name]['train_scores'],
                'bottom_test_scores': splits['bottom'][metric_name]['test_scores'],
            }

    return representation_gaps


#### Building master dataframe ####
# This function uses above utility functions to iterate through my dir structure and find all the results I have compiled so far


def build_master_dataframe(results_dir: Path) -> pd.DataFrame:
    """
    Build a comprehensive dataframe containing all representation gap metrics
    across all splits, models, and downstream model types.

    Parameters
    ----------
    results_dir : Path
        Root directory containing all result folders

    Returns
    -------
    pd.DataFrame
        Master dataframe with all results
    """
    all_data: list[dict[str, Any]] = []

    # Discover all downstream model types (like "svm-evals", "linear-evals", "mlp-evals", etc.)
    downstream_model_dirs = [
        d for d in results_dir.iterdir() if d.is_dir() and d.name.endswith('-evals')
    ]

    if not downstream_model_dirs:
        print(f'No downstream model directories found in {results_dir}')
        return

    # Process each downstream model type
    for downstream_dir in downstream_model_dirs:
        downstream_model_type = downstream_dir.name.replace('-evals', '')

        # Iterate through all backbone model directories
        for model_dir in downstream_dir.iterdir():
            if not model_dir.is_dir():
                continue

            model_name = model_dir.name
            print(f'Processing {downstream_model_type} - {model_name}')

            # Find paired results for this model
            paired_results = find_paired_results(model_dir)

            # Calculate statistics
            task_stats = calculate_metrics_stats(paired_results)

            # Calculate representation gaps
            rep_gaps = calculate_representation_gaps(task_stats)

            # Convert to dataframe rows
            for task_id, metrics in rep_gaps.items():
                # Parse task info from task_id
                task_parts = task_id.split('-')
                task_class = task_parts[0]
                task_filter = task_parts[1]
                taxa_level = task_parts[2]
                label_set = '-'.join(task_parts[3:]) if len(task_parts) > 3 else ''
                if task_class == 'PfamTaxonomyBias' and label_set == '':
                    label_set = 'pfam'

                for metric_name, gap_metrics in metrics.items():
                    # Create a row for each task-metric combination
                    row = {
                        'downstream_model_type': downstream_model_type,
                        'model': model_name,
                        'task_class': task_class,
                        'task_filter': task_filter,
                        'taxa_level': taxa_level,
                        'label_set': label_set,
                        'metric': metric_name,
                        **gap_metrics,  # Unpack all gap metrics (raw_difference, cohens_d, etc.)
                    }
                    all_data.append(row)

    # Convert to DataFrame
    master_df = pd.DataFrame(all_data)

    return master_df


#### Summaries and visualization functions ####


def create_summary_table(
    master_df: pd.DataFrame,
    downstream_model_type: str,
    task_metric: str = 'F1',
    gap_metric: str = 'raw_difference',
    task_class: str | None = None,
    model_order: list[str] | None = None,
    task_order: list[str] | None = None,
    include_mean_gap: bool = True,
) -> pd.DataFrame:
    """
    Create a summary table with models as rows, tasks as columns, and gap metrics as values.
    If model_order or task_order are provided, only those models/tasks will be included.

    Parameters
    ----------
    master_df : pd.DataFrame
        Master dataframe containing all results
    downstream_model_type : str
        Downstream model type to subset by (e.g., 'svm')
    task_metric : str, optional
        Evaluated metric to report (e.g., 'Accuracy', 'F1'), by default "F1"
    gap_metric : str, optional
        Gap metric to display in cells (e.g., 'raw_difference', 'cohens_d'), by default "raw_difference"
    task_class : str, optional
        Filter by specific task class (e.g., 'PfamTaxonomyBias'), by default None
    model_order : list[str], optional
        Specific order and selection of models (rows), by default None
    task_order : list[str], optional
        Specific order and selection of tasks (columns), by default None
    include_mean_gap : bool, optional
        Whether to include a Mean_Gap column, by default True

    Returns
    -------
    pd.DataFrame
        Pivoted summary table with models as rows and tasks as columns
    """
    # Filter the master dataframe
    filtered_df = master_df[
        (master_df['downstream_model_type'] == downstream_model_type)
        & (master_df['metric'] == task_metric)
    ]

    # copy filtered DF to avoid SettingWithCopyWarning
    filtered_df = filtered_df.copy()

    if task_class:
        filtered_df = filtered_df[filtered_df['task_class'] == task_class]

    # Create a task identifier column
    filtered_df['task_identifier'] = filtered_df.apply(
        lambda row: f'{row["task_class"]}-{row["taxa_level"]}'
        + (f'-{row["label_set"]}' if row['label_set'] else ''),
        axis=1,
    )

    # Filter by models if model_order is provided
    if model_order is not None:
        filtered_df = filtered_df[filtered_df['model'].isin(model_order)]

    # Filter by tasks if task_order is provided
    if task_order is not None:
        filtered_df = filtered_df[filtered_df['task_identifier'].isin(task_order)]

    # Pivot the dataframe
    pivot_table = filtered_df.pivot_table(
        index='model',
        columns='task_identifier',
        values=gap_metric,
        aggfunc='mean',  # In case there are duplicate entries
    )

    # Add a summary column showing the mean gap across all tasks
    if include_mean_gap:
        pivot_table['Mean_Gap'] = pivot_table.mean(axis=1)

    # Apply ordering for models
    if model_order is not None:
        # First get the models that actually exist in the data
        available_models = set(pivot_table.index)

        # Keep only models from model_order that exist in the data, preserving order
        ordered_models = [model for model in model_order if model in available_models]

        # Reindex the pivot table
        if ordered_models:
            pivot_table = pivot_table.loc[ordered_models]
    else:
        # If no model_order is provided, sort by Mean_Gap if available
        if include_mean_gap:
            pivot_table = pivot_table.sort_values('Mean_Gap', ascending=False)

    # Apply ordering for tasks
    if task_order is not None:
        # First get the tasks that actually exist in the data
        available_tasks = set(pivot_table.columns)

        # Keep only tasks from task_order that exist in the data, preserving order
        ordered_tasks = [task for task in task_order if task in available_tasks]

        # Add Mean_Gap at the end if it exists and should be included
        if (
            include_mean_gap
            and 'Mean_Gap' in available_tasks
            and 'Mean_Gap' not in ordered_tasks
        ):
            ordered_tasks.append('Mean_Gap')

        # Reindex the columns
        if ordered_tasks:
            pivot_table = pivot_table[ordered_tasks]

    return pivot_table


def styled_summary_table(summary_table: pd.DataFrame, gap_metric: str):
    """
    Apply conditional formatting to the summary table while preserving order.

    Parameters
    ----------
    summary_table : pd.DataFrame
        The pivoted summary table
    gap_metric : str
        The gap metric being displayed

    Returns
    -------
    Styled version of the summary table
    """
    # Create a copy to avoid modifying the original
    table_to_style = summary_table.copy()

    # Get the exact column and index ordering
    columns = summary_table.columns.tolist()
    index = summary_table.index.tolist()

    # Apply styling while preserving the original order
    styled = table_to_style.style

    # Determine what formatting to apply based on gap metric
    if gap_metric == 'cohens_d':
        # For Cohen's d, use a diverging color scheme centered at zero
        # Blue for negative, white for zero, red for positive
        non_mean_columns = [col for col in columns if col != 'Mean_Gap']
        mean_column = ['Mean_Gap'] if 'Mean_Gap' in columns else []

        # Find the maximum absolute value to create a symmetric scale
        if len(non_mean_columns) > 0:
            max_abs_value = max(
                abs(summary_table[non_mean_columns].min().min()),
                abs(summary_table[non_mean_columns].max().max()),
            )
        else:
            max_abs_value = 1.0  # Default if no data

        # Ensure we have some range for the colormap
        max_abs_value = max(max_abs_value, 0.001)

        if non_mean_columns:
            styled = styled.background_gradient(
                cmap='RdBu_r',  # Red-White-Blue diverging colormap (reversed)
                subset=pd.IndexSlice[index, non_mean_columns],
                vmin=-max_abs_value,  # Set minimum value for blue
                vmax=max_abs_value,  # Set maximum value for red
                axis=None,  # Apply to entire subset, not row-wise or column-wise
            )

        if mean_column:
            # For Mean_Gap, determine its own range
            if summary_table['Mean_Gap'].notna().any():
                mean_max_abs = max(
                    abs(summary_table['Mean_Gap'].min()),
                    abs(summary_table['Mean_Gap'].max()),
                )
                mean_max_abs = max(mean_max_abs, 0.001)
            else:
                mean_max_abs = max_abs_value

            styled = styled.background_gradient(
                cmap='RdBu_r',
                subset=pd.IndexSlice[index, mean_column],
                vmin=-mean_max_abs,
                vmax=mean_max_abs,
                axis=None,
            )

    elif gap_metric == 'z_score':
        # For z-scores, use the same diverging color scheme as Cohen's d
        non_mean_columns = [col for col in columns if col != 'Mean_Gap']
        mean_column = ['Mean_Gap'] if 'Mean_Gap' in columns else []

        # Find the maximum absolute value to create a symmetric scale
        if len(non_mean_columns) > 0:
            max_abs_value = max(
                abs(summary_table[non_mean_columns].min().min()),
                abs(summary_table[non_mean_columns].max().max()),
            )
        else:
            max_abs_value = 1.0  # Default if no data

        # Ensure we have some range for the colormap
        max_abs_value = max(max_abs_value, 0.001)

        if non_mean_columns:
            styled = styled.background_gradient(
                cmap='RdBu_r',  # Red-White-Blue diverging colormap (reversed)
                subset=pd.IndexSlice[index, non_mean_columns],
                vmin=-max_abs_value,
                vmax=max_abs_value,
                axis=None,
            )

        if mean_column:
            # For Mean_Gap, determine its own range
            if summary_table['Mean_Gap'].notna().any():
                mean_max_abs = max(
                    abs(summary_table['Mean_Gap'].min()),
                    abs(summary_table['Mean_Gap'].max()),
                )
                mean_max_abs = max(mean_max_abs, 0.001)
            else:
                mean_max_abs = max_abs_value

            styled = styled.background_gradient(
                cmap='RdBu_r',
                subset=pd.IndexSlice[index, mean_column],
                vmin=-mean_max_abs,
                vmax=mean_max_abs,
                axis=None,
            )

    elif gap_metric == 'percent_difference':
        # For percentage differences
        styled = styled.background_gradient(
            cmap='RdYlGn_r', subset=pd.IndexSlice[index, columns]
        )

    elif gap_metric == 'raw_difference':
        # For raw differences, use a diverging color scheme centered at zero
        # Blue for negative, white for zero, red for positive
        non_mean_columns = [col for col in columns if col != 'Mean_Gap']
        mean_column = ['Mean_Gap'] if 'Mean_Gap' in columns else []

        # Find the maximum absolute value to create a symmetric scale
        if len(non_mean_columns) > 0:
            max_abs_value = max(
                abs(summary_table[non_mean_columns].min().min()),
                abs(summary_table[non_mean_columns].max().max()),
            )
        else:
            max_abs_value = 1.0  # Default if no data

        # Ensure we have some range for the colormap
        max_abs_value = max(max_abs_value, 0.001)

        if non_mean_columns:
            styled = styled.background_gradient(
                cmap='RdBu_r',  # Red-White-Blue diverging colormap (reversed)
                subset=pd.IndexSlice[index, non_mean_columns],
                vmin=-max_abs_value,  # Set minimum value for blue
                vmax=max_abs_value,  # Set maximum value for red
                axis=None,  # Apply to entire subset, not row-wise or column-wise
            )

        if mean_column:
            # For Mean_Gap, determine its own range
            if summary_table['Mean_Gap'].notna().any():
                mean_max_abs = max(
                    abs(summary_table['Mean_Gap'].min()),
                    abs(summary_table['Mean_Gap'].max()),
                )
                mean_max_abs = max(mean_max_abs, 0.001)
            else:
                mean_max_abs = max_abs_value

            styled = styled.background_gradient(
                cmap='RdBu_r',
                subset=pd.IndexSlice[index, mean_column],
                vmin=-mean_max_abs,
                vmax=mean_max_abs,
                axis=None,
            )

    elif gap_metric == 't_statistic':
        # For t-statistics: Red for high absolute values (more significant differences)
        # Use absolute values for the gradient to highlight magnitude
        abs_values = summary_table.abs()

        non_mean_columns = [col for col in columns if col != 'Mean_Gap']
        mean_column = ['Mean_Gap'] if 'Mean_Gap' in columns else []

        if non_mean_columns:
            styled = styled.background_gradient(
                cmap='YlOrRd',  # Yellow-Orange-Red: higher values are more intense red
                subset=pd.IndexSlice[index, non_mean_columns],
                vmin=0,  # Start from 0
                vmax=max(
                    4.0, abs_values[non_mean_columns].max().max()
                ),  # Cap at 4.0 or max value
            )

        if mean_column:
            styled = styled.background_gradient(
                cmap='YlOrRd',
                subset=pd.IndexSlice[index, mean_column],
                vmin=0,
                vmax=max(4.0, abs_values[mean_column].max().max()),
            )

    elif gap_metric == 'p_value':
        # For p-values: Green for significant p-values (low values)

        # Convert the dataframe to object type to avoid dtype warnings
        formatted_p_values = pd.DataFrame(
            index=table_to_style.index, columns=table_to_style.columns, dtype='object'
        )

        # Define a function to highlight significant p-values
        def highlight_significant(val):
            if pd.isna(val):
                return ''

            if isinstance(val, str):
                if val == 'p < 0.001':
                    return 'background-color: #1B7837'  # Dark green
                elif val == 'p < 0.01':
                    return 'background-color: #5AAE61'  # Medium green
                elif val == 'p < 0.05':
                    return 'background-color: #A6DBA0'  # Light green
                else:
                    return 'background-color: #F7F7F7'  # Light gray

            # If it's a number, try to interpret it
            try:
                p_val = float(val)
                if p_val < 0.001:
                    return 'background-color: #1B7837'
                elif p_val < 0.01:
                    return 'background-color: #5AAE61'
                elif p_val < 0.05:
                    return 'background-color: #A6DBA0'
                else:
                    return 'background-color: #F7F7F7'
            except (ValueError, TypeError):
                return ''

        # Format p-values
        for col in columns:
            for idx in index:
                val = table_to_style.loc[idx, col]
                if pd.notnull(val):
                    try:
                        p_val = float(val)
                        if p_val < 0.001:
                            formatted_p_values.loc[idx, col] = 'p < 0.001'
                        elif p_val < 0.01:
                            formatted_p_values.loc[idx, col] = 'p < 0.01'
                        elif p_val < 0.05:
                            formatted_p_values.loc[idx, col] = 'p < 0.05'
                        else:
                            formatted_p_values.loc[idx, col] = f'{p_val:.3f}'
                    except (ValueError, TypeError):
                        formatted_p_values.loc[idx, col] = str(val)

        # Apply the styling function to the formatted values
        styled = formatted_p_values.style.applymap(highlight_significant)

        # Return early for p-values since we've already handled formatting
        return styled

    else:
        # For other metrics, use default red-yellow-green reversed
        styled = styled.background_gradient(
            cmap='RdYlGn_r', subset=pd.IndexSlice[index, columns]
        )

    # Format numbers based on the metric (only if we haven't already formatted for p-values)
    if gap_metric in ['percent_difference']:
        styled = styled.format('{:.1f}%')
    elif gap_metric in ['raw_difference']:
        styled = styled.format('{:.2%}')
    elif gap_metric in ['cohens_d', 'z_score', 't_statistic']:
        styled = styled.format('{:.2f}')
    elif gap_metric != 'p_value':  # Skip formatting if we've already formatted p-values
        styled = styled.format('{:.3f}')

    return styled


def save_table_latex(
    summary_table: pd.DataFrame,
    output_file: Path,
    caption: str = '',
    label: str = '',
    transpose: bool = True,
) -> str:
    """
    Create a LaTeX-formatted table from the summary data using pandas to_latex.

    Parameters
    ----------
    summary_table : pd.DataFrame
        The summary table with models as rows and tasks as columns
    output_file: Path
        Path to the output for the table
    caption : str, optional
        LaTeX table caption, by default ""
    label : str, optional
        LaTeX table label for cross-references, by default ""
    transpose : bool, optional
        Whether to transpose the table (tasks as rows), by default True

    Returns
    -------
    str
        LaTeX code for the table
    """
    # Make a copy to avoid modifying the original
    table = summary_table.copy()

    # Transpose if requested
    if transpose:
        table = table.transpose()

    # Generate LaTeX table
    latex_code = table.to_latex(
        buf=output_file,
        float_format='%.3f',
        caption=caption,
        label=label,
        bold_rows=True,  # Bold the row labels
        longtable=False,  # Regular table environment
        position='ht',  # Position hint for LaTeX
        multicolumn=True,  # Allow multicolumn headers
        multicolumn_format='c',  # Center multicolumn headers
        column_format='l'
        + 'r' * len(table.columns),  # Left-align first column, right-align numbers
        na_rep='--',  # Representation for NaN values
        escape=False,  # Allow LaTeX commands in cell values
    )

    return latex_code


def format_styled_latex(latex_code):
    """
    Convert LaTeX table with background-color syntax to HTML hex color format and wrap in adjustbox.

    This function processes a LaTeX table, converting background-color and color syntax
    to LaTeX's cellcolor with HTML hex format, and wraps the tabular environment in an adjustbox
    environment to control the width, while preserving the table environment (including caption and label).

    Required LaTeX packages:
    -----------------------
    \\usepackage[table,svgnames,dvipsnames,x11names]{xcolor}
    \\usepackage{colortbl}
    \\usepackage{adjustbox}
    \\definecolor{whitetext}{HTML}{F1F1F1}

    Parameters
    ----------
    latex_code : str
        The input LaTeX code containing a table with background-color syntax

    Returns
    -------
    str
        The converted LaTeX code with proper cellcolor[HTML] syntax and adjustbox environment

    Examples
    --------
    >>> latex = r'''\\begin{table}
    ... \\caption{My Table}
    ... \\label{tab:mytable}
    ... \\begin{tabular}{lrr}...\\end{tabular}
    ... \\end{table}'''
    >>> converted = convert_latex_table_colors(latex)
    >>> print(converted)
    \\begin{table}
    \\caption{My Table}
    \\label{tab:mytable}
    \\begin{adjustbox}{width=0.98\\textwidth,center}
    \\begin{tabular}{lrr}...\\end{tabular}
    \\end{adjustbox}
    \\end{table}
    """
    # Regular expression to find each cell with background color and text color
    cell_regex = (
        r'\\background-color#([a-fA-F0-9]{6}) \\color#([a-fA-F0-9]{6}) ([^&\\]+)'
    )

    # Function to process each match
    def replace_cell(match):
        bg_hex = match.group(1).upper()
        text_hex = match.group(2)
        value = match.group(3)

        # Escape percentage signs in the value to prevent LaTeX comments
        value = value.replace('%', '\\%')

        # Check if text should be white (for dark backgrounds)
        white_text = text_hex == 'f1f1f1' or text_hex == 'F1F1F1'

        if white_text:
            return f'\\cellcolor[HTML]{{{bg_hex}}}\\color{{whitetext}} {value}'
        else:
            return f'\\cellcolor[HTML]{{{bg_hex}}} {value}'

    # Replace the color syntax
    corrected_latex = re.sub(cell_regex, replace_cell, latex_code)

    # Check if there's a table environment
    table_start = corrected_latex.find('\\begin{table}')
    table_end = corrected_latex.find('\\end{table}') + len('\\end{table}')

    # Find the tabular environment
    tabular_start = corrected_latex.find('\\begin{tabular}')
    tabular_end = corrected_latex.find('\\end{tabular}') + len('\\end{tabular}')

    if tabular_start == -1 or tabular_end == -1:
        return corrected_latex  # No tabular environment found

    # Extract the tabular part
    tabular_part = corrected_latex[tabular_start:tabular_end]

    # Check if we're in a table environment
    if (
        table_start != -1
        and table_end != -1
        and table_start < tabular_start
        and tabular_end < table_end
    ):
        # We have a table environment - preserve it
        pre_tabular = corrected_latex[table_start:tabular_start]
        post_tabular = corrected_latex[tabular_end:table_end]

        # Create the result with adjustbox inside the table environment
        result = pre_tabular
        result += '\\begin{adjustbox}{width=0.98\\textwidth,center}\n'
        result += tabular_part
        result += '\n\\end{adjustbox}'
        result += post_tabular
    else:
        # No table environment - just wrap tabular with adjustbox
        result = '\\begin{adjustbox}{width=0.98\\textwidth,center}\n'
        result += tabular_part
        result += '\n\\end{adjustbox}'

    return result


def aggregate_precomputed_mean_gaps(
    summary_tables: list[tuple[str, pd.DataFrame]],
) -> pd.DataFrame:
    """
    Aggregate precomputed summary tables into a single DataFrame. Each element in
    summary_tables is a tuple containing the task name (or identifier) and the corresponding
    summary DataFrame which contains a 'Mean_Gap' column.

    Parameters
    ----------
    summary_tables : list of tuples (str, pd.DataFrame)
        Each tuple should be of the form (task_name, summary_table)

    Returns
    -------
    aggregated : pd.DataFrame
        A DataFrame with models as the index and each column corresponding to a task's Mean_Gap.
    """
    aggregated = pd.DataFrame()

    for task_name, table in summary_tables:
        if 'Mean_Gap' not in table.columns:
            raise ValueError(
                f"Summary table for task '{task_name}' does not contain a 'Mean_Gap' column."
            )

        # If aggregated is empty, initialize the index using the models from the first summary table.
        if aggregated.empty:
            aggregated.index = table.index
        # Extract the Mean_Gap column and rename it with the task name.
        aggregated[task_name] = table['Mean_Gap']

    return aggregated


def style_aggregated_table(
    agg_df: pd.DataFrame, gap_metric: str = 'raw_difference'
) -> pd.io.formats.style.Styler:
    """
    Apply a red-blue diverging color gradient to the aggregated DataFrame.
    Negative gaps will be displayed in blue and positive gaps in red.

    Parameters
    ----------
    agg_df : pd.DataFrame
        Aggregated table with models as index and tasks as columns.
    gap_metric : str, optional
        The metric to base the styling on; default is "cohens_d".

    Returns
    -------
    styled : pd.io.formats.style.Styler
        The styled DataFrame with conditional formatting.
    """
    max_abs_value = agg_df.abs().max().max()
    max_abs_value = max(max_abs_value, 0.001)

    styled = agg_df.style.background_gradient(
        cmap='RdBu_r', vmin=-max_abs_value, vmax=max_abs_value, axis=None
    )

    styled = styled.format('{:.2f}')
    return styled


#### Plotting domain specific performances ####


def aggregate_domain_results(
    *dfs: pd.DataFrame,
    downstream_model='svm',
    metric: str = 'F1',
    propagate_cols: list = None,
) -> pd.DataFrame:
    """
    Aggregate performance metrics from multiple dataframes for protein/DNA language models.

    This function takes an arbitrary number of dataframes containing performance metrics,
    filters the rows by a specified metric (default 'F1'), computes an aggregated accuracy
    per experiment as the mean of the top and bottom performance metrics, and calculates
    the pooled standard deviation. It then groups the results by 'model', 'task_class', and
    'task_filter' (which retains the super-kingdom information) along with any additional
    columns specified in `propagate_cols`. The function returns a single dataframe summarizing
    the mean and standard deviation of the aggregated accuracy for each group.

    Parameters
    ----------
    *dfs : pd.DataFrame
        One or more dataframes with identical columns, including:
        ['downstream_model_type', 'model', 'task_class',
         'task_filter', 'taxa_level', 'label_set', 'metric', 'raw_difference',
         'cohens_d', 'percent_difference', 'z_score', 'top_mean', 'top_std',
         'bottom_mean', 'bottom_std', 'p_value', 't_statistic'].
    metric : str, optional
        The metric to filter by from the 'metric' column (default is 'F1').
    propagate_cols : list of str, optional
        Additional column names to propagate forward in the aggregated result by taking
        the first occurrence within each group. These columns must be constant within
        each group.

    Returns
    -------
    pd.DataFrame
        A dataframe with aggregated performance metrics, with columns:
            - model
            - task_class
            - task_filter
            - [any additional propagate_cols]
            - mean_aggregated_accuracy : Mean aggregated accuracy (average of top and bottom means)
              across experiments.
            - std_aggregated_accuracy : Standard deviation of the aggregated accuracy across experiments.
    """
    if propagate_cols is None:
        propagate_cols = []

    # Concatenate all input dataframes.
    df_all = pd.concat(dfs, ignore_index=True)

    # Filter by downstream model
    df_all = df_all[df_all['downstream_model_type'] == downstream_model]

    # Filter rows by the specified metric.
    if 'metric' in df_all.columns:
        df_all = df_all[df_all['metric'] == metric]

    # Compute the aggregated accuracy and a row-wise pooled standard deviation.
    # Aggregated accuracy is computed as the mean of top_mean and bottom_mean.
    df_all['aggregated_accuracy'] = (df_all['top_mean'] + df_all['bottom_mean']) / 2
    # The row-wise pooled standard deviation is computed as:
    # sqrt((top_std^2 + bottom_std^2) / 2)
    df_all['aggregated_std'] = np.sqrt(
        (df_all['top_std'] ** 2 + df_all['bottom_std'] ** 2) / 2
    )

    # Define grouping columns: default grouping by model, task_class, task_filter,
    # plus any additional columns the user wishes to propagate.
    default_group_cols = ['model', 'task_class', 'task_filter']
    # Remove duplicates if any propagate column is already in the default group.
    group_cols = list(dict.fromkeys(default_group_cols + propagate_cols))

    # Group the data and compute the aggregated performance metrics using named aggregations.
    agg_df = (
        df_all.groupby(group_cols)
        .agg(
            mean_aggregated_accuracy=('aggregated_accuracy', 'mean'),
            std_aggregated_accuracy=('aggregated_accuracy', 'std'),
            **{col: (col, 'first') for col in propagate_cols},
        )
        .reset_index()
    )

    return agg_df


def plot_aggregated_results(
    aggregated_df: pd.DataFrame,
    filter_task_class: str = None,
    figsize: tuple = (10, 6),
    save_path: str = None,
    show: bool = True,
    filter_order: list = None,
    model_order: Union[list, dict[str, str]] = None,
) -> None:
    """
    Create publication-quality plots from an aggregated performance dataframe.

    This function creates a grouped bar plot with error bars representing the
    mean and standard deviation of the aggregated accuracy for each model and
    task_filter combination. If a specific task_class is provided via `filter_task_class`,
    only rows matching that task_class will be plotted. Otherwise, a separate subplot
    is created for each unique task_class.

    Parameters
    ----------
    aggregated_df : pd.DataFrame
        DataFrame with aggregated performance metrics. Expected columns include:
            - model
            - task_class
            - task_filter
            - mean_aggregated_accuracy
            - std_aggregated_accuracy
        Accuracy values should be between 0 and 1.
    filter_task_class : str, optional
        If provided, only rows with this task_class will be plotted (default is None).
    figsize : tuple, optional
        Size of the figure (default is (10, 6)).
    save_path : str, optional
        File path to save the generated figure (default is None).
    show : bool, optional
        If True, display the figure (default is True).
    filter_order : list, optional
        Custom order for task filters. If not provided, defaults to the sorted unique task filters.
    model_order : list or dict, optional
        Either:
        - A list of model names to determine display order and filtering. Only models that
          case-insensitively match entries in this list will be plotted, using the capitalization
          provided here.
        - A dictionary mapping internal model names (case-insensitive) to display names.
          Only models with keys that match (case-insensitive) will be plotted.
        If not provided, all models in the dataframe will be plotted in alphabetical order.

    Returns
    -------
    None
    """
    # Set default orders if none provided.
    if filter_order is None:
        filter_order = sorted(aggregated_df['task_filter'].unique())

    colorblind_palette = ['#0072B2', '#E69F00', '#009E73', '#AA4499']

    # Verify the dataframe contains the required columns.
    required_columns = {
        'model',
        'task_class',
        'task_filter',
        'mean_aggregated_accuracy',
        'std_aggregated_accuracy',
    }
    if not required_columns.issubset(aggregated_df.columns):
        raise ValueError(f'Input dataframe must contain columns: {required_columns}')

    # Apply a professional plotting style.
    plt.style.use('seaborn-v0_8-whitegrid')

    def create_grouped_bar(ax, df_subset: pd.DataFrame, title: str) -> None:
        """
        Create a grouped bar plot with error bars on the provided axis from the subset dataframe.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axis on which the plot is drawn.
        df_subset : pd.DataFrame
            Data subset for a single task_class.
        title : str
            Title of the subplot.
        """
        # Create lowercase mapping for case-insensitive matching
        # and a mapping from data models to their display names
        df_models = df_subset['model'].unique()
        model_dict = {
            m.lower(): m for m in df_models
        }  # For lookup from lowercase to actual

        # Determine the order of models and their display names
        if model_order is not None:
            if isinstance(model_order, dict):
                # If model_order is a dictionary, use it for both order and display names
                ordered_models = []
                display_names = {}

                # Convert all keys in model_order to lowercase for case-insensitive matching
                model_order_lower = {k.lower(): v for k, v in model_order.items()}

                # Only include models specified in model_order that exist in the data
                for model_lower in model_order_lower:
                    if model_lower in model_dict:
                        actual_model = model_dict[model_lower]
                        ordered_models.append(actual_model)
                        display_names[actual_model] = model_order_lower[model_lower]
            else:
                # If model_order is a list, use it for both order and display names when possible
                ordered_models = []
                display_names = {}  # Start with empty dictionary

                # Convert model_order items to lowercase for case-insensitive matching
                model_order_lower = [m.lower() for m in model_order]

                # Only include models specified in model_order that exist in the data
                for i, m_lower in enumerate(model_order_lower):
                    if m_lower in model_dict:
                        actual_model = model_dict[m_lower]
                        ordered_models.append(actual_model)
                        # Use the display version from model_order instead of the one in the dataframe
                        display_names[actual_model] = model_order[i]
        else:
            # Default: use sorted models with original names
            ordered_models = sorted(df_models)
            display_names = {m: m for m in ordered_models}  # Use original names

        # If we have no models to display after filtering, show a message and return
        if not ordered_models:
            ax.text(
                0.5,
                0.5,
                'No matching models found',
                ha='center',
                va='center',
                fontsize=14,
            )
            ax.set_title(title, fontsize=14)
            # Remove axes ticks and spines for a cleaner look
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            return

        x = np.arange(len(ordered_models))

        total_width = 0.8  # Total width reserved for bars in one group.
        num_filters = len(filter_order)
        bar_width = total_width / num_filters

        # Calculate offsets to center the bars.
        offsets = np.linspace(
            -total_width / 2 + bar_width / 2,
            total_width / 2 - bar_width / 2,
            num_filters,
        )

        # Plot each task_filter's bars with error bars.
        for off, t_filter, color in zip(offsets, filter_order, colorblind_palette):
            means, stds = [], []
            for model in ordered_models:
                row = df_subset[
                    (df_subset['model'] == model)
                    & (df_subset['task_filter'] == t_filter)
                ]
                if not row.empty:
                    means.append(row['mean_aggregated_accuracy'].values[0])
                    stds.append(row['std_aggregated_accuracy'].values[0])
                else:
                    means.append(np.nan)
                    stds.append(0)
            # Adding a black edge to each bar for a more defined look.
            ax.bar(
                x + off,
                means,
                width=bar_width,
                yerr=stds,
                capsize=5,
                label=t_filter,
                color=color,
                edgecolor='black',
            )

        ax.set_xticks(x)
        ax.set_xticklabels(
            [display_names[model] for model in ordered_models],
            rotation=45,
            ha='right',
            fontsize=10,
        )
        ax.set_ylabel('F1', fontsize=12)
        ax.set_ylim(0, 1)
        ax.set_title(title, fontsize=14)
        # Add a legend with a border, slightly moved up.
        legend = ax.legend(
            title='Domain Filter',
            fontsize=10,
            title_fontsize=11,
            # bbox_to_anchor=(1, 1.02),
            frameon=True,
            framealpha=0.95,
            edgecolor='gray',
            loc='lower left',
        )
        legend.get_frame().set_linewidth(0.8)

        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    # Generate plots depending on whether a specific task_class is filtered.
    if filter_task_class is not None:
        df_plot = aggregated_df[aggregated_df['task_class'] == filter_task_class]
        if df_plot.empty:
            raise ValueError(f"No data available for task_class '{filter_task_class}'")
        fig, ax = plt.subplots(figsize=figsize)
        title = f'Aggregated Performance for {filter_task_class.replace("TaxonomyBias", " Classification")}'
        create_grouped_bar(ax, df_plot, title)
    else:
        unique_classes = sorted(aggregated_df['task_class'].unique())
        n_classes = len(unique_classes)
        fig, axes = plt.subplots(
            n_classes, 1, figsize=(figsize[0], figsize[1] * n_classes), squeeze=False
        )
        for idx, task_class in enumerate(unique_classes):
            df_subset = aggregated_df[aggregated_df['task_class'] == task_class]
            title = f'Aggregated Performance for {task_class.replace("TaxonomyBias", " Classification")}'
            create_grouped_bar(axes[idx, 0], df_subset, title)
        plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


# Cross Eval utilities
def parse_metrics_file(file_path):
    # Load the JSON data
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Group the data by the implicit groups
    top_top = data[0:2]  # Top trained, top evaluated
    top_bottom = data[2:4]  # Top trained, bottom evaluated
    bottom_bottom = data[4:6]  # Bottom trained, bottom evaluated
    bottom_top = data[6:8]  # Bottom trained, top evaluated

    # Calculate mean and std for each group's metrics
    result = {
        'top_trained_top_eval': {},
        'top_trained_bottom_eval': {},
        'bottom_trained_bottom_eval': {},
        'bottom_trained_top_eval': {},
    }

    # Process each group and metric type
    for group_name, group_data in [
        ('top_trained_top_eval', top_top),
        ('top_trained_bottom_eval', top_bottom),
        ('bottom_trained_bottom_eval', bottom_bottom),
        ('bottom_trained_top_eval', bottom_top),
    ]:
        for metric in group_data:
            metric_name = metric['class_name']

            # Add raw scores
            result[group_name][f'{metric_name}_train'] = metric['train_scores']
            result[group_name][f'{metric_name}_test'] = metric['test_scores']

            # Add mean and std
            result[group_name][f'{metric_name}_train_mean'] = np.mean(
                metric['train_scores']
            )
            result[group_name][f'{metric_name}_train_std'] = np.std(
                metric['train_scores']
            )
            result[group_name][f'{metric_name}_test_mean'] = np.mean(
                metric['test_scores']
            )
            result[group_name][f'{metric_name}_test_std'] = np.std(
                metric['test_scores']
            )

    # Add metadata
    result['metric_is_higher_better'] = {
        metric['class_name']: metric['is_higher_better'] for metric in data[:2]
    }

    return result


def parse_metadata(file_path):
    """
    Extract metadata from file path based on the specified structure:
    <DOMAINFILTER>/<DOWNSTREAMMODEL>/<BASEMODEL>/<BASEMODEL>_CrossEval<TASKNAME>-<KINGDOMFILTER>-<TAXAFILTER>-<LABEL>
    """
    path = Path(file_path)
    filename = path.stem  # Remove .metrics extension

    # Extract path components
    parts = list(path.parts)

    # Find domain filter component (should be the first part of our pattern)
    domain_idx = -1
    for i, part in enumerate(parts):
        if part.endswith('-filter'):
            domain_idx = i
            break

    if domain_idx == -1 or len(parts) < domain_idx + 3:
        print(f'Directory structure not recognized for: {file_path}')
        return None

    # Extract directory components
    domainfilter = parts[domain_idx]
    downstreammodel = parts[domain_idx + 1].replace('-evals', '')
    basemodel = parts[domain_idx + 2]

    # Parse filename
    # Format: <BASEMODEL>_CrossEval<TASKNAME>-<KINGDOMFILTER>-<TAXAFILTER>-<LABEL>
    name_parts = filename.split('_', 1)
    if len(name_parts) < 2:
        print(f'Filename format not recognized: {filename}')
        return None

    basemodel_from_filename = name_parts[0]
    crosseval_part = name_parts[1]

    # Verify CrossEval prefix
    if not crosseval_part.startswith('CrossEval'):
        print(f'CrossEval not found in filename: {filename}')
        return None

    # Extract taskname and filters
    remaining = crosseval_part[len('CrossEval') :]
    parts = remaining.split('-')

    # Parse components with defaults for optional elements
    taskname = parts[0]
    kingdomfilter = parts[1] if len(parts) > 1 else ''
    taxafilter = parts[2] if len(parts) > 2 else ''
    label = parts[3] if len(parts) > 3 else ''

    return {
        'domainfilter': domainfilter,
        'downstreammodel': downstreammodel,
        'basemodel': basemodel,
        'basemodel_from_filename': basemodel_from_filename,
        'taskname': taskname,
        'kingdomfilter': kingdomfilter,
        'taxafilter': taxafilter,
        'label': label,
        'filename': filename,
    }


def load_metrics_data(root_dir):
    """
    Process all metrics files in the directory structure and return a list of data dictionaries
    that can be easily converted to a pandas DataFrame.

    Args:
        root_dir: Directory containing the metrics data (parent of domain filter directories)

    Returns:
        List of dictionaries, each containing metadata and metrics for a single file
    """
    all_data = []

    # Find all .metrics files
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.metrics') and 'CrossEval' in filename:
                file_path = os.path.join(dirpath, filename)

                try:
                    # Extract metadata
                    metadata = parse_metadata(file_path)
                    if metadata is None:
                        continue

                    # Parse metrics
                    metrics = parse_metrics_file(file_path)

                    # Combine metadata and metrics
                    result = {
                        'metadata': metadata,
                        'metrics': metrics,
                        'file_path': file_path,
                    }

                    all_data.append(result)

                except Exception as e:
                    print(f'Error processing {file_path}: {e}')

    return all_data


def prepare_dataframe_data(all_data):
    """
    Convert the nested data structure to a flattened format suitable for DataFrame creation.
    Each row represents one metrics file with all its metadata and performance metrics.

    Args:
        all_data: List of dictionaries returned by load_metrics_data

    Returns:
        List of flat dictionaries ready for pandas DataFrame creation
    """
    flattened_data = []

    for data in all_data:
        # Start with metadata
        row = {**data['metadata']}

        # Add file path
        row['file_path'] = data['file_path']

        # Add flattened metrics with clear naming
        for group_name, group_metrics in data['metrics'].items():
            for metric_name, value in group_metrics.items():
                # Skip raw score arrays to keep the dataframe manageable
                if isinstance(value, list):
                    continue

                # Create descriptive column name
                column_name = f'{group_name}_{metric_name}'
                row[column_name] = value

        flattened_data.append(row)

    return flattened_data


def create_robustness_ratio_comparison(
    df: pd.DataFrame,
    metric: str = 'F1',
    task: str = None,
    downstream_model: str = None,
    top_n: int = 10,
    models: list | dict | None = None,
    figsize: tuple = (10, 6),
    save_path: str = None,
    show: bool = True,
    fixed_ylim: bool = True,
    significance_threshold: float = 1.96,  # 1.96 -> 95%
) -> tuple:
    """
    Compare robustness ratios of models when tested on out-of-distribution data.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the flattened metrics data
    metric : str, optional
        Metric to visualize ('Accuracy' or 'F1') (default is 'F1')
    task : str, optional
        Filter for specific task (default is None)
    downstream_model : str, optional
        Filter for specific downstream model (default is None)
    top_n : int, optional
        Number of top models to show (default is 10, only used if models is None)
    models : list or dict, optional
        Either:
        - A list of model names to include in the plot
        - A dictionary mapping model names to display names
        If None, the top_n best models will be selected automatically
    figsize : tuple, optional
        Size of the figure (default is (10, 6))
    save_path : str, optional
        File path to save the generated figure (default is None)
    show : bool, optional
        If True, display the figure (default is True)
    fixed_ylim : bool, optional
        If True, use consistent y-axis limits for easier comparison across tasks
    significance_threshold : float, optional
        Z-score threshold for statistical significance (default is 1.96, corresponding to p < 0.05)

    Returns
    -------
    tuple
        A tuple containing (figure, metrics_dataframe)
    """
    # Verify input data
    required_columns = {'basemodel', 'taskname', 'downstreammodel'}
    df_columns = set(df.columns)
    if not required_columns.issubset(df_columns):
        raise ValueError(f'Input dataframe must contain columns: {required_columns}')

    # Filter data
    filtered_df = df.copy()
    if task is not None:
        filtered_df = filtered_df[filtered_df['taskname'] == task]
    if downstream_model is not None:
        filtered_df = filtered_df[filtered_df['downstreammodel'] == downstream_model]

    # Skip if no data
    if len(filtered_df) == 0:
        print(f'No data found for task={task}, downstream_model={downstream_model}')
        return None, None

    # Column names for the performance values
    top_top_col = f'top_trained_top_eval_{metric}_test_mean'
    top_bottom_col = f'top_trained_bottom_eval_{metric}_test_mean'
    bottom_bottom_col = f'bottom_trained_bottom_eval_{metric}_test_mean'
    bottom_top_col = f'bottom_trained_top_eval_{metric}_test_mean'

    # Check if standard deviation columns exist
    std_columns_exist = all(
        col.replace('_mean', '_std') in filtered_df.columns
        for col in [top_top_col, top_bottom_col, bottom_bottom_col, bottom_top_col]
    )

    # Group by model to aggregate across all taxonomic levels
    agg_columns = {
        top_top_col: 'mean',
        top_bottom_col: 'mean',
        bottom_bottom_col: 'mean',
        bottom_top_col: 'mean',
    }

    # Add standard deviation columns if they exist
    if std_columns_exist:
        for col in [top_top_col, top_bottom_col, bottom_bottom_col, bottom_top_col]:
            std_col = col.replace('_mean', '_std')
            agg_columns[std_col] = 'mean'

    # Perform aggregation
    agg_df = filtered_df.groupby('basemodel').agg(agg_columns).reset_index()

    # Calculate metrics
    model_metrics = []

    for _, row in agg_df.iterrows():
        # Calculate robustness ratios
        top_trained_robustness = row[top_bottom_col] / row[top_top_col]
        bottom_trained_robustness = row[bottom_top_col] / row[bottom_bottom_col]

        metrics_dict = {
            'basemodel': row['basemodel'],
            'top_trained_in_dist': row[top_top_col],
            'top_trained_out_dist': row[top_bottom_col],
            'bottom_trained_in_dist': row[bottom_bottom_col],
            'bottom_trained_out_dist': row[bottom_top_col],
            'top_trained_robustness': top_trained_robustness,
            'bottom_trained_robustness': bottom_trained_robustness,
            'robustness_difference': bottom_trained_robustness - top_trained_robustness,
        }

        # Add standard deviations if available
        if std_columns_exist:
            metrics_dict.update(
                {
                    'top_trained_in_dist_std': row[
                        top_top_col.replace('_mean', '_std')
                    ],
                    'top_trained_out_dist_std': row[
                        top_bottom_col.replace('_mean', '_std')
                    ],
                    'bottom_trained_in_dist_std': row[
                        bottom_bottom_col.replace('_mean', '_std')
                    ],
                    'bottom_trained_out_dist_std': row[
                        bottom_top_col.replace('_mean', '_std')
                    ],
                }
            )

        model_metrics.append(metrics_dict)

    # Convert to DataFrame
    metrics_df = pd.DataFrame(model_metrics)

    # Calculate significance and add it to the DataFrame
    if std_columns_exist:
        significance_data = []
        for _, row in metrics_df.iterrows():
            # For significance testing: use proper error propagation
            # Error propagation for ratio a/b: sqrt((σa/a)² + (σb/b)²) * (a/b)
            top_ratio_error_propagated = (
                np.sqrt(
                    (row['top_trained_out_dist_std'] / row['top_trained_out_dist']) ** 2
                    + (row['top_trained_in_dist_std'] / row['top_trained_in_dist']) ** 2
                )
                * row['top_trained_robustness']
            )

            bottom_ratio_error_propagated = (
                np.sqrt(
                    (
                        row['bottom_trained_out_dist_std']
                        / row['bottom_trained_out_dist']
                    )
                    ** 2
                    + (
                        row['bottom_trained_in_dist_std']
                        / row['bottom_trained_in_dist']
                    )
                    ** 2
                )
                * row['bottom_trained_robustness']
            )

            # For display: use a simpler one standard deviation approach
            # Simply scale the out-dist std dev by the same factor as the ratio
            top_ratio_error_display = (
                row['top_trained_out_dist_std'] / row['top_trained_in_dist']
            )
            bottom_ratio_error_display = (
                row['bottom_trained_out_dist_std'] / row['bottom_trained_in_dist']
            )

            # Calculate significance of difference between the two ratios
            diff = row['bottom_trained_robustness'] - row['top_trained_robustness']
            diff_error = np.sqrt(
                top_ratio_error_propagated**2 + bottom_ratio_error_propagated**2
            )
            z_score = abs(diff) / diff_error if diff_error > 0 else 0

            significance_data.append(
                {
                    'top_error': top_ratio_error_display,  # For display (one std deviation)
                    'bottom_error': bottom_ratio_error_display,  # For display (one std deviation)
                    'z_score': z_score,
                    'is_significant': z_score > significance_threshold,
                }
            )

        # Add significance data to DataFrame
        significance_df = pd.DataFrame(significance_data)
        metrics_df = pd.concat([metrics_df, significance_df], axis=1)

    # Create display name mapping and select/order models
    display_names = {}
    if models is not None:
        # Case insensitive lookup for model names
        model_lookup = {m.lower(): m for m in metrics_df['basemodel']}

        if isinstance(models, dict):
            # Create order mapping and display names
            model_order = {}
            for i, model_name in enumerate(models.keys()):
                model_lower = model_name.lower()
                if model_lower in model_lookup:
                    actual_model = model_lookup[model_lower]
                    model_order[actual_model] = i
                    display_names[actual_model] = models[model_name]
        else:
            # Create order mapping using list positions
            model_order = {}
            for i, model_name in enumerate(models):
                model_lower = model_name.lower()
                if model_lower in model_lookup:
                    actual_model = model_lookup[model_lower]
                    model_order[actual_model] = i
                    display_names[actual_model] = actual_model

        # Filter to models that exist in the data
        found_models = list(model_order.keys())

        if found_models:
            # Filter dataframe to selected models and sort by specified order
            metrics_df = metrics_df[metrics_df['basemodel'].isin(found_models)]
            metrics_df['_order'] = metrics_df['basemodel'].map(model_order)
            metrics_df = metrics_df.sort_values('_order').drop('_order', axis=1)
        else:
            print('Warning: None of the specified models were found in the data')

    else:
        # No models specified, select top_n based on performance
        metrics_df['avg_robustness'] = (
            metrics_df['top_trained_robustness']
            + metrics_df['bottom_trained_robustness']
        ) / 2
        metrics_df = metrics_df.sort_values('avg_robustness', ascending=False).head(
            top_n
        )

        # Create abbreviated names for long model names
        for model in metrics_df['basemodel']:
            if len(model) > 15:
                # Extract meaningful parts
                parts = model.split('-')
                if len(parts) > 1:
                    abbr = parts[0][:10] + '...' + parts[-1]
                else:
                    abbr = model[:12] + '...'
                display_names[model] = abbr
            else:
                display_names[model] = model

    # Apply a professional plotting style
    plt.style.use('seaborn-v0_8-whitegrid')

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Define colors with higher contrast
    colors = {
        'top_to_bottom': '#2c7c2c',  # green
        'bottom_to_top': '#882e72',  # magenta
        'reference_line': '#d62728',  # Red
        'annotation_bg': '#f8f8f8',  # Light gray
    }

    # Create plot data
    ratio_data = []

    for _, row in metrics_df.iterrows():
        model_name = display_names.get(row['basemodel'], row['basemodel'])

        ratio_data.append(
            {
                'basemodel': model_name,
                'training': 'Common Taxa',
                'ratio': row['top_trained_robustness'],
                'error': row.get('top_error', 0) if std_columns_exist else 0,
                'is_significant': row.get('is_significant', False)
                if std_columns_exist
                else False,
            }
        )
        ratio_data.append(
            {
                'basemodel': model_name,
                'training': 'Rare Taxa',
                'ratio': row['bottom_trained_robustness'],
                'error': row.get('bottom_error', 0) if std_columns_exist else 0,
                'is_significant': row.get('is_significant', False)
                if std_columns_exist
                else False,
            }
        )

    ratio_df = pd.DataFrame(ratio_data)

    # Set up plot
    bar_width = 0.35
    x = np.arange(len(metrics_df))

    # Parse out data for each training type
    top_trained_data = ratio_df[ratio_df['training'] == 'Common Taxa']
    top_trained_vals = top_trained_data['ratio'].values

    bottom_trained_data = ratio_df[ratio_df['training'] == 'Rare Taxa']
    bottom_trained_vals = bottom_trained_data['ratio'].values

    # Plot bars with error bars if available
    ax.bar(
        x - bar_width / 2,
        top_trained_vals,
        width=bar_width,
        yerr=top_trained_data['error'].values if std_columns_exist else None,
        capsize=4,
        label='Common Taxa',
        color=colors['top_to_bottom'],
        edgecolor='black',
        linewidth=0.8,
    )

    ax.bar(
        x + bar_width / 2,
        bottom_trained_vals,
        width=bar_width,
        yerr=bottom_trained_data['error'].values if std_columns_exist else None,
        capsize=4,
        label='Rare Taxa',
        color=colors['bottom_to_top'],
        edgecolor='black',
        linewidth=0.8,
    )

    # Add reference lines
    ax.axhline(
        y=1.0, color=colors['reference_line'], linestyle='--', alpha=0.7, linewidth=1
    )
    ax.axhline(y=0.75, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(y=0.25, color='gray', linestyle=':', alpha=0.5)

    # Set y-axis limits
    if fixed_ylim:
        ax.set_ylim(0, 1.05)  # Consistent y-limits for comparison across tasks

    # Set x-axis ticks and labels
    ax.set_xticks(x)

    # Prepare x-tick labels
    if len(metrics_df) > 6:
        # For many models, use vertical labels
        rotation = 90
        ha = 'center'
        fontsize = 9
    else:
        # For fewer models, use rotated labels
        rotation = 45
        ha = 'right'
        fontsize = 10

    # Get basemodel names in the current order
    basemodels = metrics_df['basemodel'].tolist()

    # Apply the labels
    ax.set_xticklabels(
        [display_names.get(m, m) for m in basemodels],
        rotation=rotation,
        ha=ha,
        fontsize=fontsize,
    )

    # Bold the labels for significant differences
    if std_columns_exist:
        for i, (_, row) in enumerate(metrics_df.iterrows()):
            if row.get('is_significant', False):
                tick = ax.get_xticklabels()[i]
                tick.set_fontweight('bold')

    # Get names of models with significant differences for debug
    if std_columns_exist:
        sig_models = metrics_df[metrics_df['is_significant'] == True][
            'basemodel'
        ].tolist()

    # Add labels and title
    ax.set_ylabel('Robustness Ratio (Out-of-dist/In-dist)', fontsize=12)
    ax.set_title('Robustness Ratio', fontsize=14)

    # Add legend with border
    legend = ax.legend(
        title='Training Data',
        fontsize=10,
        title_fontsize=11,
        frameon=True,
        framealpha=0.95,
        edgecolor='gray',
        loc='lower right',
    )
    legend.get_frame().set_linewidth(0.8)

    # Add grid
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.90 if task else 0.95)

    # Save figure if path is provided
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    # Show or close figure
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, metrics_df
