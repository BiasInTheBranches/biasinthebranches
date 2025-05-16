"""Command-line interface for generating benchmark results."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from harness.reporting import utils

# setting up display orders
model_order = {
    'ankh-base': 'Ankh-base',
    'esm-150m': 'ESM2-150M',
    'esm-650m': 'ESM2-650M',
    'esmc-300m': 'ESMC-300M',
    'esmc-600m': 'ESMC-600M',
    'onehot-AA': 'One Hot (AA)',
    'calm': 'CaLM',
    'dnabert2': 'DNABERT-2',
    'nucleotidetransformer-500m-multi-species': 'NT-500M-MultiSpecies',
    'onehot-DNA': 'One Hot (Codon)',
}

global_task_orders = {
    'pfam': [
        'PfamTaxonomyBias-phylum-pfam',
        'PfamTaxonomyBias-class-pfam',
        'PfamTaxonomyBias-order-pfam',
        'PfamTaxonomyBias-family-pfam',
        'PfamTaxonomyBias-genus-pfam',
    ],
    'ec_0': [
        'ECTaxonomyBias-phylum-ec0',
        'ECTaxonomyBias-class-ec0',
        'ECTaxonomyBias-order-ec0',
        'ECTaxonomyBias-family-ec0',
        'ECTaxonomyBias-genus-ec0',
    ],
    'ec_1': [
        'ECTaxonomyBias-phylum-ec1',
        'ECTaxonomyBias-class-ec1',
        'ECTaxonomyBias-order-ec1',
        'ECTaxonomyBias-family-ec1',
        'ECTaxonomyBias-genus-ec1',
    ],
    'ec_2': [
        'ECTaxonomyBias-phylum-ec2',
        'ECTaxonomyBias-class-ec2',
        'ECTaxonomyBias-order-ec2',
        'ECTaxonomyBias-family-ec2',
        'ECTaxonomyBias-genus-ec2',
    ],
    'ec_3': [
        'ECTaxonomyBias-phylum-ec3',
        'ECTaxonomyBias-class-ec3',
        'ECTaxonomyBias-order-ec3',
        'ECTaxonomyBias-family-ec3',
        'ECTaxonomyBias-genus-ec3',
    ],
    'gene3d_0': [
        'Gene3DTaxonomyBias-phylum-gene3d0',
        'Gene3DTaxonomyBias-class-gene3d0',
        'Gene3DTaxonomyBias-order-gene3d0',
        'Gene3DTaxonomyBias-family-gene3d0',
        'Gene3DTaxonomyBias-genus-gene3d0',
    ],
    'gene3d_1': [
        'Gene3DTaxonomyBias-phylum-gene3d1',
        'Gene3DTaxonomyBias-class-gene3d1',
        'Gene3DTaxonomyBias-order-gene3d1',
        'Gene3DTaxonomyBias-family-gene3d1',
        'Gene3DTaxonomyBias-genus-gene3d1',
    ],
    'gene3d_2': [
        'Gene3DTaxonomyBias-phylum-gene3d2',
        'Gene3DTaxonomyBias-class-gene3d2',
        'Gene3DTaxonomyBias-order-gene3d2',
        'Gene3DTaxonomyBias-family-gene3d2',
        'Gene3DTaxonomyBias-genus-gene3d2',
    ],
    'gene3d_3': [
        'Gene3DTaxonomyBias-phylum-gene3d3',
        'Gene3DTaxonomyBias-class-gene3d3',
        'Gene3DTaxonomyBias-order-gene3d3',
        'Gene3DTaxonomyBias-family-gene3d3',
        'Gene3DTaxonomyBias-genus-gene3d3',
    ],
}


def make_global_bias_table(nodomain_master_df: pd.DataFrame, output_dir: Path) -> None:
    """Generate global bias table."""

    downstream_model = 'svm'
    task_metric = 'F1'
    gap_metric = 'raw_difference'
    latex_save_fmt = '{}-{}-{}-aggregate.tex'

    formatted_tables = {}

    for task_name, task_order in global_task_orders.items():
        summary_table = utils.create_summary_table(
            master_df=nodomain_master_df,
            downstream_model_type=downstream_model,
            task_metric=task_metric,
            gap_metric=gap_metric,
            model_order=model_order,
            task_order=task_order,
        )

        formatted_tables[task_name] = summary_table

    # Aggregate the precomputed "Mean_Gap" columns into one table.
    aggregated_table = utils.aggregate_precomputed_mean_gaps(formatted_tables.items())
    styled_aggregated = utils.styled_summary_table(aggregated_table, gap_metric)

    # Save table
    table_out_path = output_dir / latex_save_fmt.format(
        downstream_model, task_metric, gap_metric
    )
    output_latex = utils.format_styled_latex(
        styled_aggregated.to_latex(position='h', caption=' ', label=' ')
    )
    with open(table_out_path, 'w') as f:
        f.write(output_latex)


def make_domain_bias_tables(all_dfs: dict[str, pd.DataFrame], output_dir: Path) -> None:
    """Generate domain bias tables."""
    downstream_model = 'svm'
    task_metric = 'F1'
    gap_metric = 'raw_difference'
    latex_save_fmt = '{}-{}-{}-{}-aggregate.tex'

    domain_splits = ['bacteria', 'eukaryota', 'archaea', 'viruses']
    for domain_split in domain_splits:
        formatted_tables = {}
        for task_name, task_order in global_task_orders.items():
            summary_table = utils.create_summary_table(
                master_df=all_dfs[domain_split],
                downstream_model_type=downstream_model,
                task_metric=task_metric,
                gap_metric=gap_metric,
                model_order=model_order,
                task_order=task_order,
            )

            formatted_tables[task_name] = summary_table

        # Aggregate the precomputed "Mean_Gap" columns into one table.
        aggregated_table = utils.aggregate_precomputed_mean_gaps(
            formatted_tables.items()
        )

        styled_aggregated = utils.styled_summary_table(aggregated_table, gap_metric)

        # Save table
        table_out_path = output_dir / latex_save_fmt.format(
            domain_split, downstream_model, task_metric, gap_metric
        )
        output_latex = utils.format_styled_latex(
            styled_aggregated.to_latex(position='h', caption=' ', label=' ')
        )
        with open(table_out_path, 'w') as f:
            f.write(output_latex)


def plot_domain_performance(
    bacteria_master_df: pd.DataFrame,
    eukaryota_master_df: pd.DataFrame,
    archaea_master_df: pd.DataFrame,
    viruses_master_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    downstream_model = 'svm'
    metric = 'F1'

    # setting up hierarchical labels
    hierarchical = ['ec1', 'ec2', 'ec3', 'gene3d1', 'gene3d2', 'gene3d3']
    # Filter out the hierarchical labels from the master dataframes (only keep the first level labels)
    domain_aggregated_df = utils.aggregate_domain_results(
        bacteria_master_df[~bacteria_master_df['label_set'].isin(hierarchical)],
        eukaryota_master_df[~eukaryota_master_df['label_set'].isin(hierarchical)],
        archaea_master_df[~archaea_master_df['label_set'].isin(hierarchical)],
        viruses_master_df[~viruses_master_df['label_set'].isin(hierarchical)],
        downstream_model=downstream_model,
        metric=metric,
    )

    for task in ['PfamTaxonomyBias', 'ECTaxonomyBias', 'Gene3DTaxonomyBias']:
        fig = utils.plot_aggregated_results(
            domain_aggregated_df,
            model_order=model_order,
            filter_task_class=task,
            figsize=(6, 6),
        )
        output_path = (
            output_dir / f'colorblind-domain-performance-barchart-{task.lower()}.png'
        )
        fig.savefig(output_path, dpi=300, bbox_inches='tight')


def make_robustness_plot(df: pd.DataFrame, output_dir: Path) -> None:
    """Generate robustness plots."""

    for task in ['PfamTaxonomyBias', 'ECTaxonomyBias', 'Gene3DTaxonomyBias']:
        fig, metrics_df = utils.create_robustness_ratio_comparison(
            df,
            metric='F1',
            task=task,
            downstream_model='svm',
            models=model_order,
            figsize=(6, 6),
            significance_threshold=1.0,
        )
        outpath = (
            output_dir
            / f'domain-aggregated-downstream-robustness-barchart-{task.lower()}.png'
        )
        fig.savefig(outpath, dpi=300, bbox_inches='tight')


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Generate benchmark results.')
    parser.add_argument(
        'input_dir', type=Path, help='Directory containing the benchmark result files.'
    )
    parser.add_argument(
        'output_dir', type=Path, help='Output directory to place figures and results.'
    )
    return parser.parse_args()


def main() -> None:
    """Main function to generate benchmark results."""
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir

    # Ensure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    bacteria_results_root = input_dir / 'bacteria-filter'
    bacteria_master_df = utils.build_master_dataframe(bacteria_results_root)

    eukaryota_results_root = input_dir / 'eukarya-filter'
    eukaryota_master_df = utils.build_master_dataframe(eukaryota_results_root)

    archaea_results_root = input_dir / 'archaea-filter'
    archaea_master_df = utils.build_master_dataframe(archaea_results_root)

    viruses_results_root = input_dir / 'viruses-filter'
    viruses_master_df = utils.build_master_dataframe(viruses_results_root)

    nodomain_results_root = input_dir / 'nodomain-filter'
    nodomain_master_df = utils.build_master_dataframe(nodomain_results_root)

    all_dfs = {
        'nodomain': nodomain_master_df,
        'bacteria': bacteria_master_df,
        'eukaryota': eukaryota_master_df,
        'archaea': archaea_master_df,
        'viruses': viruses_master_df,
    }

    # Save the master dataframes to CSV files
    for name, df in all_dfs.items():
        df.to_csv(output_dir / f'{name}_master_df.csv', index=False)

    # Generate global bias table
    make_global_bias_table(nodomain_master_df=nodomain_master_df, output_dir=output_dir)

    # Generate domain bias tables
    make_domain_bias_tables(all_dfs=all_dfs, output_dir=output_dir)

    # Plot domain performance
    plot_domain_performance(
        bacteria_master_df=bacteria_master_df,
        eukaryota_master_df=eukaryota_master_df,
        archaea_master_df=archaea_master_df,
        viruses_master_df=viruses_master_df,
        output_dir=output_dir,
    )

    # Make robustness plot
    robustness_flat_data = utils.prepare_dataframe_data(
        utils.load_metrics_data(nodomain_results_root)
    )
    robustness_flat_df = pd.DataFrame(robustness_flat_data)

    robustness_flat_df = make_robustness_plot(robustness_flat_df, output_dir=output_dir)

    print(f'All tasks completed successfully, saved to {output_dir}.')


if __name__ == '__main__':
    main()
