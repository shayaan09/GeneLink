from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# FILE PATHS
# ============================================================

# This script should live inside:
#
# GeneLink/
# └── scripts/
#     └── inspect_expression.py

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# This is the processed expression table we created earlier.
EXPRESSION_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "expression_features.tsv"
)

# We will save a small summary table here so you can inspect
# the statistics for every individual gene if you want.
SUMMARY_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "expression_inspection_summary.tsv"
)


# ============================================================
# LOAD EXPRESSION DATA
# ============================================================

def load_expression_features(path: Path) -> pd.DataFrame:
    """
    Load the processed expression table.

    Expected structure:

        node_index
        gene_id
        expression_1
        expression_2
        ...
        expression_805

    Each ROW represents one gene.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Could not find expression file:\n{path}"
        )

    data = pd.read_csv(
        path,
        sep="\t",
    )

    print(f"Loaded expression table: {data.shape[0]} genes")

    return data



def inspect_experiment_statistics(
    data: pd.DataFrame,
    expression_columns: list,
) -> None:
    """
    Inspect each experiment/feature across all 1,643 genes.

    Earlier, we looked at each GENE across 805 experiments.

    Now we do the opposite:

        expression_1 -> values for all 1,643 genes
        expression_2 -> values for all 1,643 genes
        ...
        expression_805 -> values for all 1,643 genes

    We want to know whether some experiments are on very
    different numerical scales from others.

    If one experiment had dramatically larger values than the
    others, normalization might be useful before training.
    """

    # Keep only the 805 actual expression columns.
    expression_data = data[expression_columns]

    # --------------------------------------------------------
    # Calculate statistics for every experiment column.
    # --------------------------------------------------------

    experiment_summary = pd.DataFrame({
        "experiment": expression_columns,

        # Minimum expression across all genes in each experiment.
        "min_expression": expression_data.min(axis=0).values,

        # Maximum expression across all genes in each experiment.
        "max_expression": expression_data.max(axis=0).values,

        # Average expression across all genes in each experiment.
        "mean_expression": expression_data.mean(axis=0).values,

        # Standard deviation across all genes in each experiment.
        "std_expression": expression_data.std(
            axis=0,
            ddof=0,
        ).values,
    })

    print("\n--- Experiment / Feature Statistics ---")

    # --------------------------------------------------------
    # Compare the average expression levels of experiments.
    # --------------------------------------------------------

    print(
        f"Lowest experiment mean:  "
        f"{experiment_summary['mean_expression'].min():.6f}"
    )

    print(
        f"Highest experiment mean: "
        f"{experiment_summary['mean_expression'].max():.6f}"
    )

    # --------------------------------------------------------
    # Compare how spread out the genes are within experiments.
    # --------------------------------------------------------

    print(
        f"Lowest experiment std:   "
        f"{experiment_summary['std_expression'].min():.6f}"
    )

    print(
        f"Highest experiment std:  "
        f"{experiment_summary['std_expression'].max():.6f}"
    )

    # --------------------------------------------------------
    # Show the experiments with the smallest variation.
    # --------------------------------------------------------

    lowest_variation = experiment_summary.sort_values(
        "std_expression",
        ascending=True,
    ).head(10)

    print("\n--- 10 Experiments With Lowest Variation ---")

    print(
        lowest_variation[
            [
                "experiment",
                "mean_expression",
                "std_expression",
                "min_expression",
                "max_expression",
            ]
        ].to_string(index=False)
    )

    # --------------------------------------------------------
    # Show the experiments with the largest variation.
    # --------------------------------------------------------

    highest_variation = experiment_summary.sort_values(
        "std_expression",
        ascending=False,
    ).head(10)

    print("\n--- 10 Experiments With Highest Variation ---")

    print(
        highest_variation[
            [
                "experiment",
                "mean_expression",
                "std_expression",
                "min_expression",
                "max_expression",
            ]
        ].to_string(index=False)
    )


# ============================================================
# FIND THE EXPRESSION COLUMNS
# ============================================================

def get_expression_columns(data: pd.DataFrame) -> list:
    """
    Find only the columns containing expression measurements.

    We do NOT want to accidentally include:

        node_index
        gene_id

    because those are identifiers, not biological measurements.
    """

    expression_columns = [
        column
        for column in data.columns
        if column.startswith("expression_")
    ]

    if len(expression_columns) == 0:
        raise ValueError(
            "No expression columns were found."
        )

    print(
        f"Found {len(expression_columns)} "
        f"expression measurements per gene."
    )

    return expression_columns


# ============================================================
# CHECK FOR MISSING VALUES
# ============================================================

def inspect_missing_values(
    data: pd.DataFrame,
    expression_columns: list,
) -> None:
    """
    Check whether any expression measurements are missing.

    Missing values are a problem because neural networks
    generally cannot safely train on NaN values.
    """

    expression_data = data[expression_columns]

    total_missing = expression_data.isna().sum().sum()

    print("\n--- Missing Values ---")
    print(f"Missing expression values: {total_missing}")

    if total_missing == 0:
        print("No missing values found.")
    else:
        print("WARNING: Missing values exist.")


# ============================================================
# OVERALL DATASET STATISTICS
# ============================================================

def inspect_overall_statistics(
    data: pd.DataFrame,
    expression_columns: list,
) -> None:
    """
    Look at all expression measurements together.

    This gives us the overall numerical scale of the dataset.

    We check:

        minimum
        maximum
        mean
        standard deviation

    These values help us decide later whether normalization
    might be useful.
    """

    expression_matrix = data[
        expression_columns
    ].to_numpy(dtype=np.float64)

    overall_min = expression_matrix.min()
    overall_max = expression_matrix.max()
    overall_mean = expression_matrix.mean()

    # ddof=0 means we calculate population standard deviation.
    overall_std = expression_matrix.std(ddof=0)

    print("\n--- Overall Expression Statistics ---")

    print(f"Minimum:            {overall_min:.6f}")
    print(f"Maximum:            {overall_max:.6f}")
    print(f"Mean:               {overall_mean:.6f}")
    print(f"Standard deviation: {overall_std:.6f}")


# ============================================================
# CREATE PER-GENE STATISTICS
# ============================================================

def create_gene_statistics(
    data: pd.DataFrame,
    expression_columns: list,
) -> pd.DataFrame:
    """
    Calculate statistics separately for every gene.

    For each gene, we look across its 805 experiments and find:

        minimum expression
        maximum expression
        average expression
        standard deviation

    Standard deviation tells us how much that gene changes
    across experiments.

    A very small standard deviation means the gene barely
    changes.

    A larger standard deviation means its activity changes
    more strongly between experiments.
    """

    expression_data = data[
        expression_columns
    ]

    summary = pd.DataFrame()

    # Keep the identifiers so we know which statistics
    # belong to which gene.
    summary["node_index"] = data["node_index"]
    summary["gene_id"] = data["gene_id"]

    # Calculate statistics across the 805 measurements
    # for each individual gene.
    summary["min_expression"] = expression_data.min(axis=1)

    summary["max_expression"] = expression_data.max(axis=1)

    summary["mean_expression"] = expression_data.mean(axis=1)

    # ddof=0 makes this population standard deviation.
    summary["std_expression"] = expression_data.std(
        axis=1,
        ddof=0,
    )

    return summary


# ============================================================
# INSPECT DIFFERENCES BETWEEN GENES
# ============================================================

def inspect_gene_statistics(summary: pd.DataFrame) -> None:
    """
    Print useful information about how genes differ.

    In particular, we want to know:

        Do some genes have much larger average values?

        Do some genes barely change at all?

        Are some genes much more variable than others?
    """

    print("\n--- Differences Between Genes ---")

    print(
        f"Lowest gene mean:  "
        f"{summary['mean_expression'].min():.6f}"
    )

    print(
        f"Highest gene mean: "
        f"{summary['mean_expression'].max():.6f}"
    )

    print(
        f"Lowest gene std:   "
        f"{summary['std_expression'].min():.6f}"
    )

    print(
        f"Highest gene std:  "
        f"{summary['std_expression'].max():.6f}"
    )


# ============================================================
# FIND GENES THAT CHANGE THE LEAST
# ============================================================

def show_lowest_variance_genes(
    summary: pd.DataFrame,
    count: int = 10,
) -> None:
    """
    Show the genes whose expression changes the least
    across the 805 experiments.

    Small standard deviation = very little change.
    """

    lowest = summary.sort_values(
        "std_expression",
        ascending=True,
    ).head(count)

    print(
        f"\n--- {count} Genes With Lowest Variation ---"
    )

    print(
        lowest[
            [
                "gene_id",
                "mean_expression",
                "std_expression",
            ]
        ].to_string(index=False)
    )


# ============================================================
# FIND GENES THAT CHANGE THE MOST
# ============================================================

def show_highest_variance_genes(
    summary: pd.DataFrame,
    count: int = 10,
) -> None:
    """
    Show the genes whose activity changes the most
    across the 805 experiments.
    """

    highest = summary.sort_values(
        "std_expression",
        ascending=False,
    ).head(count)

    print(
        f"\n--- {count} Genes With Highest Variation ---"
    )

    print(
        highest[
            [
                "gene_id",
                "mean_expression",
                "std_expression",
            ]
        ].to_string(index=False)
    )


# ============================================================
# CHECK FOR COMPLETELY CONSTANT GENES
# ============================================================

def inspect_constant_genes(summary: pd.DataFrame) -> None:
    """
    Check whether any gene has exactly zero variation.

    std = 0 means the gene had the same expression value
    in every single experiment.
    """

    constant_genes = summary[
        summary["std_expression"] == 0
    ]

    print("\n--- Constant Genes ---")

    print(
        f"Genes with zero expression variation: "
        f"{len(constant_genes)}"
    )

    if len(constant_genes) > 0:
        print(
            constant_genes[
                ["gene_id", "mean_expression"]
            ].to_string(index=False)
        )


# ============================================================
# SAVE PER-GENE SUMMARY
# ============================================================

def save_summary(summary: pd.DataFrame) -> None:
    """
    Save the per-gene statistics so we can inspect them later.

    IMPORTANT:

    This does NOT modify the expression data.

    It simply creates a separate summary file.
    """

    summary.to_csv(
        SUMMARY_FILE,
        sep="\t",
        index=False,
    )

    print(
        f"\nSaved gene statistics to:\n{SUMMARY_FILE}"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """
    Run the complete expression inspection.

    Nothing in this script changes the dataset.

    We are only looking at the numbers so that we can make
    an informed decision about normalization afterward.
    """

    print("=" * 60)
    print("GeneLink - Expression Data Inspection")
    print("=" * 60)

    # Load our processed expression table.
    data = load_expression_features(
        EXPRESSION_FILE
    )

    # Identify the 805 expression columns.
    expression_columns = get_expression_columns(
        data
    )

    # Make sure there are no missing values.
    inspect_missing_values(
        data,
        expression_columns,
    )

    # Look at the overall numerical scale.
    inspect_overall_statistics(
        data,
        expression_columns,
    )

    inspect_experiment_statistics(
    data,
    expression_columns,
)

    # Calculate statistics for every gene individually.
    summary = create_gene_statistics(
        data,
        expression_columns,
    )

    # Compare genes with each other.
    inspect_gene_statistics(summary)

    # Show genes that barely change.
    show_lowest_variance_genes(summary)

    # Show genes that change the most.
    show_highest_variance_genes(summary)

    # Check whether any genes are completely constant.
    inspect_constant_genes(summary)

    # Save the statistics for manual inspection.
    save_summary(summary)

    print("\n" + "=" * 60)
    print("Expression inspection complete.")
    print("=" * 60)


# ============================================================
# SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()