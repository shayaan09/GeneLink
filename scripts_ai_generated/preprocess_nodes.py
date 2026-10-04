from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

# DREAM5 Network 1 contains 1,643 genes.
EXPECTED_GENE_COUNT = 1643

# We observed 805 expression experiments/conditions.
EXPECTED_EXPERIMENT_COUNT = 805

# DREAM5 Network 1 contains 195 possible transcription factors.
EXPECTED_TF_COUNT = 195


# ============================================================
# PROJECT PATHS
# ============================================================

# This script should live here:
#
# GeneLink/
# └── scripts/
#     └── preprocess_nodes.py
#
# __file__ gives us the location of this Python file.
SCRIPT_DIR = Path(__file__).resolve().parent

# Move one directory upward:
#
# GeneLink/scripts/
#       ↓
# GeneLink/
PROJECT_ROOT = SCRIPT_DIR.parent


# ---------------- RAW INPUT FILES ----------------

GENE_IDS_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "net1_gene_ids.tsv"
)

EXPRESSION_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "net1_expression_data.tsv"
)

TF_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "net1_transcription_factors.tsv"
)


# ---------------- PROCESSED OUTPUT FOLDER ----------------

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


# ============================================================
# LOAD GENE IDS
# ============================================================

def load_gene_ids(path: Path) -> pd.DataFrame:
    """
    Load the DREAM5 gene-ID file.

    The file already contains a header row, so pandas should
    use that first row as the column names.

    We then rename the two columns to:
        gene_id
        biological_name

    Finally, we create a numeric node index:
        G1 -> 0
        G2 -> 1
        ...
        G1643 -> 1642
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Gene-ID file not found:\n{path}"
        )

    # header=0 means:
    # "the first row of the file contains the column names."
    genes = pd.read_csv(
        path,
        sep="\t",
        header=0,
    )

    # Rename the two columns into names we will consistently
    # use throughout GeneLink.
    genes.columns = [
        "gene_id",
        "biological_name",
    ]

    # Create PyTorch-friendly numeric node IDs:
    #
    # G1 -> 0
    # G2 -> 1
    # ...
    genes.insert(
        0,
        "node_index",
        range(len(genes)),
    )

    print(f"Loaded {len(genes)} genes.")

    return genes

# ============================================================
# LOAD EXPRESSION DATA
# ============================================================

def load_expression_data(path: Path) -> pd.DataFrame:
    """
    Load the DREAM5 expression table.

    The raw structure looks like:

                  G1      G2      G3   ...
    experiment1   0.2     1.1     0.7
    experiment2   0.5     0.9     1.3
    experiment3   0.8     1.4     0.4
    ...

    There are:

        805 rows      = experiments / conditions
        1643 columns  = genes

    At this stage we simply load the table.

    We transpose it later so that each ROW represents one gene.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Expression file not found:\n{path}"
        )

    # Unlike the gene-ID file, the expression file DOES
    # contain gene names as its column headers.
    expression = pd.read_csv(
        path,
        sep="\t",
        header=0,
    )

    print(
        f"Loaded expression matrix: "
        f"{expression.shape[0]} experiments x "
        f"{expression.shape[1]} genes."
    )

    return expression


# ============================================================
# LOAD TRANSCRIPTION FACTORS
# ============================================================

def load_transcription_factors(path: Path) -> set:
    """
    Load the list of genes that are allowed to act as
    transcription factors.

    The raw file looks like:

        G1
        G2
        G3
        ...
        G195

    We return a Python set because membership checks are easy:

        if "G24" in transcription_factors:
            ...

    A TF can be a controller.

    A non-TF cannot be used as the SOURCE of a regulatory edge
    in this DREAM5 dataset.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Transcription-factor file not found:\n{path}"
        )

    tf_table = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["gene_id"],
    )

    # Convert the one-column DataFrame into a Python set.
    transcription_factors = set(
        tf_table["gene_id"]
    )

    print(
        f"Loaded {len(transcription_factors)} "
        f"transcription factors."
    )

    return transcription_factors


# ============================================================
# VALIDATE RAW FILES AGAINST EACH OTHER
# ============================================================

def validate_raw_data(
    genes: pd.DataFrame,
    expression: pd.DataFrame,
    transcription_factors: set,
) -> None:
    """
    Confirm that the three DREAM5 files agree with each other.

    This is extremely important.

    For example:

        If G500 exists in the gene-ID file,
        but there is no G500 expression column,
        then we could not correctly build that node.

    The checks here make sure every file refers to the same genes.
    """

    print("\nValidating raw node data...")

    # ========================================================
    # CHECK 1: Correct number of genes
    # ========================================================

    if len(genes) != EXPECTED_GENE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_GENE_COUNT} genes, "
            f"but found {len(genes)}."
        )

    # ========================================================
    # CHECK 2: Correct expression dimensions
    # ========================================================

    if expression.shape[0] != EXPECTED_EXPERIMENT_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_EXPERIMENT_COUNT} "
            f"expression experiments, but found "
            f"{expression.shape[0]}."
        )

    if expression.shape[1] != EXPECTED_GENE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_GENE_COUNT} expression "
            f"columns, but found {expression.shape[1]}."
        )

    # ========================================================
    # CHECK 3: Correct number of transcription factors
    # ========================================================

    if len(transcription_factors) != EXPECTED_TF_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_TF_COUNT} transcription "
            f"factors, but found "
            f"{len(transcription_factors)}."
        )

    # ========================================================
    # CHECK 4: No duplicate gene IDs
    # ========================================================

    if genes["gene_id"].duplicated().any():
        raise ValueError(
            "Duplicate gene IDs found in gene-ID file."
        )

    # ========================================================
    # CHECK 5: No duplicate biological names
    #
    # We DO NOT reject duplicate biological names.
    #
    # Biological aliases/names are metadata.
    # The DREAM5 G-number is our actual unique identifier.
    # ========================================================

    # ========================================================
    # CHECK 6: No duplicate expression columns
    # ========================================================

    if expression.columns.duplicated().any():
        raise ValueError(
            "Duplicate gene columns found in expression data."
        )

    # ========================================================
    # CHECK 7: Gene IDs match expression columns
    # ========================================================

    gene_id_set = set(genes["gene_id"])

    expression_gene_set = set(expression.columns)

    # Genes that exist in the official gene list but are
    # missing from the expression file.
    missing_from_expression = (
        gene_id_set - expression_gene_set
    )

    # Genes that appear in expression data but not in the
    # official gene list.
    extra_in_expression = (
        expression_gene_set - gene_id_set
    )

    if missing_from_expression:
        raise ValueError(
            "Genes missing from expression data:\n"
            f"{sorted(missing_from_expression)}"
        )

    if extra_in_expression:
        raise ValueError(
            "Unexpected genes in expression data:\n"
            f"{sorted(extra_in_expression)}"
        )

    # ========================================================
    # CHECK 8: Expression columns are in the same order
    # ========================================================

    # Sets only tell us whether the SAME genes exist.
    #
    # We also want to make sure the order is:
    #
    # G1, G2, G3, ..., G1643
    #
    # matching the gene-ID table.
    gene_order = genes["gene_id"].tolist()

    expression_order = expression.columns.tolist()

    if gene_order != expression_order:
        raise ValueError(
            "Expression columns contain the correct genes, "
            "but they are not in the same order as the "
            "gene-ID file."
        )

    # ========================================================
    # CHECK 9: Every transcription factor is a real gene
    # ========================================================

    invalid_tfs = (
        transcription_factors - gene_id_set
    )

    if invalid_tfs:
        raise ValueError(
            "The TF file contains genes not present in the "
            f"gene-ID file:\n{sorted(invalid_tfs)}"
        )

    # ========================================================
    # CHECK 10: No missing expression measurements
    # ========================================================

    if expression.isna().any().any():
        raise ValueError(
            "Expression matrix contains missing values."
        )

    # ========================================================
    # CHECK 11: Expression values must actually be numbers
    # ========================================================

    # select_dtypes checks how many columns pandas recognizes
    # as numeric.
    numeric_columns = expression.select_dtypes(
        include=[np.number]
    ).columns

    if len(numeric_columns) != EXPECTED_GENE_COUNT:
        raise ValueError(
            "One or more expression columns contain "
            "non-numeric values."
        )

    print("Raw node-data validation passed.")


# ============================================================
# CREATE GENE MAPPING / NODE METADATA
# ============================================================

def create_gene_metadata(
    genes: pd.DataFrame,
    transcription_factors: set,
) -> pd.DataFrame:
    """
    Create the main metadata table for our graph nodes.

    The output will look like:

        node_index  gene_id  biological_name  is_tf
        0           G1       ...              1
        1           G2       ...              1
        ...
        195         G196     ...              0

    is_tf:

        1 = this gene is a transcription factor
        0 = this gene is not a transcription factor
    """

    metadata = genes.copy()

    # For every gene, ask:
    #
    # "Is this gene ID inside the TF set?"
    #
    # True / False is then converted to:
    #
    # 1 / 0
    metadata["is_tf"] = (
        metadata["gene_id"]
        .isin(transcription_factors)
        .astype(int)
    )

    return metadata


# ============================================================
# TRANSPOSE EXPRESSION DATA
# ============================================================

def create_expression_features(
    genes: pd.DataFrame,
    expression: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert the expression matrix into node-feature format.

    ORIGINAL DREAM5 FORMAT:

        rows    = experiments
        columns = genes

        shape:
        805 x 1643


    GRAPH NODE FORMAT:

        rows    = genes
        columns = expression measurements

        shape:
        1643 x 805


    Example:

    Original:

                G1     G2
        exp1    0.2    0.7
        exp2    0.5    0.9


    After transpose:

                exp_1   exp_2
        G1      0.2     0.5
        G2      0.7     0.9
    """

    # .T means transpose:
    #
    # rows become columns
    # columns become rows
    features = expression.T.copy()

    # Create clear feature names:
    #
    # expression_1
    # expression_2
    # ...
    # expression_805
    features.columns = [
        f"expression_{i + 1}"
        for i in range(features.shape[1])
    ]

    # After transposing, the row index currently contains:
    #
    # G1
    # G2
    # ...
    #
    # Turn that index into a normal column.
    features.index.name = "gene_id"

    features = features.reset_index()

    # Add node_index so the feature table has exactly the
    # same numerical node IDs as our metadata table.
    node_lookup = dict(
        zip(
            genes["gene_id"],
            genes["node_index"],
        )
    )

    features.insert(
        0,
        "node_index",
        features["gene_id"].map(node_lookup),
    )

    return features


# ============================================================
# FINAL PROCESSED-DATA VALIDATION
# ============================================================

def validate_processed_data(
    metadata: pd.DataFrame,
    features: pd.DataFrame,
) -> None:
    """
    Validate the outputs BEFORE saving them.

    The most important idea:

    Row 0 must refer to G1 everywhere.
    Row 1 must refer to G2 everywhere.
    ...
    Row 1642 must refer to G1643 everywhere.

    If this alignment breaks, the graph would attach the wrong
    expression values to the wrong genes.
    """

    print("\nValidating processed node data...")

    # Metadata should have 1,643 rows.
    if len(metadata) != EXPECTED_GENE_COUNT:
        raise ValueError(
            "Processed metadata has the wrong number of genes."
        )

    # Expression feature table should also have 1,643 rows.
    if len(features) != EXPECTED_GENE_COUNT:
        raise ValueError(
            "Processed feature table has the wrong "
            "number of genes."
        )

    # Check gene order matches exactly.
    if (
        metadata["gene_id"].tolist()
        != features["gene_id"].tolist()
    ):
        raise ValueError(
            "Gene order does not match between metadata "
            "and expression features."
        )

    # Check node indices match exactly.
    if (
        metadata["node_index"].tolist()
        != features["node_index"].tolist()
    ):
        raise ValueError(
            "Node indices do not match between metadata "
            "and expression features."
        )

    # Confirm TF count survived preprocessing correctly.
    tf_count = metadata["is_tf"].sum()

    if tf_count != EXPECTED_TF_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_TF_COUNT} TF flags, "
            f"but found {tf_count}."
        )

    # There should be 805 expression columns PLUS:
    #
    # node_index
    # gene_id
    #
    # So total columns should be 807.
    expression_columns = [
        col
        for col in features.columns
        if col.startswith("expression_")
    ]

    if len(expression_columns) != EXPECTED_EXPERIMENT_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_EXPERIMENT_COUNT} "
            f"expression features per gene, but found "
            f"{len(expression_columns)}."
        )

    print("Processed node-data validation passed.")


# ============================================================
# SAVE PROCESSED FILES
# ============================================================

def save_processed_data(
    metadata: pd.DataFrame,
    features: pd.DataFrame,
) -> None:
    """
    Save the processed gene information.

    We create three outputs:

    1. gene_metadata.tsv

       Human-readable information:

           node_index
           gene_id
           biological_name
           is_tf


    2. expression_features.tsv

       Human-readable expression matrix:

           one row per gene
           805 expression features


    3. expression_features.npy

       The same expression measurements stored as a NumPy
       numerical matrix.

       Shape:

           1643 x 805

       This will be convenient later when we start using ML
       libraries.

       IMPORTANT:
       We are NOT training anything yet.
    """

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata_path = (
        PROCESSED_DIR / "gene_metadata.tsv"
    )

    features_path = (
        PROCESSED_DIR / "expression_features.tsv"
    )

    numpy_path = (
        PROCESSED_DIR / "expression_features.npy"
    )

    # --------------------------------------------------------
    # Save human-readable gene metadata
    # --------------------------------------------------------

    metadata.to_csv(
        metadata_path,
        sep="\t",
        index=False,
    )

    # --------------------------------------------------------
    # Save human-readable expression features
    # --------------------------------------------------------

    features.to_csv(
        features_path,
        sep="\t",
        index=False,
    )

    # --------------------------------------------------------
    # Save only the 805 numerical expression columns as a
    # NumPy matrix.
    #
    # We deliberately exclude:
    #
    # node_index
    # gene_id
    #
    # because those are identifiers, not expression values.
    # --------------------------------------------------------

    expression_columns = [
        col
        for col in features.columns
        if col.startswith("expression_")
    ]

    feature_matrix = features[
        expression_columns
    ].to_numpy(
        dtype=np.float32
    )

    np.save(
        numpy_path,
        feature_matrix,
    )

    print("\nSaved processed node data:")
    print(f"  {metadata_path}")
    print(f"  {features_path}")
    print(f"  {numpy_path}")


# ============================================================
# PRINT FINAL SUMMARY
# ============================================================

def print_summary(
    metadata: pd.DataFrame,
    features: pd.DataFrame,
) -> None:
    """
    Print a quick summary so we can confirm what was generated.
    """

    expression_columns = [
        col
        for col in features.columns
        if col.startswith("expression_")
    ]

    print("\n" + "=" * 60)
    print("PREPROCESSING SUMMARY")
    print("=" * 60)

    print(
        f"Genes:                    "
        f"{len(metadata)}"
    )

    print(
        f"Transcription factors:    "
        f"{metadata['is_tf'].sum()}"
    )

    print(
        f"Expression features/gene: "
        f"{len(expression_columns)}"
    )

    print(
        f"Feature matrix shape:     "
        f"{len(features)} x "
        f"{len(expression_columns)}"
    )

    print("=" * 60)


# ============================================================
# MAIN PIPELINE
# ============================================================

def main() -> None:
    """
    Run the complete node preprocessing pipeline.

    Pipeline:

        raw gene IDs
              │
              ├─────────────┐
              │             │
        raw TF list         │
              │             │
              ▼             │
        gene metadata       │
                            │
        raw expression ─────┘
              │
              ▼
        expression features

    Then validate and save everything.
    """

    print("=" * 60)
    print("GeneLink - DREAM5 Node Preprocessing")
    print("=" * 60)

    # --------------------------------------------------------
    # STEP 1:
    # Load the three raw DREAM5 files.
    # --------------------------------------------------------

    genes = load_gene_ids(
        GENE_IDS_FILE
    )

    expression = load_expression_data(
        EXPRESSION_FILE
    )

    transcription_factors = (
        load_transcription_factors(
            TF_FILE
        )
    )

    # --------------------------------------------------------
    # STEP 2:
    # Make sure all three files agree with each other.
    # --------------------------------------------------------

    validate_raw_data(
        genes=genes,
        expression=expression,
        transcription_factors=transcription_factors,
    )

    # --------------------------------------------------------
    # STEP 3:
    # Create one metadata record for every gene.
    # --------------------------------------------------------

    metadata = create_gene_metadata(
        genes=genes,
        transcription_factors=transcription_factors,
    )

    # --------------------------------------------------------
    # STEP 4:
    # Transpose expression data so every graph node gets its
    # own 805-value expression vector.
    # --------------------------------------------------------

    features = create_expression_features(
        genes=genes,
        expression=expression,
    )

    # --------------------------------------------------------
    # STEP 5:
    # Confirm gene IDs and numerical node indices align
    # perfectly between the two processed outputs.
    # --------------------------------------------------------

    validate_processed_data(
        metadata=metadata,
        features=features,
    )

    # --------------------------------------------------------
    # STEP 6:
    # Save everything inside data/processed/.
    # --------------------------------------------------------

    save_processed_data(
        metadata=metadata,
        features=features,
    )

    # --------------------------------------------------------
    # STEP 7:
    # Show a final summary.
    # --------------------------------------------------------

    print_summary(
        metadata=metadata,
        features=features,
    )

    print("\nNode preprocessing complete.")


# ============================================================
# SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()