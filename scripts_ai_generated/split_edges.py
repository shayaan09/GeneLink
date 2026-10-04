from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

# We use a fixed random seed so that the shuffle is reproducible.
#
# Without a fixed seed:
#   Run 1 could put G24 -> G1437 in training.
#   Run 2 could put G24 -> G1437 in testing.
#
# With a fixed seed, we get the exact same split every time.
RANDOM_SEED = 42


# Fractions of the gold-standard edges that will go into
# training, validation, and testing.
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


# DREAM5 Network 1 should contain exactly 4,012 known
# regulatory connections.
EXPECTED_EDGE_COUNT = 4012


# ============================================================
# FILE PATHS
# ============================================================

# __file__ is the location of this script.
#
# If this script is:
#
# GeneLink/scripts/split_edges.py
#
# then:
#
# Path(__file__).resolve().parent
#
# gives:
#
# GeneLink/scripts/
SCRIPT_DIR = Path(__file__).resolve().parent


# .parent again moves us from:
#
# GeneLink/scripts/
#
# to:
#
# GeneLink/
PROJECT_ROOT = SCRIPT_DIR.parent


# Location of the untouched DREAM5 gold-standard file.
RAW_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "DREAM5_NetworkInference_GoldStandardSigned_Network1 - in silico.tsv"
)


# Folder where our generated train/validation/test files
# will be saved.
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"


# ============================================================
# LOAD THE GOLD STANDARD
# ============================================================

def load_edges(path: Path) -> pd.DataFrame:
    """
    Load the DREAM5 gold-standard regulatory edges.

    The original DREAM5 file has no header.

    Each row looks like:

        G24    G1437    +

    where:

        source = controller / transcription factor
        target = gene being controlled
        sign   = + for activation or - for repression

    Returns
    -------
    pd.DataFrame
        A table with columns:
        source, target, sign
    """

    # Check that the file actually exists before trying
    # to read it.
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find gold-standard file:\n{path}"
        )

    # sep="\t" means this is a TSV:
    # values are separated by tabs.
    #
    # header=None tells pandas that the file does NOT
    # already contain column names.
    #
    # names=[...] gives the three columns useful names.
    edges = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["source", "target", "sign"],
    )

    print(f"Loaded {len(edges)} regulatory edges.")

    return edges


# ============================================================
# VALIDATE THE ORIGINAL GOLD STANDARD
# ============================================================

def validate_edges(edges: pd.DataFrame) -> None:
    """
    Check that the raw DREAM5 gold-standard data looks correct.

    We perform these checks BEFORE splitting so that we do not
    accidentally build train/validation/test files from bad data.

    If anything is wrong, the function raises an error and stops
    the script.
    """

    print("\nValidating original gold-standard file...")

    # --------------------------------------------------------
    # Check 1: Correct number of edges
    # --------------------------------------------------------

    if len(edges) != EXPECTED_EDGE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_EDGE_COUNT} edges, "
            f"but found {len(edges)}."
        )

    # --------------------------------------------------------
    # Check 2: Correct columns
    # --------------------------------------------------------

    expected_columns = ["source", "target", "sign"]

    if list(edges.columns) != expected_columns:
        raise ValueError(
            f"Expected columns {expected_columns}, "
            f"but found {list(edges.columns)}."
        )

    # --------------------------------------------------------
    # Check 3: No missing values
    # --------------------------------------------------------

    # Example of something invalid:
    #
    # G24    <missing>    +
    #
    # isna() finds missing values.
    # any().any() checks whether ANY cell is missing.
    if edges.isna().any().any():
        raise ValueError(
            "The gold-standard file contains missing values."
        )

    # --------------------------------------------------------
    # Check 4: Every sign must be + or -
    # --------------------------------------------------------

    valid_signs = {"+", "-"}

    # unique() gets every different sign found in the file.
    found_signs = set(edges["sign"].unique())

    invalid_signs = found_signs - valid_signs

    if invalid_signs:
        raise ValueError(
            f"Found invalid regulatory signs: {invalid_signs}"
        )

    # --------------------------------------------------------
    # Check 5: No duplicate source-target pairs
    # --------------------------------------------------------

    # We care about the connection itself:
    #
    # G24 -> G1437
    #
    # There should not be multiple copies of the same connection.
    duplicate_edges = edges.duplicated(
        subset=["source", "target"]
    )

    if duplicate_edges.any():
        duplicate_count = duplicate_edges.sum()

        raise ValueError(
            f"Found {duplicate_count} duplicate regulatory edges."
        )

    # --------------------------------------------------------
    # Check 6: No self-connections
    # --------------------------------------------------------

    # A self-edge would look like:
    #
    # G24 -> G24
    #
    # We already know DREAM5 Network 1 should not contain these.
    self_edges = edges["source"] == edges["target"]

    if self_edges.any():
        self_edge_count = self_edges.sum()

        raise ValueError(
            f"Found {self_edge_count} self-connections."
        )

    print("Original gold-standard validation passed.")


# ============================================================
# SPLIT THE EDGES
# ============================================================

def split_edges(
    edges: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Randomly divide the known regulatory connections into:

        70% training
        15% validation
        15% testing

    IMPORTANT:
    We are splitting CONNECTIONS, not genes.

    All 1,643 genes will still exist later.

    What changes is whether a particular known arrow is:
        - visible during training
        - hidden for validation
        - hidden for final testing

    The random seed makes the split reproducible.
    """

    print("\nShuffling and splitting edges...")

    # --------------------------------------------------------
    # Step 1: Shuffle all 4,012 edges
    # --------------------------------------------------------

    # sample(frac=1) means:
    # "take 100% of the rows, but return them in random order."
    #
    # random_state ensures that the random order is the same
    # every time the script runs.
    #
    # reset_index(drop=True) removes the old row numbers after
    # shuffling and replaces them with:
    #
    # 0, 1, 2, 3, ...
    shuffled = edges.sample(
        frac=1,
        random_state=RANDOM_SEED,
    ).reset_index(drop=True)

    total_edges = len(shuffled)

    # --------------------------------------------------------
    # Step 2: Calculate split sizes
    # --------------------------------------------------------

    # 70% of 4,012:
    #
    # 4012 * 0.70 = 2808.4
    #
    # int() drops the decimal, giving 2,808 training edges.
    train_size = int(total_edges * TRAIN_RATIO)

    # 15% of 4,012:
    #
    # 4012 * 0.15 = 601.8
    #
    # We use round() here, giving 602 validation edges.
    val_size = round(total_edges * VAL_RATIO)

    # Whatever remains goes into the test set.
    #
    # This is safer than separately calculating all three
    # percentages because it guarantees:
    #
    # train + validation + test = exactly 4,012
    test_size = total_edges - train_size - val_size

    # --------------------------------------------------------
    # Step 3: Slice the shuffled table
    # --------------------------------------------------------

    # Training gets the first chunk.
    train_edges = shuffled.iloc[
        :train_size
    ].copy()

    # Validation gets the next chunk.
    val_start = train_size
    val_end = train_size + val_size

    val_edges = shuffled.iloc[
        val_start:val_end
    ].copy()

    # Test gets everything remaining.
    test_edges = shuffled.iloc[
        val_end:
    ].copy()

    print(f"Training edges:   {len(train_edges)}")
    print(f"Validation edges: {len(val_edges)}")
    print(f"Test edges:       {len(test_edges)}")

    return train_edges, val_edges, test_edges


# ============================================================
# HELPER: TURN EDGES INTO A SET
# ============================================================

def edges_to_set(edges: pd.DataFrame) -> set:
    """
    Convert a DataFrame of edges into a Python set.

    Example DataFrame row:

        source = G24
        target = G1437
        sign   = +

    becomes the tuple:

        ("G24", "G1437", "+")

    Sets make it very easy to test whether two groups overlap.
    """

    return set(
        edges[
            ["source", "target", "sign"]
        ].itertuples(
            index=False,
            name=None,
        )
    )


# ============================================================
# VALIDATE THE SPLIT
# ============================================================

def validate_split(
    original_edges: pd.DataFrame,
    train_edges: pd.DataFrame,
    val_edges: pd.DataFrame,
    test_edges: pd.DataFrame,
) -> None:
    """
    Verify that our random splitting procedure did not lose,
    duplicate, or leak any edges.

    We want to prove:

    1. train + validation + test = original number of edges
    2. train and validation do not overlap
    3. train and test do not overlap
    4. validation and test do not overlap
    5. combining all three exactly recreates the original set
    """

    print("\nValidating train/validation/test split...")

    # --------------------------------------------------------
    # Check 1: Total edge count
    # --------------------------------------------------------

    combined_count = (
        len(train_edges)
        + len(val_edges)
        + len(test_edges)
    )

    if combined_count != len(original_edges):
        raise ValueError(
            "Split sizes do not add up to the original "
            "number of edges."
        )

    # --------------------------------------------------------
    # Convert all DataFrames into sets
    # --------------------------------------------------------

    original_set = edges_to_set(original_edges)
    train_set = edges_to_set(train_edges)
    val_set = edges_to_set(val_edges)
    test_set = edges_to_set(test_edges)

    # --------------------------------------------------------
    # Check 2: Train and validation overlap
    # --------------------------------------------------------

    train_val_overlap = train_set.intersection(val_set)

    if train_val_overlap:
        raise ValueError(
            f"Found {len(train_val_overlap)} edges appearing "
            "in BOTH training and validation."
        )

    # --------------------------------------------------------
    # Check 3: Train and test overlap
    # --------------------------------------------------------

    train_test_overlap = train_set.intersection(test_set)

    if train_test_overlap:
        raise ValueError(
            f"Found {len(train_test_overlap)} edges appearing "
            "in BOTH training and testing."
        )

    # --------------------------------------------------------
    # Check 4: Validation and test overlap
    # --------------------------------------------------------

    val_test_overlap = val_set.intersection(test_set)

    if val_test_overlap:
        raise ValueError(
            f"Found {len(val_test_overlap)} edges appearing "
            "in BOTH validation and testing."
        )

    # --------------------------------------------------------
    # Check 5: Reconstruct original gold standard
    # --------------------------------------------------------

    # "|" means set union.
    #
    # This combines all edges from:
    #
    # train + validation + test
    combined_set = (
        train_set
        | val_set
        | test_set
    )

    # The combined set should be EXACTLY the same as the
    # original gold-standard set.
    if combined_set != original_set:
        raise ValueError(
            "Combining the three splits does not exactly "
            "recreate the original gold standard."
        )

    print("Split validation passed.")


# ============================================================
# SAVE THE SPLITS
# ============================================================

def save_splits(
    train_edges: pd.DataFrame,
    val_edges: pd.DataFrame,
    test_edges: pd.DataFrame,
) -> None:
    """
    Save the three generated edge splits as TSV files.

    Unlike the original DREAM5 gold-standard file, our generated
    files WILL contain column headers:

        source    target    sign
    """

    print("\nSaving split files...")

    # mkdir creates the output directory.
    #
    # parents=True:
    # creates parent folders too if necessary.
    #
    # exist_ok=True:
    # do not crash if data/splits already exists.
    SPLIT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_path = SPLIT_DIR / "train_edges.tsv"
    val_path = SPLIT_DIR / "val_edges.tsv"
    test_path = SPLIT_DIR / "test_edges.tsv"

    # sep="\t" means save as TSV.
    #
    # index=False prevents pandas from adding an unnecessary
    # extra column containing row numbers.
    #
    # header=True keeps:
    #
    # source    target    sign
    train_edges.to_csv(
        train_path,
        sep="\t",
        index=False,
        header=True,
    )

    val_edges.to_csv(
        val_path,
        sep="\t",
        index=False,
        header=True,
    )

    test_edges.to_csv(
        test_path,
        sep="\t",
        index=False,
        header=True,
    )

    print(f"Saved training edges to:\n  {train_path}")
    print(f"Saved validation edges to:\n  {val_path}")
    print(f"Saved test edges to:\n  {test_path}")


# ============================================================
# MAIN PIPELINE
# ============================================================

def main() -> None:
    """
    Run the complete edge-splitting pipeline.

    Order:

        1. Load raw DREAM5 gold standard
        2. Validate raw edges
        3. Shuffle and split edges
        4. Validate the split
        5. Save generated split files
    """

    print("=" * 60)
    print("GeneLink - DREAM5 Edge Split")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. Load the original gold-standard edges
    # --------------------------------------------------------

    edges = load_edges(RAW_FILE)

    # --------------------------------------------------------
    # 2. Make sure the original data is valid
    # --------------------------------------------------------

    validate_edges(edges)

    # --------------------------------------------------------
    # 3. Randomly split the connections
    # --------------------------------------------------------

    train_edges, val_edges, test_edges = split_edges(edges)

    # --------------------------------------------------------
    # 4. Make sure the split is safe
    # --------------------------------------------------------

    validate_split(
        original_edges=edges,
        train_edges=train_edges,
        val_edges=val_edges,
        test_edges=test_edges,
    )

    # --------------------------------------------------------
    # 5. Save the three generated files
    # --------------------------------------------------------

    save_splits(
        train_edges=train_edges,
        val_edges=val_edges,
        test_edges=test_edges,
    )

    print("\n" + "=" * 60)
    print("Edge split complete.")
    print("=" * 60)


# ============================================================
# SCRIPT ENTRY POINT
# ============================================================

# Python sets __name__ to "__main__" when this file is run
# directly, for example:
#
#     python scripts/split_edges.py
#
# This prevents main() from automatically running if we later
# import functions from this file into another Python script.
if __name__ == "__main__":
    main()