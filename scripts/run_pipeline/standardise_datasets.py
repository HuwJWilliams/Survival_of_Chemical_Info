"""
Script to standardise datasets into the format the pipeline by doing:
    - Removal of chemical fragment- and metal-containing SMILES strings
    - Standardisation of SMILES
    - Averaging target ranges and removes value uncertainties
    - Randomly shuffling the data
    - Creating sequential IDs
    - Updating the pathing.json
"""

# region Imports and Pathing
from pathlib import Path
import sys
import argparse
import json

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
SRC_DIR = SCRIPTS_DIR / "src"

# Pathing
sys.path.insert(0, str(SRC_DIR / "pathing"))
from get_paths import addNewDatasetPaths, getPaths

# Standardisation Functions
sys.path.insert(0, str(SRC_DIR / "datasets"))
from standardise_dataset import cleanAndSaveDataset

PATHS_JSON = str(SRC_DIR / "pathing" / "paths.json")
paths = getPaths(PATHS_JSON)
# endregion

# region Argument Parsing
parser = argparse.ArgumentParser(
    description="Standardising datasets for the workflow and adding cleaned paths to global pathing"
)
supported_names = list(paths["raw_data"].keys())

parser.add_argument(
    "--dataset-name",
    required=True,
    choices=supported_names,
    help="Name of raw data paths in pathing json. Current supported:\n"
    f"{supported_names}",
)

parser.add_argument(
    "--target-col", required=True, help="Name of target column in the raw dataset file"
)

parser.add_argument(
    "--smiles-col", required=True, help="Name of smiles column in raw dataset file"
)

parser.add_argument(
    "--id-col", default=None, help="Name of the id column in raw dataset file"
)

parser.add_argument(
    "--id-prefix",
    default="id",
    help="If no existing ID column, set this to create unique ID column (e.g., bp-1, ...)",
)

parser.add_argument("--shuffle-data", action="store_true", help="Flag to shuffle data")

parser.add_argument(
    "--plot-distribution",
    action="store_true",
    help="Plots distribution of Lipinski features in dataset",
)

parser.add_argument(
    "--rename-cols", default=None, help="Dictionary to rename columns (e.g., \n \
        {'Boiling Point {measured, converted}: 'Boiling _Point, 'Unit, K' : 'Unit'})"
)
# endregion

# region Preparing Arguments for Functions
args = parser.parse_args()
dataset_name = args.dataset_name

raw_data_path = Path(paths["raw_data"][dataset_name])
data_name = raw_data_path.stem
cleaned_data_path = (
    paths["imp_dirs"]["datasets_dir"] / dataset_name / f"{data_name}_cleaned.csv"
)

usecols = [args.smiles_col, args.target_col]
if bool(args.id_col):
    usecols.append(args.id_col)

rename_cols = json.loads(args.rename_cols) if args.rename_cols else None

# endregion

# region Calling Functions
cleanAndSaveDataset(
    in_path=raw_data_path,
    out_path=cleaned_data_path,
    usecols=usecols,
    target_col=args.target_col,
    smiles_col=args.smiles_col,
    id_col=args.id_col,
    rename=rename_cols,
    shuffle_rows=args.shuffle_data,
    plot_feature_distribution=args.plot_distribution,
    id_prefix=args.id_prefix,
)

addNewDatasetPaths(
    dataset_key=dataset_name,
    target_file=cleaned_data_path.name,
    dataset_prefix=dataset_name,
    dataset_folder_name=dataset_name,
    json_path=PATHS_JSON,
)
# endregion
