"""
Run this script to set up the project
"""

# %% ===== Python Imports =====
import sys
from pathlib import Path
import argparse

# %% ===== Project Imports & Pathing Setup =====
SCRIPTS_DIR = Path(__file__).parents[1] / "scripts"
RUN_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(RUN_DIR / "config"))

sys.path.insert(0, str(SCRIPTS_DIR / "src" / "pathing"))
from get_paths import createPathingJSON, addRawDataPaths
from config import PATHING_JSON_NAME

# %% ===== Argument Parsing =====
parser = argparse.ArgumentParser(
    description="Correctly setting up the project directory"
)

parser.add_argument(
    "--create-pathing",
    action="store_true",
    help="Flag to create pathing json for project",
)

parser.add_argument(
    "--set-paths",
    nargs="+",
    required=False,
    help="Adding new raw data set to the project directory",
)

parser.add_argument(
    "--set-names", nargs="+", required=False, help="Set names of the new raw data sets"
)

args = parser.parse_args()

# %% ===== Create Project Pathing and Add New Data Paths =====
if args.create_pathing:
    createPathingJSON(json_name=PATHING_JSON_NAME)

if bool(args.set_names) != bool(args.set_paths):
    raise ValueError("Set both '--set-paths' and '--set-names' to save new raw data")

else:
    addRawDataPaths(
        raw_data_paths=args.set_paths,
        set_names=args.set_names,
        json_name=args.json_name,
    )
