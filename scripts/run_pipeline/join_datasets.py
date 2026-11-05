"""
Script to join all datasets together
"""

# region Script Functionality
# region Imports
import argparse
from pathlib import Path
import sys

# endregion

# region Path Setup
SCRIPTS_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = SCRIPTS_DIR / "src"

sys.path.insert(0, str(SRC_DIR / "pathing"))
sys.path.insert(0, str(SRC_DIR / "datasets"))
sys.path.insert(0, str(SCRIPTS_DIR / "config"))

from get_paths import getPaths
from join_desc_funcs import makeUniqueSMILES, combineAllFeats
from pipeline_config import DEFAULT_TARGET_COLUMNS, SUPPORTED_FEATURE_SETS

paths = getPaths()
# endregion

# region Argument Setup
parser = argparse.ArgumentParser(
    description="Combine target datasets and feature tables into joined modelling datasets."
)

parser.add_argument(
    "--target-properties",
    nargs="+",
    required=True,
    choices=DEFAULT_TARGET_COLUMNS.keys(),
    help="Target dataset keys from paths.json, e.g. bp logd pka.",
)

parser.add_argument(
    "--feature-list",
    nargs="+",
    required=True,
    choices=SUPPORTED_FEATURE_SETS,
    help="Feature sets to join, e.g. rdkit mordred chemberta.",
)

parser.add_argument(
    "--save-path",
    default="all",
    choices=paths["full_features"].keys(),
    help="Destination key within paths['full_features'].",
)

parser.add_argument(
    "--override-existing",
    action="store_true",
    help="Overwrite the combined target file if it already exists.",
)

parser.add_argument(
    "--align-common-ids",
    action="store_true",
    help="Keep only IDs shared by all selected feature tables.",
)

parser.add_argument(
    "--max-mw",
    type=float,
    default=None,
    help="Optional maximum molecular-weight filter.",
)

parser.add_argument(
    "--max-atoms",
    type=int,
    default=None,
    help="Optional maximum atom-count filter.",
)

parser.add_argument(
    "--fit-lipinski",
    action="store_true",
    help="Keep only molecules passing the configured Lipinski-style thresholds.",
)

parser.add_argument(
    "--lipinski-mw-limit",
    type=float,
    default=600.0,
    help="Lipinski molecular-weight threshold.",
)

parser.add_argument(
    "--logp-limit",
    type=float,
    default=6.0,
    help="Lipinski logP threshold.",
)

parser.add_argument(
    "--hbd-limit",
    type=int,
    default=6,
    help="Lipinski hydrogen-bond donor threshold.",
)

parser.add_argument(
    "--hba-limit",
    type=int,
    default=11,
    help="Lipinski hydrogen-bond acceptor threshold.",
)
# endregion
# endregion

# region Running Script
if __name__ == "__main__":
    args = parser.parse_args()

    target_columns = [DEFAULT_TARGET_COLUMNS[key] for key in args.target_properties]

    makeUniqueSMILES(
        properties=target_columns,
        override_full_csv=args.override_existing,
    )

    combineAllFeats(
        feat_set_ls=args.feature_list,
        properties=args.target_properties,
        align_common_ids=args.align_common_ids,
        max_mw=args.max_mw,
        max_atoms=args.max_atoms,
        feat_path=args.save_path,
        lipinski_criteria=args.fit_lipinski,
        lipinski_mw=args.lipinski_mw_limit,
        lipinski_logp=args.logp_limit,
        lipinski_n_hbd=args.hbd_limit,
        lipinski_n_hba=args.hba_limit,
    )
# endregion
