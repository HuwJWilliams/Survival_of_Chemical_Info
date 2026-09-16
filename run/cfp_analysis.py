"""
Run analysis for the Cross-Feature Prediction (CFP) experiments
"""

# %% ===== Python Imports =====
import sys
import pandas as pd
from pathlib import Path
import argparse
from glob import glob

# %% ===== Project Imports & Pathing Setup=====
RUN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN_DIR / "config"))
sys.path.insert(0, str(RUN_DIR / "fns"))

from config import SRC_DIR, PATHING_JSON_PATH
from cfp_analysis_fns import *

sys.path.insert(0, str(SRC_DIR / "pathing"))
from get_paths import getPaths

FULL_PATHING = getPaths(PATHING_JSON_PATH)
RESULTS_DIR = FULL_PATHING["imp_dirs"]["results_dir"]

sys.path.insert(0, str(SRC_DIR / "datasets"))

# %% ===== Setting Up The Visualisation Class  =====
cfp = listCFPExperiments(
    FULL_PATHING["prediction_output_dirs"]["cross_feature_predictions"]
)
lcfp = listCFPExperiments(
    FULL_PATHING["prediction_output_dirs"]["lipinski_cross_feature_predictions"]
)
unique_exp_names = sorted(set(cfp + lcfp))


# %% ===== Argument Parsing =====
parser = argparse.ArgumentParser(description="Generating cross-feature analysis")

parser.add_argument(
    "--results-dir",
    help="Directory to look for cross-feature predictions.\
          Put here the name of dir as it appears in the pathing json (e,g,. 'cross_feature_predictions')",
)

parser.add_argument(
    "--run-all",
    action="store_true",
    help="Flag to run all available cross-prediction analysis",
)

parser.add_argument(
    "--property",
    default="all",
    help="Dataset/property key to analyse, e.g. bp, logd, pka, all.",
)

parser.add_argument(
    "--run-experiment",
    nargs="+",
    help="Name of the experiments ran.",
)

parser.add_argument(
    "--exclude-low-var",
    action="store_true",
    help="Flag to exclude low variance columns",
)

parser.add_argument(
    "--show-var",
    action="store_true",
    help="Flag to show the variance of target features",
)

parser.add_argument(
    "--var-threshold",
    type=float,
    default=0.8,
    help="Fraction of entries which have the same common value \n"
    "(i.e., 0.8 = 20 % of values are different from the most common)",
)

parser.add_argument(
    "--skip-cols",
    nargs="+",
    help="Feature columns to skip from feature/member bar plots",
)

parser.add_argument("--radar-task", default="regression")

parser.add_argument(
    "--run-avg",
    action="store_true",
    help="Flag to average across all specified experiments",
)

parser.add_argument(
    "--plot-poor-distributions",
    action="store_true",
    help="Plot raw descriptor distributions for group members below a performance threshold",
)

parser.add_argument(
    "--poor-distribution-threshold",
    type=float,
    default=0.6,
    help="Performance threshold used with --plot-poor-distributions",
)

parser.add_argument(
    "--skip-group-member-bars",
    action="store_true",
    help="Skip individual descriptor group/member bar plots",
)

args = parser.parse_args()


# %% ===== Running Analysis =====
property_dataset = args.property.lower()

cfp_dir = resolveCFPDir(
    prediction_output_dirs=FULL_PATHING["prediction_output_dirs"],
    result_dir=args.results_dir,
    property_dataset=property_dataset,
)

if args.run_all:
    exp_list = list(cfp_dir.keys())
else:
    exp_list = args.run_experiment

if not exp_list:
    raise ValueError("No experiments specified. Use --run-experiment or --run-all.")

if args.run_avg:
    exp_perf_df, exp_dir, exp_name, pred, averaged_train_sets = (
        averageExperimentPerformance(
            exp_list=exp_list,
            cfp_dir=cfp_dir,
        )
    )

    feature_source = args.property or "all"
    pred_ft_df_path = Path(FULL_PATHING["full_features"][feature_source][pred])
    path_ls = glob(str(pred_ft_df_path))

    pred_ft_df_ls = [pd.read_csv(f, index_col=0) for f in path_ls]
    pred_ft_df = pd.concat(pred_ft_df_ls, axis=0)

    runCFPAnalysisForPerformanceDF(
        exp_perf_df=exp_perf_df,
        pred_ft_df=pred_ft_df,
        pred=pred,
        exp_name=exp_name,
        exp_dir=exp_dir,
        args=args,
    )

else:
    for exp in exp_list:
        split_name = exp.split("_")
        pred = split_name[1]

        exp_dir = Path(cfp_dir[exp])
        exp_perf_df_path = exp_dir / f"{exp}.csv"
        exp_perf_df = pd.read_csv(exp_perf_df_path, index_col=0)

        feature_source = args.property or "all"
        pred_ft_df_path = Path(FULL_PATHING["full_features"][feature_source][pred])
        path_ls = glob(str(pred_ft_df_path))

        pred_ft_df_ls = [pd.read_csv(f, index_col=0) for f in path_ls]
        pred_ft_df = pd.concat(pred_ft_df_ls, axis=0)

        runCFPAnalysisForPerformanceDF(
            exp_perf_df=exp_perf_df,
            pred_ft_df=pred_ft_df,
            pred=pred,
            exp_name=exp,
            exp_dir=exp_dir,
            args=args,
        )
