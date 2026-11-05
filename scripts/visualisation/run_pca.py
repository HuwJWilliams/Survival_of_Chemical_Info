"""
Separated script to run any of the PCA plotting functions defined in src/visualisation/vis.py
"""

# region Imports and Pathing
import sys
import pandas as pd
from pathlib import Path
import numpy as np
import argparse
import random

# --- Paths
FILE_DIR = Path(__file__).parent
SCRIPTS_DIR = FILE_DIR.parent
SRC_DIR = SCRIPTS_DIR / "src"

sys.path.insert(0, str(SRC_DIR / "pathing"))
from get_paths import getPaths

PATHS = getPaths()

sys.path.insert(0, str(SRC_DIR / "visualisation"))
from vis import Visualise

sys.path.insert(0, str(SRC_DIR / "misc"))
from misc_fns import (
    center_rows,
    scale_rows,
    center_columns,
    scale_columns,
    get_ids_in_mw_range,
)

sys.path.insert(0, str(SCRIPTS_DIR / "config"))
from pipeline_config import (
    DEFAULT_TARGET_COLUMNS,
    SUPPORTED_FEATURE_SETS,
    resolve_target_column,
)

# endregion

# region Class and Parser Setup
v = Visualise(save_all=False)

parser = argparse.ArgumentParser(description="Generating PCA analysis plots")

feature_ls = list(PATHS["full_features"]["all"].keys())
# endregion

# region Argument Parsing
parser.add_argument(
    "--type", choices=["mol", "feat", "joined"], help="Type of PCA analysis to do.\n \
          'mol' shows the PCA analysis of molecules \n \
          'feat' shows the PCA analysis of features \n \
          'joined' shows molecule PCA on joined columns from all --feats"
)

parser.add_argument(
    "--feats",
    nargs="+",
    choices=SUPPORTED_FEATURE_SETS,
    default=["rdkit", "mordred"],
    help=f"Feature sets to perform analysis with. Choices:\n{SUPPORTED_FEATURE_SETS}",
)

parser.add_argument(
    "--feats-for-pca",
    nargs="+",
    choices=SUPPORTED_FEATURE_SETS,
    help="Feature sets used to fit PCA space (feature mode). If omitted, uses --feats.",
)

parser.add_argument(
    "--feat-groups",
    nargs="+",
    choices=SUPPORTED_FEATURE_SETS,
    help="Feature groups/sources to display on PCA plots (feature mode). If omitted, uses --feats-for-pca.",
)

targs = list(DEFAULT_TARGET_COLUMNS.keys())
parser.add_argument(
    "--targs",
    nargs="+",
    choices=targs,
    default=["bp", "logd"],
    help=f"Target properties to perform analysis with. Choices:\n{targs}",
)

parser.add_argument(
    "--scale-all", action="store_true", help="Flag to scale both rows and columns"
)

parser.add_argument("--scale-r", action="store_true", help="Flag to scale rows only")

parser.add_argument("--scale-c", action="store_true", help="Flag to scale columns only")

parser.add_argument(
    "--center-all", action="store_true", help="Flag to center both rows and columns"
)

parser.add_argument("--center-r", action="store_true", help="Flag to center rows only")

parser.add_argument(
    "--center-c", action="store_true", help="Flag to center columns only"
)

parser.add_argument(
    "--filter-mw", nargs=2, type=int, help="Two molecular weights bounds required"
)

parser.add_argument("--n-mols", type=int, help="Number of molecules to process in PCA")

parser.add_argument(
    "--seed",
    type=int,
    default=random.randint(0, 2**32),
    help="Random seed for making reproducible results",
)

parser.add_argument(
    "--n_comp", type=int, default=5, help="Number of components to generate"
)

parser.add_argument(
    "--biplot", action="store_true", help="Flag to save the bi plots for PC1 and PC2"
)

parser.add_argument(
    "--pc-x",
    type=int,
    default=1,
    help="Principal component to plot on X-axis of biplot",
)

parser.add_argument(
    "--pc-y",
    type=int,
    default=2,
    help="Principal component to plot on Y-axis of biplot",
)

parser.add_argument(
    "--remove-points",
    action="store_true",
    help="Remove individual points on the biplot",
)

parser.add_argument(
    "--n-loadings",
    type=int,
    default=15,
    help="Number of loadings to plot on the biplot (sorted by highest weight)",
)

parser.add_argument(
    "--not-pca", action="store_true", help="Flag to not run full pca plot"
)

parser.add_argument(
    "--heatmap",
    action="store_true",
    help="Flag to save a PCA loadings heatmap (top features by loading magnitude)",
)

parser.add_argument(
    "--heatmap-top-n",
    type=int,
    default=100,
    help="Number of top loadings/features to show on the heatmap",
)

parser.add_argument(
    "--save-path",
    default=str(PATHS["imp_dirs"]["results_dir"] / "pca_analysis"),
    help="Path to save plots to",
)

parser.add_argument("--save-name", default="pca", help="Name to save plot under")

args = parser.parse_args()

# --- Defining commonly used variables
feat_ls = args.feats
targ_ls = args.targs
feats_for_pca = args.feats_for_pca if args.feats_for_pca else feat_ls
feat_groups = args.feat_groups if args.feat_groups else feats_for_pca

mw_bounds = args.filter_mw

scaling_cfg = {
    "center_rows": args.center_all or args.center_r,
    "center_cols": args.center_all or args.center_c,
    "scale_rows": args.scale_all or args.scale_r,
    "scale_cols": args.scale_all or args.scale_c,
}


# endregion

# region Data Setup


# -- DF Processing (scaling, centering and trimming)
def trim_df(df, selected_ids=[], n_mols=None, seed=42):
    """Filter by IDs (optional) and sample rows."""
    working_df = df

    if selected_ids:
        working_df = working_df.loc[working_df.index.intersection(selected_ids)]

    if n_mols is None:
        n_mols = len(working_df)
    n_mols = min(n_mols, len(working_df))

    return working_df.sample(n=n_mols, random_state=seed, replace=False).copy()


def scale_df(df, cfg):
    """Apply centering/scaling to numeric columns only"""
    out = df.copy()

    num_cols = out.select_dtypes(include=[np.number]).columns
    if len(num_cols) == 0:
        return out

    num_df = out[num_cols].copy()

    if cfg["center_rows"]:
        num_df = center_rows(num_df)
    if cfg["scale_rows"]:
        num_df = scale_rows(num_df)
    if cfg["center_cols"]:
        num_df = center_columns(num_df)
    if cfg["scale_cols"]:
        num_df = scale_columns(num_df)

    out[num_cols] = num_df
    return out


def debug_df(name, df):
    num = df.select_dtypes(include=[np.number])
    print(f"\n[{name}] shape={df.shape}, numeric_cols={num.shape[1]}")
    if num.shape[1] == 0:
        return
    row_var = num.var(axis=1)
    col_var = num.var(axis=0)
    print(
        f"[{name}] row-var median={row_var.median():.3e}, near0={(row_var < 1e-12).mean():.1%}"
    )
    print(
        f"[{name}] col-var median={col_var.median():.3e}, near0={(col_var < 1e-12).mean():.1%}"
    )
    print(
        f"[{name}] abs max={num.abs().to_numpy().max():.3e}, std mean={num.std(axis=0).mean():.3e}"
    )


mw_filtered_ids = []
mw_bounds = args.filter_mw
if mw_bounds is not None:
    rdkit_df = pd.read_csv(PATHS["full_features"]["all"]["rdkit"], index_col=0)
    mw_filtered_ids = get_ids_in_mw_range(
        rdkit_df,
        min_mw=mw_bounds[0],
        max_mw=mw_bounds[1],
    )

# endregion


# region Performing PCA

if args.type == "mol":
    print("Performing molecule PCA analysis.")

    for feat in feat_ls:
        pca_frames = []
        loadings_df_out = None

        for targ in targ_ls:
            temp_df_path = str(PATHS["full_features"][targ][feat]).replace("*", "1")
            temp_df = pd.read_csv(temp_df_path, index_col=0)
            temp_df = trim_df(
                temp_df,
                selected_ids=mw_filtered_ids if mw_filtered_ids else None,
                n_mols=args.n_mols,
                seed=args.seed,
            )
            temp_df["Source"] = targ
            pca_frames.append(temp_df)

        final_pca_df = pd.concat(pca_frames, axis=0)
        final_pca_df = scale_df(final_pca_df, scaling_cfg)

        if not args.not_pca:
            fig, pca_df, loadings_df, abs_loadings_df = v.plotPCA(
                data_dict={"Data": final_pca_df},
                n_components=args.n_comp,
                plot_area=False,
                save_plot=True,
                save_path=args.save_path,
                save_fname=f"{args.save_name}_{feat}",
                axis_fontsize=14,
                label_fontsize=10,
                legend_fontsize=10,
            )
            loadings_df_out = loadings_df

        if args.biplot:
            if args.remove_points:
                point_size, point_alpha, show_legend = 0, 0, False
            else:
                point_size, point_alpha, show_legend = 30, 0.4, True

            fig, pca_df, loadings_df, abs_loadings_df = v.plotPCABiplot(
                data_dict={"Data": final_pca_df},
                pc_x=1,
                pc_y=2,
                n_components=args.n_comp,
                top_n_loadings=args.n_loadings,
                save_plot=True,
                save_path=args.save_path,
                save_fname=f"{args.save_name}_{feat}_biplot",
                axis_fontsize=14,
                label_fontsize=10,
                legend_fontsize=10,
                point_size=point_size,
                point_alpha=point_alpha,
                show_legend=show_legend,
            )
            loadings_df_out = loadings_df

        if args.heatmap:
            if loadings_df_out is None:
                print(
                    f"Skipping heatmap for {feat}: no loadings available (enable --biplot or full PCA)."
                )
            else:
                v.plotLoadingsHeatmap(
                    loadings_df=loadings_df_out,
                    pc_x=args.pc_x,
                    pc_y=args.pc_y,
                    top_n=args.heatmap_top_n,
                    save_plot=True,
                    save_path=args.save_path,
                    save_fname=f"{args.save_name}_{feat}_loadings_heatmap",
                )


elif args.type == "feat":
    print("Performing feature PCA analysis.")

    final_pca_df = pd.DataFrame()
    loadings_df_out = None

    for feat in feats_for_pca:
        temp_df = pd.read_csv(PATHS["full_features"]["all"][feat], index_col=0)
        temp_df = temp_df.select_dtypes(include=[np.number]).copy()
        temp_df = trim_df(
            temp_df, selected_ids=mw_filtered_ids, n_mols=args.n_mols, seed=args.seed
        )
        temp_df = temp_df.T
        temp_df = scale_df(temp_df, scaling_cfg)
        temp_df["Source"] = feat

        final_pca_df = pd.concat([final_pca_df, temp_df], axis=0)
        final_pca_df = final_pca_df.dropna(axis=1)

    if not args.not_pca:
        fig, pca_df, loadings_df, abs_loadings_df = v.plotPCA(
            data_dict={"Data": final_pca_df},
            n_components=args.n_comp,
            plot_sources=feat_groups,
            plot_area=False,
            save_plot=True,
            save_path=args.save_path,
            save_fname=f"{args.save_name}_feat",
            axis_fontsize=14,
            label_fontsize=10,
            legend_fontsize=10,
            scale=False,
        )
        loadings_df_out = loadings_df

    if args.biplot:
        if args.remove_points:
            point_size, point_alpha, show_legend = 0, 0, False
        else:
            point_size, point_alpha, show_legend = 30, 0.4, True

        fig, pca_df, loadings_df, abs_loadings_df = v.plotPCABiplot(
            data_dict={"Data": final_pca_df},
            pc_x=1,
            pc_y=2,
            n_components=args.n_comp,
            plot_sources=feat_groups,
            top_n_loadings=args.n_loadings,
            save_plot=True,
            save_path=args.save_path,
            save_fname=f"{args.save_name}_feat_biplot",
            axis_fontsize=14,
            label_fontsize=10,
            legend_fontsize=10,
            scale=False,
            point_size=point_size,
            point_alpha=point_alpha,
            show_legend=show_legend,
        )
        loadings_df_out = loadings_df

    if args.heatmap:
        if loadings_df_out is None:
            print(
                "Skipping heatmap: no loadings available (enable --biplot or full PCA)."
            )
        else:
            v.plotLoadingsHeatmap(
                loadings_df=loadings_df_out,
                pc_x=args.pc_x,
                pc_y=args.pc_y,
                top_n=args.heatmap_top_n,
                save_plot=True,
                save_path=args.save_path,
                save_fname=f"{args.save_name}_loadings_heatmap",
            )

elif args.type == "joined":
    print("Performing joined-feature molecule PCA analysis.")

    joined_blocks = []
    shared_ids = None
    loadings_df_out = None

    for feat in feat_ls:
        temp_df = pd.read_csv(PATHS["full_features"]["all"][feat], index_col=0)
        temp_df = temp_df.select_dtypes(include=[np.number]).copy()

        if mw_filtered_ids:
            temp_df = temp_df.loc[temp_df.index.intersection(mw_filtered_ids)]

        if shared_ids is None:
            shared_ids = temp_df.index
        else:
            shared_ids = shared_ids.intersection(temp_df.index)

        joined_blocks.append((feat, temp_df))

    if shared_ids is None or len(shared_ids) == 0:
        raise ValueError("No common molecule IDs found across selected feature sets.")

    aligned_blocks = []
    for feat, temp_df in joined_blocks:
        aligned_df = temp_df.loc[shared_ids].copy()
        aligned_blocks.append(aligned_df)

    final_pca_df = pd.concat(aligned_blocks, axis=1).dropna(axis=0)
    final_pca_df = trim_df(
        final_pca_df,
        selected_ids=None,
        n_mols=args.n_mols,
        seed=args.seed,
    )
    final_pca_df = scale_df(final_pca_df, scaling_cfg)
    final_pca_df["Source"] = "joined"

    if not args.not_pca:
        fig, pca_df, loadings_df, abs_loadings_df = v.plotPCA(
            data_dict={"Data": final_pca_df},
            n_components=args.n_comp,
            plot_area=False,
            save_plot=True,
            save_path=args.save_path,
            save_fname=f"{args.save_name}_joined",
            axis_fontsize=14,
            label_fontsize=10,
            legend_fontsize=10,
            scale=False,
        )
        loadings_df_out = loadings_df

    if args.biplot:
        if args.remove_points:
            point_size, point_alpha, show_legend = 0, 0, False
        else:
            point_size, point_alpha, show_legend = 30, 0.4, True

        fig, pca_df, loadings_df, abs_loadings_df = v.plotPCABiplot(
            data_dict={"Data": final_pca_df},
            pc_x=args.pc_x,
            pc_y=args.pc_y,
            n_components=args.n_comp,
            top_n_loadings=args.n_loadings,
            save_plot=True,
            save_path=args.save_path,
            save_fname=f"{args.save_name}_joined_biplot",
            axis_fontsize=14,
            label_fontsize=10,
            legend_fontsize=10,
            scale=False,
            point_size=point_size,
            point_alpha=point_alpha,
            show_legend=show_legend,
        )
        loadings_df_out = loadings_df

    if args.heatmap:
        if loadings_df_out is None:
            print(
                "Skipping heatmap: no loadings available (enable --biplot or full PCA)."
            )
        else:
            v.plotLoadingsHeatmap(
                loadings_df=loadings_df_out,
                pc_x=args.pc_x,
                pc_y=args.pc_y,
                top_n=args.heatmap_top_n,
                save_plot=True,
                save_path=args.save_path,
                save_fname=f"{args.save_name}_joined_loadings_heatmap",
            )


# endregion
