"""
Functions used for joining datasets
"""

# region Imports
import pandas as pd
from glob import glob
import numpy as np
from pathlib import Path
import sys
from rdkit import Chem
from rdkit.Chem import Descriptors

# endregion

# region Pathing
SCRIPTS_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = SCRIPTS_DIR / "src"

sys.path.insert(0, str(SRC_DIR / "pathing"))
sys.path.insert(0, str(SRC_DIR / "datasets"))

from get_paths import getPaths
from analyse_datasets import checkLipinskiCriteria

paths = getPaths()
# endregion


# region Function Definitions
def makeUniqueSMILES(
    properties: list = ["Boiling_Point", "LogD", "LD50", "pKa", "pIC50"],
    target_dict: dict = paths["targets"],
    override_full_csv: bool = True,
):
    """
    Creates a Data Frame which contains only unique SMILES from across all specified properties
    """

    full_smi_df = pd.DataFrame()

    # Looping over all target data frames
    for prop, path in target_dict.items():
        if prop == "all":
            continue
        try:
            df = pd.read_csv(
                path,
                # usecols=["SMILES", "ID"],
                index_col="ID",
            )
        except FileNotFoundError as e:
            print("Property file not found. Skipping.\n{e}")
            continue

        # Concatenate all target data frames and drop the duplicates
        full_smi_df = pd.concat([full_smi_df, df], axis=0)

    full_smi_df = full_smi_df.drop_duplicates(subset="SMILES")

    # Keep all available propery data
    cols_to_keep = ["SMILES"] + properties
    full_smi_df = full_smi_df[cols_to_keep]

    out_path = target_dict["all"]

    # Saving the full csv
    if override_full_csv or not out_path.exists():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        full_smi_df.to_csv(out_path, index_label="ID")
        print(f"Saved '{out_path}'")
    else:
        print(f"'{out_path}' already exists. Skipping.")


def _find_bad_ids(df: pd.DataFrame, file: str, n=20):
    idx = df.index

    # If index isn't strings, make it so for regex checks
    idx_s = idx.astype(str)

    extracted = idx_s.str.extract(r"(\d+)")[0]  # digits or NaN

    # NaN index values OR "no digits found"
    bad_mask = extracted.isna() | idx.isna()

    if bad_mask.any():
        bad_vals = idx_s[bad_mask].unique()
        print(f"\n[BAD IDS] file={file}")
        print(
            f"  bad_rows={bad_mask.sum()} unique_bad={len(bad_vals)} sample={bad_vals[:n]}"
        )
        print("  index dtype:", idx.dtype)
        # show which rows they correspond to
        print(df.iloc[list(pd.Series(bad_mask).to_numpy().nonzero()[0])].head(3))


def combineAllFeats(
    full_smi_df: pd.DataFrame | str | Path = paths["targets"]["all"],
    feat_set_ls: list = [
        "rdkit",
        "mordred",
        "chemberta",
        "molformer",
        "chembertasey",
        "molformer-c3-1b",
        "maccs",
        "selformer",
        "morgan",
    ],
    feature_paths: dict = paths["full_features"],
    cols_to_drop: list[str] | None = None,
    properties: list = None,
    save: bool = True,
    save_path: str | Path = paths["full_features"]["all"],
    align_common_ids: bool = False,
    max_mw: float | None = None,
    max_atoms: int | None = None,
    feat_path: str | Path = "all",
    lipinski_criteria: bool = False,
    lipinski_mw: int = 600,
    lipinski_logp: float = 6,
    lipinski_n_hbd: int = 6,
    lipinski_n_hba: int = 11,
):
    built_feature_dfs = {}

    if properties is None:
        properties = list(feature_paths.keys())[:-1]

    if isinstance(full_smi_df, (str, Path)):
        full_smi_df = pd.read_csv(full_smi_df, index_col="ID")

    keep_ids = set(full_smi_df.index.astype(str))

    if max_atoms is not None:
        atom_keep_ids = set(
            _get_ids_below_atom_threshold(
                full_smi_df=full_smi_df,
                max_atoms=max_atoms,
            )
        )
        keep_ids &= atom_keep_ids
        print(f"Keeping {len(keep_ids)} IDs after atom-count filter (< {max_atoms})")

    if max_mw is not None:
        mw_keep_ids = set(
            _get_ids_below_mw_threshold(
                full_smi_df=full_smi_df,
                max_mw=max_mw,
            )
        )
        keep_ids &= mw_keep_ids
        print(f"Keeping {len(keep_ids)} IDs after molecular-weight filter (< {max_mw})")

    if lipinski_criteria:
        lipinski_keep_ids = set(
            checkLipinskiCriteria(
                df=feature_paths["all"]["rdkit"],
                mw=lipinski_mw,
                logp=lipinski_logp,
                n_hbd=lipinski_n_hbd,
                n_hba=lipinski_n_hba,
            )
        )
        keep_ids &= {str(idx) for idx in lipinski_keep_ids}
        print(
            "Keeping %d IDs after Lipinski filter "
            "(mw<%s, logp<%s, n_hbd<%s, n_hba<%s)"
            % (
                len(keep_ids),
                lipinski_mw,
                lipinski_logp,
                lipinski_n_hbd,
                lipinski_n_hba,
            )
        )

    for desc_emb in feat_set_ls:
        out_path = feature_paths[feat_path][desc_emb]
        print(f"- Processing {desc_emb}")
        print(f"- Output path {out_path}")

        full_feat_df = pd.DataFrame()

        for prop in properties:
            print(f"    - Processing {prop}")
            file_prefix = feature_paths[prop][desc_emb]
            files = glob(str(file_prefix))

            temp_df = pd.DataFrame()
            for file in files:
                df = pd.read_csv(file, index_col="ID", low_memory=False)

                if cols_to_drop:
                    df = df.drop(
                        columns=[col for col in cols_to_drop if col in df.columns]
                    )

                # diagnose raw IDs
                _find_bad_ids(df, file)
                temp_df = pd.concat([temp_df, df], axis=0)

            # safe sort
            temp_df = temp_df.sort_index(
                key=lambda x: pd.to_numeric(
                    x.astype(str).str.extract(r"(\d+)")[0], errors="coerce"
                ).fillna(10**18)
            )

            full_feat_df = pd.concat([full_feat_df, temp_df], axis=0)

        print(f"Length of DF: {len(full_feat_df)}")
        print("Rows with any NaN:", full_feat_df.isna().any(axis=1).sum())

        if full_feat_df.index.has_duplicates:
            duplicate_count = int(full_feat_df.index.duplicated(keep="first").sum())
            raise ValueError(
                f"{desc_emb} has {duplicate_count} duplicate IDs after joining. "
                "Fix duplicate IDs upstream before joining feature datasets."
            )

        if max_atoms is not None or max_mw is not None or lipinski_criteria:
            full_feat_df = full_feat_df.loc[
                full_feat_df.index.astype(str).isin(keep_ids)
            ].copy()
            print(
                f"Filtered {desc_emb} dataframe to {len(full_feat_df)} rows "
                "after SMILES-based thresholds"
            )

        print("No feature columns dropped during joining")
        built_feature_dfs[desc_emb] = full_feat_df

    if align_common_ids and built_feature_dfs:
        aligned_dfs, common_ids = getCommonIDs(
            df_path_ls=list(built_feature_dfs.values()),
            save=False,
        )
        for desc_emb, aligned_df in zip(built_feature_dfs.keys(), aligned_dfs):
            built_feature_dfs[desc_emb] = aligned_df
            print(f"Retained {len(common_ids)} common IDs for {desc_emb}")

    if save:
        for desc_emb, full_feat_df in built_feature_dfs.items():
            out_path = feature_paths[feat_path][desc_emb]
            out_path.parent.mkdir(parents=True, exist_ok=True)
            full_feat_df.to_csv(out_path, index_label="ID")


def getCommonIDs(
    df_path_ls: list[pd.DataFrame | str | Path] = paths["full_features"][
        "all"
    ].values(),
    output_path_ls: list[Path] | None = None,
    save: bool = True,
):

    common_ids = None
    loaded_dfs = []

    for df_path in df_path_ls:
        print(f"Processing {df_path}")
        if isinstance(df_path, pd.DataFrame):
            df = df_path.copy()
            if df.index.name != "ID":
                df.index.name = "ID"
        else:
            df = pd.read_csv(df_path, index_col="ID", low_memory=False)

        if df.index.has_duplicates:
            df = df.loc[~df.index.duplicated(keep="first")]

        loaded_dfs.append(df)
        id_set = set(df.index)

        if common_ids is None:
            common_ids = id_set

        else:
            common_ids &= id_set

        print(f"Current intersection size: {len(common_ids)}")

        if not common_ids:
            break

    common_ids = sorted(common_ids) if common_ids is not None else []
    aligned_dfs = [
        df.loc[df.index.intersection(common_ids)].copy() for df in loaded_dfs
    ]

    if save:
        if output_path_ls is None:
            output_path_ls = paths["full_features"]["all"].values()

        for df_path, df, output_path in zip(df_path_ls, aligned_dfs, output_path_ls):
            print(f"Generating common IDs between all features:")
            print(f"{df_path} -> {output_path}")
            df.to_csv(output_path, index_label="ID")

    return aligned_dfs, common_ids


def _load_all_targets_df(
    full_smi_df: pd.DataFrame | str | Path = paths["targets"]["all"],
) -> pd.DataFrame:

    if isinstance(full_smi_df, pd.DataFrame):
        df = full_smi_df.copy()
    else:
        df = pd.read_csv(full_smi_df, index_col="ID", low_memory=False)

    if "SMILES" not in df.columns:
        raise ValueError("Input dataframe must contain a 'SMILES' column.")

    return df


def _get_ids_below_atom_threshold(
    full_smi_df: pd.DataFrame | str | Path = paths["targets"]["all"],
    max_atoms: int = 100,
    smiles_col: str = "SMILES",
) -> list[str]:

    if max_atoms == 0:
        return full_smi_df.index.tolist()

    df = _load_all_targets_df(full_smi_df=full_smi_df)
    mols = df[smiles_col].apply(Chem.MolFromSmiles)
    atom_counts = mols.apply(
        lambda mol: mol.GetNumAtoms() if mol is not None else np.nan
    )

    keep_ids = df.index[atom_counts < max_atoms]
    return keep_ids.astype(str).tolist()


def _get_ids_below_mw_threshold(
    full_smi_df: pd.DataFrame | str | Path = paths["targets"]["all"],
    max_mw: float = 500.0,
    smiles_col: str = "SMILES",
) -> list[str]:

    if max_mw == 0:
        return full_smi_df.index.tolist()

    df = _load_all_targets_df(full_smi_df=full_smi_df)
    mols = df[smiles_col].apply(Chem.MolFromSmiles)
    mol_wts = mols.apply(
        lambda mol: Descriptors.MolWt(mol) if mol is not None else np.nan
    )

    keep_ids = df.index[mol_wts < max_mw]
    return keep_ids.astype(str).tolist()


# endregion
