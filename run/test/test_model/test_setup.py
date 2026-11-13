"""Script to run tests to check the full workflow"""

# %% ===== Imports
import math
import shutil
from pathlib import Path
import sys

import pandas as pd
import pytest

pytestmark = [
    pytest.mark.filterwarnings("ignore:.*least populated class.*:UserWarning"),
]

# %% ==== Pathing
ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = ROOT / "run"
SCRIPTS_DIR = ROOT / "scripts"
SRC_DIR = SCRIPTS_DIR / "src"
DATASET_DIR = ROOT / "datasets"
EXPECTED_DATA = RUN_DIR / "test" / "expected_test_results"

FEATURES = [
    "rdkit",
    "mordred",
]

sys.path.insert(0, str(SRC_DIR / "pathing"))
from get_paths import (
    createPathingJSON,
    addRawDataPaths,
    addNewDatasetPaths,
    loadJSON,
    getPaths,
)

# %% ========== PATHING JSON SETUP TESTING
CLEAN_CREATED_PATHS = False
TEST_JSON_NAME = "test.json"
TEST_JSON_PATH = SRC_DIR / "pathing" / TEST_JSON_NAME
EXPECTED_PATHING_PATH = EXPECTED_DATA / "expected_pathing.json"
EXPECTED_FEATURE_PATHS = {
    "rdkit": EXPECTED_DATA / "expected_rdkit_features.csv",
    "mordred": EXPECTED_DATA / "expected_mordred_features.csv",
}

EXPECTED_INT_PERFORMANCE = EXPECTED_DATA / "expected_int_perf.json"
MODEL_OUTPUT_DIR = RUN_DIR / "test" / "test_model"
TARGET_COLUMN = "Solubility"
REQUIRED_EXTERNAL_METRICS = [
    "bias",
    "sdep",
    "mse",
    "rmse",
    "r2",
    "r_pearson",
    "p_pearson",
    "r_spearman",
]


def assert_performance_close(actual_perf, expected_perf, rel=1e-2, abs_tol=1e-2):
    for metric, expected_value in expected_perf.items():
        assert metric in actual_perf
        assert actual_perf[metric] == pytest.approx(
            expected_value,
            rel=rel,
            abs=abs_tol,
        )


def assert_valid_external_performance(performance):
    assert isinstance(performance, dict)

    for metric in REQUIRED_EXTERNAL_METRICS:
        assert metric in performance
        assert performance[metric] is not None
        assert math.isfinite(float(performance[metric]))


@pytest.fixture
def clean_test_json():
    TEST_JSON_PATH.unlink(missing_ok=True)
    yield
    TEST_JSON_PATH.unlink(missing_ok=True)


@pytest.fixture(scope="session")
def created_paths():
    paths = []
    yield paths

    if not CLEAN_CREATED_PATHS:
        return

    for path in reversed(paths):
        path = Path(path)
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        elif path.is_file():
            path.unlink(missing_ok=True)


def test_pathing_setup(clean_test_json):
    createPathingJSON(json_name=TEST_JSON_NAME)

    addRawDataPaths(
        raw_data_paths=[str(DATASET_DIR / "test" / "dummy_data.csv")],
        set_names=["test"],
        json_name=TEST_JSON_NAME,
    )

    addNewDatasetPaths(
        dataset_key="test",
        target_file="dummy_data.csv",
        dataset_prefix="TEST",
        dataset_folder_name="test",
        feature_sets=FEATURES,
        json_name=TEST_JSON_NAME,
    )

    generated_json = loadJSON(TEST_JSON_PATH)
    expected_json = loadJSON(EXPECTED_PATHING_PATH)

    assert generated_json == expected_json


# %% ========== FEATURE GENERATION TESTING
sys.path.insert(0, str(SRC_DIR / "datasets"))
FULL_TEST_PATHING = getPaths(EXPECTED_PATHING_PATH)

from feature_generator import FeatureGenerator


@pytest.mark.parametrize("feature_set", FEATURES)
def test_generate_features(feature_set, created_paths):
    input_df = pd.read_csv(FULL_TEST_PATHING["targets"]["test"], index_col="ID")
    output_path = Path(FULL_TEST_PATHING["full_features"]["test"][feature_set])
    output_path.unlink(missing_ok=True)
    created_paths.append(output_path)

    fg = FeatureGenerator(feature_set)
    fg.calcBatchFeatures(
        smiles_ls=input_df["SMILES"].tolist(),
        id_ls=input_df.index.to_list(),
        fpath=output_path,
        batch_size=25,
    )

    assert output_path.exists()
    generated_df = pd.read_csv(output_path, index_col="ID")

    assert len(generated_df) > 0
    assert generated_df.index.notna().all()

    expected_feature_path = EXPECTED_FEATURE_PATHS.get(feature_set)
    if expected_feature_path is not None:
        expected_df = pd.read_csv(expected_feature_path, index_col="ID")
        pd.testing.assert_frame_equal(
            generated_df,
            expected_df,
            check_exact=False,
            rtol=1e-10,
            atol=1e-12,
        )


# %% ========== SINGLE PROPERTY PREDICTION TESTING
sys.path.insert(0, str(SRC_DIR / "datasets"))
from feature_cleaning import cleanFeatureDF

sys.path.insert(0, str(SRC_DIR / "models"))
from transfer_model import TL


def test_single_property_prediction(created_paths):
    X = pd.read_csv(EXPECTED_FEATURE_PATHS["rdkit"], index_col="ID")
    clean_X, _ = cleanFeatureDF(X)

    y = pd.read_csv(DATASET_DIR / "test" / "dummy_data.csv", index_col="ID")
    y = y[[TARGET_COLUMN]]

    common = clean_X.index.intersection(y.index)
    final_X = clean_X.loc[common]
    final_y = y.loc[common]

    assert final_X.shape == (25, 172)
    assert final_y.shape == (25, 1)

    data = final_X.join(final_y)

    assert data.shape == (25, 173)

    train_data = data.iloc[:20]
    test_data = data.iloc[20:]

    shutil.rmtree(MODEL_OUTPUT_DIR, ignore_errors=True)
    created_paths.append(MODEL_OUTPUT_DIR)

    model = TL()
    final_model, best_params, internal_perf, feat_importance = (
        model.trainSingleTargetRFModel(
            data=train_data,
            target_column=TARGET_COLUMN,
            hyper_params={
                "n_estimators": [5],
                "max_features": ["sqrt"],
                "max_depth": [5],
                "min_samples_split": [2],
                "min_samples_leaf": [1],
            },
            n_resamples=1,
            test_size=0.2,
            cv_splits=2,
            random_seed=42,
            save_models=False,
            save_path=MODEL_OUTPUT_DIR,
            n_jobs=1,
            trim_3xIQR=False,
        )
    )

    assert final_model is not None
    assert isinstance(best_params, dict)
    assert isinstance(internal_perf, dict)
    expected_int_perf = loadJSON(EXPECTED_INT_PERFORMANCE)
    assert_performance_close(internal_perf, expected_int_perf)
    assert not feat_importance.empty

    _, _, external_perf = model.predictSingleTargetRF(
        model=final_model,
        data=test_data,
        target_column=TARGET_COLUMN,
        calc_perf=True,
        save_preds=True,
        save_path=MODEL_OUTPUT_DIR,
    )
    assert_valid_external_performance(external_perf)

    performance_path = MODEL_OUTPUT_DIR / "performance.json"
    assert performance_path.exists()

    preds_path = MODEL_OUTPUT_DIR / "preds.csv.gz"
    assert preds_path.exists()


# %% ========== CROSS FEATURE PREDICTION TESTING
def test_cross_feature_prediction(created_paths):

    CFP_TARGET_COLUMNS = [
        # regression
        "ABC_mordred",
        "ABCGG_mordred",
        "SpAbs_A_mordred",
        # binary classification
        "naHRing_mordred",
        # multiclass classification
        "n6HRing_mordred",
    ]

    output_dir = RUN_DIR / "test" / "test_cfp_model"
    shutil.rmtree(output_dir, ignore_errors=True)
    created_paths.append(output_dir)

    train_ft = pd.read_csv(EXPECTED_FEATURE_PATHS["rdkit"], index_col="ID")
    train_ft = train_ft.drop(columns=["SMILES"], errors="ignore")
    train_ft = train_ft.select_dtypes(include="number")

    test_ft = pd.read_csv(EXPECTED_FEATURE_PATHS["mordred"], index_col="ID")
    test_ft = test_ft[CFP_TARGET_COLUMNS]

    assert list(test_ft.columns) == CFP_TARGET_COLUMNS

    assert train_ft.shape[1] > 0
    assert test_ft.shape[1] == 5

    assert test_ft.shape[1] == 5

    model = TL()

    perf_df = model.trainMultiTargetRFModels(
        features_df=train_ft,
        targets_df=test_ft,
        hyper_params={
            "n_estimators": [5],
            "max_features": ["sqrt"],
            "max_depth": [5],
            "min_samples_split": [2],
            "min_samples_leaf": [1],
        },
        output_csv="cfp_test.csv",
        n_resamples=1,
        test_size=0.3,
        cv_splits=2,
        random_seed=42,
        save_models=False,
        save_path=output_dir,
        trim_by_percentile=False,
        min_training_samples=10,
        min_samples_per_class=3,
        batch_size=1,
        n_jobs=1,
    )

    assert isinstance(perf_df, pd.DataFrame)
    assert not perf_df.empty
    assert (output_dir / "cfp_test.csv").exists()
