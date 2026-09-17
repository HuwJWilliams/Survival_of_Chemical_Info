# Feature Investigation Workflow

This file will help you understand and use each of the scripts in the `run/` directory.

## Overall Structure Flowchart

```mermaid
flowchart TD
    A([Environment Setup])
    B([Importing Data])
    C([Running Setup])
    D([Generate<br/>Features])
    E([Single Property<br/>Prediction])
    F([Cross-Feature<br/>Prediction])
    G([Single Property<br/>Prediction Analysis])
    H([Cross Feature<br/>Prediction Analysis])


    A --> B
    B --> C
    C --> D
    D --> E
    D --> F
    E --> G
    F --> H
```

## How To Run The Project
### 1. Environment Setup

Install the environment `tlp_py312` using the `config/environment.yml` file:

#### Run:
```bash
module load miniforge/python-3.12.10/***Version***
mamba env create --file config/environment.yml
mamba activate tlp_py312
```

#### Note: 
The original project was run using miniforge v25.3.0

### 2. Importing Data
Import your datasets into the `/datasets/` directory, having each unique dataset in its own directory. For example:

```text
project_dir/
    ├── ...
    └── datasets/
        ├── Boiling_Point/
        |   └── bp_mols.csv
        ├── ...
        └── pKa/
            └── pka_mols.csv
```

### 3. Editing Configuration File
There is a config file `config/config.py` in the `run/` directory. This file contains information which is referred to throughout the project scripts:

`TARGET_COLUMNS`
Dictionary for which the dataset target column can be located from the dataset identifier. (e.g., dataset "bp" has the target column "Boiling_Point")

`SUPPORTED_FEATURE_SETS`
List of feature sets which the project currently supports.

`SUPPORTED_TARGET_SETS`
List of targets which have been added to the pathing JSON.

`TRANSFORMER_FEATURE_SPECS`
Dictionary of transformer models used and their specifications used in the project.

`CFP_ANALYSIS_METRICS`
Dictionary specifying the metrics used during each cross-feature analysis job.

`PP_ANALYSIS_METRICS`
Dictionary specifying the metrics used for each single property prediction job.


### 4. Create Global Pathing
This project uses a single JSON for all of its pathing. The script does two things: 1- Sets up the global pathing JSON for the project, and 2- Allows you to append new datasets to the JSON.

#### Script & Arguments
**`setup.py`**

`--create-pathing` <br/>
Flag argument to create the pathing JSON, use this flag on project setup but not when simply adding new datasets.

`--set-paths` <br/>
Argument for which you specify the paths to your raw imported data (e.g., `~/project_dir/datasets/Boiling_Point/bp_mols.csv`).

`--set-names` <br/>
Argument to set the keys within the JSON file which will be used to reference that dataset throughout the project (e.g. Boiling Point -> bp).

#### Example Calls:
```bash
# To create the pathing JSON
python setup.py \
    --create-pathing \
    --set-paths  `~/project_dir/datasets/Boiling_Point/bp_mols.csv` \
    --set-names bp

# To append to the pathing JSON
python setup.py \
    --set-paths  `~/project_dir/datasets/pKa/pka_mols.csv` \
    ---set-names pka

```

#### Note: 
Example pathing JSON:

```json
{
 "imp_dirs": {
    "scripts_dir": "...",
    "..." : "..."
 },
 "raw_data": {
    "bp": "~/project_dir/datasets/Boiling_Point/bp_mols.csv",
    "...": "..."
 },
 "targets": {
    "bp": "~/project_dir/datasets/Boiling_Point/cleaned_bp_mols.csv"
 },
 "full_features": {
    "bp": {
        "rdkit":   "~/project_dir/datasets/Boiling_Point/bp_rdkit.csv",
        "...": "..."
        },
 },
 "prediction_output_dirs": {
    "rf": {},
    "cross_feature_predictions": {}
 }
}

```

### 5. Generating Features
Script to generate the features for the workflow.

#### Script & Arguments
`generate_features.py`

`--task`
The pathing dataset key to generate features for (e.g., "bp")

`--feature-set`
The set of features to generate. These must be in line with the keys in `SUPPORTED_FEATURE_SETS`. (e.g., "rdkit")

`--batch-size`
The number of molecules to process per batch (default = 100,000).

`--fine-tune`
Argument flag to carry out transformer fine-tuning using a portion of target molecules.

`--target-col`
Column to use for fine-tuning (defaults to the configured target for `--task`).

`--fine-tune-test-frac`
The fraction of target molecules to select for fine-tuning (default = 0.2).

`--split-by`
The way in which molecules are selected for fine-tuning.
**"random"** for a random selection, and **"scaffold"** for a scaffold split selection

#### Example Use:
```bash
python generate_features.py \
    --task bp \
    --feature-set chemberta-dc \
```


#### Note:
There are more arguments for optimisation parameters which can be found in the `generate_features.py` script, or by using:
```bash
python generate_features.py --help
```

### 6. Running Cross-Feature Predictions
Cross-Feature Prediction (CFP) experiments train RF models on one input feature representation to predict each feature in another representation. 
For example, the **tr_mordred_pred_rdkit** run would (theoretically) train 216 individual models to predict each individual RDKit descriptor.

#### Script & Arguments
**`run_cfp.py`**

`--train`
The feature set to train the models on.

`--target`
The target feature set to predict.

`--property`
The dataset to carry out predictions on (e.g., "bp").

`--save-dir`
The directory key to save results in (default = "cross_feature_predictions")

`--test-size`
Size of the train/test split (0.3 = 30 % train)

`--skip-existing`
Flag to allow continuation of runs.

`--lipinski-mols`
Flag to filter out molecules which dont fit the relaxed Lipinski criteria (MW=600, HBD=6, HBA=11, LogP=6)

`--minimum-targs`
The minimum number of valid targets to train a model with (default = 2500)

`--save-feat-imp`
Flag to save feature importance data

`--save-models`
Flag to save each trained model (takes up large amounts of space).

#### Example Call
```bash
python run_cfp.py \
    --train mordred \
    --target rdkit \
    --property bp \
    --skip-existing \
    --lipinski-mols \
    --save-feat-imp
```

#### Note:
There are more arguments for optimisation parameters which can be found in the `run_cfp.py` script, or by using:
```bash
python run_cfp.py --help
```

### 7. Running Cross-Feature Prediction Analysis
#### Script & Arguments
**`cfp_analysis.py`**

`--results-dir`
Pathing key for the results directory (use the same as used when calling `run_cfp.py`).

`--run-all`
Flag to run all available CFP analysis.

`--property`
The dataset to run cfp analysis for (e.g., "bp").

`--run-experiment` (Optional)
Argument to specify analysis for individual experiments.

`--run-avg` (Optional)
Flag to average all experiment performances.

`--exclude-low-var` (Optional)
Flag to exclude low variance columns from the analysis plots

`--var-threshold` (Optional)
If the fraction of entities with the same common value to filter out targets by (e.g., 0.8 = 80 % of values are non-variant).

`--skip-cols` (Optional)
Argument to skip some columns (e.g. Ipc)

`--plot-poor-distributions` (Optional)
Plot value distributions for targets predicted below a performance threshold.

`--radar-task` (Optional)
The task type (regression, classification) to plot radar plots for (defualt = "regression").

`--poor-distribution-threshold` (Optional)
The threshold used with **--plot-poor-distributions**

`--skip-group-member-bars` (Optional)
Flag to skip individual descriptor group/member bar plots

#### Example Call:
```bash
python cfp_analysis.py \
    --results-dir cross_feature_predictions \
    --property bp \
    --run-experiment tr_mordred_pred_rdkit \
    --exclude-low-var \
    --skip-cols Ipc
```

#### Output
```text
<trained/pred experiment dir>/
├── low_variance_columns.txt
├── excluded_columns.txt
├── regression_group_perf.csv
├── classification_group_perf.csv
├── multiclass_classification_group_perf.csv
├── <experiment>_group_radar_<task>_<metric>
├── <experiment>_<group>_<pred>_bar_<metric>
├── <experiment>_full_task_bar_<task>_<metric>.png
├── <experiment>_group_fraction_lt0p7_summary.csv
└── <experiment>_group_task_fraction_<task>_lt0p7.png
```

### 8. Running Single Property Predictions
Single property predictions run RF models trained on a speficied feature type to predict a target value.

#### Script & Arguments
**`run_single_prediction.py`**

`--predict-on`
The target set to train an RF model to predict. Must be the key in the pathing JSON (e.g., 'bp', 'logd')

`--feature-set`
The feature set used to train the RF model on

`--target-column`  (Optional)
The column name within the target CSV. Omit if using the default specified in `config/config.py`

`--identifier`  (Optional)
Argument to uniquely name the run.

`--max-nan-frac`  (Optional)
The fraction of NaN rows to drop a column. Default = 0

`--corr-threshold`  (Optional)
The threshold for dropping correlated feature columns. Default = 0.9 (90%)

`--additional-features`
Argument to include features from other sets when training the RF model. Can be set to include individual, groups, or entire feature sets using a dictionary shown below:

```json
{
    "feature_set": "mordred",
    "feature_group": ["Autocorrelation", "..."],
    "individual_features": ["ABC", "..."]
}
```

#### Example Call
```bash
python run_single_prediction.py \
    --predict-on bp \
    --feature-set rdkit \
    --additional-features '{"feature_set": "mordred", "feature_group": ["Autocorrelation", "MoRSE"]}'
```

### 9. Running Single Property Prediction Analysis
The analysis suite for the single predictions.

#### Script & Arguments
**`pp_analysis.py`**

`--properties` (Optional)
List of the properties which will be analysed. Must be the key within the pathing JSON (e.g., "bp", "logd", etc). Defaults to all available properties.

`--feature-sets` (Optional)
List of the feature sets whose performance will be analysed. Defaults to the
feature names registered in the selected properties' RF paths, including legacy
names. Random-split fine-tuned features remain excluded. Missing performance
files are skipped per feature; properties without a configured target column or
target path are skipped with a message.

`--save-dir` (Optional)
The directory to save all analaysis plots and results to. Defaults to 'pp_analysis' in the results directory.

`--analysis-metrics` (Optional)
Metrics to use in analysis plots. Defaults to Pearson r.

Fine-tuned versus pretrained difference plots recompute both scores on the
intersection of molecule IDs with finite targets and predictions for that pair.
They use the saved mean out-of-sample prediction per molecule in
`last_20pct_pred.csv.gz`. The 3xIQR comparisons additionally apply the same target
bounds to both models. Different model pairs can have different intersections.
Internal difference plots are skipped because the saved aggregate internal
scores cannot be restricted to shared molecule IDs.

`ft_matched_performances.csv` records both matched scores and molecule counts;
`ft_differences.csv` includes `n_matched`. The per-property `ft_differences`
directories also contain the aligned targets and predictions with their IDs.
General performance plots still describe each model's available evaluation set.
Use a new `--save-dir` when rerunning to keep old unmatched plots separate.

#### Example Call
```bash
python pp_analysis.py \
    --properties bp logd pka_basic \
    --feature-sets rdkit mordred
```

#### Example Outputs
```text
pp_analysis/
├── internal_average_performances.csv
├── external_average_performances.csv
├── external_lipinski_average_performances.csv
├── external_3xIQR_average_performances.csv
├── ft_differences.csv
├── heatmaps/
├── grouped_bars/
├── ft_differences/
└── <property>/
    ├── internal/
    ├── external/
    └── external_3xIQR/
```

### 10. Running feature importance analysis for single predictions

#### Script & Arguments
**`feature_importance_analysis.py`**

`--properties` (Optional)
Properties to run the FI analysis for. Defaults to all in the pathing JSON.

`--feature-sets` (Optional)
Feature sets to run the FI analysis for. Defaults to all in the pathing JSON.

`--save_dir` (Optional)
Directory to have the FI results to. Defaults to 'feature_importance/' in the results directory.

`--top-n-feats` (Optional)
The number of features to show which have the highest feature importance. Defaults to 50.

#### Example Call
```bash
python feature_importance_analysis.py
    --properties bp logd pka_basic
    --feature-set rdkit mordred
    --top-n-feats 100
```
#### Example Outputs
```text
feature_importance/
├── descriptor_group_importance.csv
└── heatmaps/
|   ├── rdkit_descriptor_group_average_importance.csv
|   ├── rdkit_descriptor_group_summative_importance.csv
|   ├── rdkit_descriptor_group_max_importance.csv
|   ├── rdkit_descriptor_group_average_importance_heatmap.png
|   ├── rdkit_descriptor_group_average_importance_heatmap_no_values.png
|   ├── rdkit_descriptor_group_max_importance_heatmap.png
|   └── rdkit_descriptor_group_max_importance_heatmap_no_values.png
└── <property>/
    └── <feature_set>/
        ├── <property>_<feature_set>_average_feature_importance.png
        ├── feature_importance_df.csv
        ├── feature_importance_display_df.csv
        ├── <property>_<feature_set>_summative_descriptor_group_importance.png
        └── summative_descriptor_group_importance_df.csv
```

#### Note
The heatmaps will only work for the feature sets whose grouping have been explicitly set (e.g., MACCS, Mordred and RDKit)
