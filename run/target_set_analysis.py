# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
from pathlib import Path

RUN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN_DIR / "config"))

from config import PATHING_JSON_PATH, SRC_DIR, TARGET_COLUMNS

print(f"\n{SRC_DIR}\n")

sys.path.insert(0, str(SRC_DIR / "pathing"))
from get_paths import getPaths

FULL_PATHING = getPaths(PATHING_JSON_PATH)

for key, targ_path in FULL_PATHING["targets"].items():
    df = pd.read_csv(targ_path)

    target_col = TARGET_COLUMNS[key]
    output_path = targ_path.parent / f"{key}_distribution.png"

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.histplot(data=df, x=target_col, stat="count", bins="auto", ax=ax)

    ax.set_xlabel(target_col)
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of {target_col}")

    fig.tight_layout()
    fig.savefig(
        targ_path.parent / f"{key}_{target_col}_distribution.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)
