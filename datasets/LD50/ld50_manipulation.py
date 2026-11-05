# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %%
data = pd.read_csv("./raw_ochem_ld50_molecules.csv")
columns = data.columns

print("Columns:")
print(columns)

unique_organisms = data["Organism"].unique()

print("Organisms")
print(unique_organisms)

new_columns = [
    "SMILES",
    "LD50",
    "UNIT {LD50}",
    "LD50 {measured, converted}",
    "UNIT {LD50}.1",
    "Organism",
    "Administration route",
]
new_data = data[new_columns]
# %%
new_data["ID"] = ["HW-" + str(i) for i in range(1, len(new_data) + 1)]

new_data.to_csv("./ld50_mammals_molecules.csv", index_label="ID")

# %%
plt.figure(figsize=(10, 6))
sns.countplot(x="Organism", data=data, order=data["Organism"].value_counts().index)

plt.title("Count of Molecules by Organism")
plt.xlabel("Organism")
plt.ylabel("Count")
plt.xticks(rotation=90)  # Rotate labels for readability
plt.tight_layout()
plt.show()
# %%
mammals = [
    "mouse",
    "rat",
    "guinea pig",
    "rabbit",
    "dog",
    "mammal (species unspecified)",
    "sheep",
    "hamster",
    "cat",
    "monkey",
    "pig",
]

data_mammals = data[data["Organism"].isin(mammals)]
# %%
plt.figure(figsize=(10, 6))
sns.histplot(data=data_mammals, x="LD50 {measured, converted}", bins=50, kde=True)

plt.title("Distribution of LD50 Values")
plt.xlabel("LD50 (measured, converted)")
plt.ylabel("Frequency")
plt.show()

# %%
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

plt.rcParams.update({"font.size": 14})

plt.figure(figsize=(14, 8))
ax = sns.violinplot(
    data=data_mammals,
    x="Organism",
    y="LD50 {measured, converted}",
    hue="Organism",  # explicitly set hue
    legend=False,  # suppress seaborn legend
    dodge=False,
    palette="tab10",
    inner="box",
)

# Manually create legend handles
unique_orgs = data_mammals["Organism"].unique()
palette = sns.color_palette("tab10", len(unique_orgs))
handles = [
    mpatches.Patch(color=palette[i], label=org) for i, org in enumerate(unique_orgs)
]

plt.legend(
    handles=handles, title="Organism", bbox_to_anchor=(1.05, 1), loc="upper left"
)
plt.xticks(rotation=45, ha="right")
plt.title("Distribution of LD50 Values per Mammalian Organism", fontsize=16)
plt.xlabel("Organism")
plt.ylabel("LD50 (measured, converted)")
plt.tight_layout()
plt.show()


# %%
