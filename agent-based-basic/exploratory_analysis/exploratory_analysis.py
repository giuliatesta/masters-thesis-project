from matplotlib import pyplot as plt

from preprocessing import load_dataset_csv
import seaborn as sns


def countplot(attribute):
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(x=attribute, data=df, order=df[attribute].value_counts().index, hue=attribute, legend="brief")
    ax.set_xticklabels([])
    for container in ax.containers:
        ax.bar_label(container)
    plt.legend(loc='upper right')
    plt.title(f'{attribute.replace("_", " ")} Distribution')
    plt.savefig(f"./countplot/{attribute.lower()}_distribution.png")

def kdeplot(attribute):
    plt.figure(figsize=(12, 6))
    sns.kdeplot(df[attribute], fill=True, legend=True)
    plt.grid()
    plt.title(f'KDE Plot of {attribute.replace("_", " ")}')
    plt.savefig(f"./kdeplot/{attribute.lower()}_distribution.png")


df = load_dataset_csv("../../dataset/df_DNA_sharingEU.csv",
                      index=False)
df.rename(columns={"Age_c": "Age", "Location_of_resudence": "Location_of_residence"}, inplace=True)

description = df.describe()
description.to_csv("description_numerical.csv")
# distribution("Education")
# distribution("Gender")
# distribution("Profession")
# distribution("Age")
# distribution("Income_level")
# distribution("Location_of_residence")

kdeplot("Education_DNA")
kdeplot("Income_level_DNA")
kdeplot("Profession_DNA")
kdeplot("Age_DNA")
kdeplot("Considering_electric_or_hybrid_vehicle_next_purchase_DNA")
kdeplot("Concern_environmental_impacts_DNA")
kdeplot("Country_DNA")
kdeplot("sha_ind")
kdeplot("sha_ind_norm")