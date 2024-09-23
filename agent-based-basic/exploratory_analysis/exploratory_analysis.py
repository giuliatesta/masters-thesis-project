from matplotlib import pyplot as plt

from preprocessing import load_dataset_csv
import seaborn as sns
import pandas as pd


def countplot(attribute):
    ax = sns.countplot(x=attribute, data=df, order=df[attribute].value_counts().index, hue=attribute, legend="brief")
    ax.set_xticklabels([])
    for container in ax.containers:
        ax.bar_label(container)
    plt.legend(loc='upper right', ncol=2)
    plt.title(f'{attribute.replace("_", " ")} Distribution')

def kdeplot(attribute):
    sns.kdeplot(df[attribute], fill=True, legend=True)
    plt.grid()
    plt.title(f'KDE Plot of {attribute.replace("_", " ")}')
    # plt.savefig(f"./kdeplot/{attribute.lower()}_distribution.png")

def crosstab(attribute_1, attribute_2):
    pd.crosstab(df[attribute_1], df[attribute_2]).plot(kind='bar', stacked=True,
                                                                                     colormap='viridis')
    plt.title(f'{attribute_1.replace("_", " ")} and {attribute_2.replace("_", " ")} crosstab')
    plt.xlabel(attribute_1.replace("_", " "))
    plt.ylabel(attribute_2.replace("_", " "))
    plt.legend()
    plt.subplots_adjust(bottom=0.75)
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3),)
    plt.tight_layout()
    plt.xticks(rotation=0, fontsize=8)
    plt.savefig(f"./crosstab/{attribute_1.lower()}_and_{attribute_2.lower()}_crosstab.png")


df = load_dataset_csv("../../dataset/df_DNA_sharingEU.csv",
                      index=False)
df.rename(columns={"Age_c": "Age", "Location_of_resudence": "Location_of_residence", "Would_subsribe_car_sharing_if_available": "Would_subscribe_car_sharing_if_available"}, inplace=True)
print(df.columns)

df["Education"] = df["Education"].replace("Tertiary and higher (University degree, PhD or similar degrees).", "Tertiary and higher")
df["Education"] = df["Education"].replace("Lower secondary (upper elementary school or similar);", "Lower secondary")
df["Education"] = df["Education"].replace("Upper secondary (high school or similar);", "Upper secondary")
df["Education"] = df["Education"].replace("Primary (elementary school or similar);", "Primary")

df["Location_of_residence"] = df["Location_of_residence"].replace("Metropolitan area of a big city with more than 1.000.000  inhabitants", "Metropolitan area")
df["Location_of_residence"] = df["Location_of_residence"].replace("Small or medium town (less than 250.000 inhabitants)", "Small or medium town")
df["Location_of_residence"] = df["Location_of_residence"].replace("Large city (from 250.000 to 1.000.000 inhabitants)", "Large cit")

df["Profession"] = df["Profession"].replace("manual worker/agricultural worker/farmer", "manual worker")
df["Profession"] = df["Profession"].replace("business owner/entrepreneur", "business owner")
df["Profession"] = df["Profession"].replace("registered freelance professional", "registered freelance")
df["Profession"] = df["Profession"].replace("storekeeper/tradesman/craftsman", "storekeeper")
# DESCRIPTION for CATEGORICAL and NUMERICAL values
# description = df.describe()
# description.to_csv("description_numerical.csv")

# COUNT PLOTS
# plt.figure(figsize=(16, 12))
# plt.subplot(3, 2, 1)
# countplot("Education")
# plt.subplot(3, 2, 2)
# countplot("Income_level")
# plt.subplot(3, 2, 3)
# countplot("Profession")
# plt.subplot(3, 2, 4)
# countplot("Age")
# plt.subplot(3, 2, 5)
# countplot("Gender")
# plt.subplot(3, 2, 6)
# countplot("Location_of_residence")
# plt.tight_layout()
# plt.savefig(f"./exploratory_analysis/countplot/countplots_DNA.png")

# countplot("Education")
# countplot("Gender")
# countplot("Profession")
# countplot("Age")
# countplot("Income_level")
# countplot("Location_of_residence")

# KDE PLOTS
# plt.figure(figsize=(16, 12))
# plt.subplot(5, 2, 1)
# kdeplot("Education_DNA")
# plt.subplot(5, 2, 2)
# kdeplot("Income_level_DNA")
# plt.subplot(5, 2, 3)
# kdeplot("Profession_DNA")
# plt.subplot(5, 2, 4)
# kdeplot("Age_DNA")
# plt.subplot(5, 2, 5)
# kdeplot("Considering_electric_or_hybrid_vehicle_next_purchase_DNA")
# plt.subplot(5, 2, 6)
# kdeplot("Concern_environmental_impacts_DNA")
# plt.subplot(5, 2, 7)
# kdeplot("Country_DNA")
# plt.subplot(5, 2, 8)
# kdeplot("sha_ind_norm")
# plt.tight_layout()
# plt.savefig(f"./exploratory_analysis/kdeplot/kdeplots_DNA.png")

# BOX PLOTS
# plt.figure(figsize=(12, 8))
# numerical_columns = ["Education_DNA", "Income_level_DNA", "Profession_DNA", "Age_DNA", "Considering_electric_or_hybrid_vehicle_next_purchase_DNA", "Concern_environmental_impacts_DNA", "Country_DNA"]
# sns.boxplot(data=df[numerical_columns], legend=True)
# plt.subplots_adjust(bottom=0.25)
# plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.35))
# plt.xticks([])
# plt.grid()
# plt.title(f'Box Plot of sharing DNA')
# plt.savefig(f"./boxplot/sharing_dna_distribution.png")


# Stacked bar plot of Education vs Willingness to subscribe to car sharing
#crosstab("Education", "Would_subscribe_car_sharing_if_available")
#crosstab("Income_level", "Considering_electric_or_hybrid_vehicle_next_purchase")
#crosstab("Gender", "Concern_environmental_impacts")

