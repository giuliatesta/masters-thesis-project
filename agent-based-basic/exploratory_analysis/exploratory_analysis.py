from matplotlib import pyplot as plt

from preprocessing import load_dataset_csv
import seaborn as sns


def distribution(attribute):
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(x=attribute, data=df, order=df[attribute].value_counts().index, hue=attribute, legend="brief")
    ax.set_xticklabels([])
    for container in ax.containers:
        ax.bar_label(container)
    plt.legend(loc='upper right')
    plt.title(f'{attribute.replace("_", " ")} Distribution')
    plt.savefig(f"./exploratory_analysis/{attribute.lower()}_distribution.png")


df = load_dataset_csv("/Users/giuliatesta/PycharmProjects/masters-thesis-project/dataset/df_DNA_sharingEU.csv",
                      index=False)
df.rename(columns={"Age_c": "Age", "Location_of_resudence": "Location_of_residence"}, inplace=True)

# description = data.describe(include="all")
# description.to_csv("description.csv")

distribution("Education")
distribution("Gender")
distribution("Profession")
distribution("Age")
distribution("Income_level")
distribution("Location_of_residence")
