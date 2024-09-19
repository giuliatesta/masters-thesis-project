from matplotlib import pyplot as plt

from preprocessing import load_dataset_csv
import seaborn as sns

data = load_dataset_csv("../dataset/df_DNA_sharingEU.csv", index=False)
print(data.shape)
print(data.columns)

description = data.describe(include="all")
description.to_csv("description.csv")

