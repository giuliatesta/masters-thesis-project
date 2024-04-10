import numpy as np
import pandas as pd
from pandas import DataFrame


# reads the csv file at the pointed location
# index_col = 0 means that it uses the first column as index for the rows
# header = 0 means that the first row contains the columns' names
# encoding= cp1252 since some values cannot be read by default utf8
def load_dataset_csv(path: str):
    print(f"Loading the dataset from {path}")
    dataset = pd.read_csv(path, sep=',', index_col=0, header=0, encoding="cp1252")
    dataset.replace("", np.nan, inplace=True)  # Replace empty strings with NaN
    dataset.dropna(inplace=True)
    print(f"Loaded {len(dataset)} records")
    return dataset


# filters the dataframe df by the columnName column
def filter_by(df: DataFrame, column_name: str, column_value: str):
    print(f"Filtering by {column_name}={column_value}")
    return df.loc[df[column_name] == column_value]
