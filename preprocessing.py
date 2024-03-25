import pandas as pd
from pandas import DataFrame


# reads the csv file at the pointed location
# index_col = 0 means that it uses the first column as index for the rows
# header = 0 means that the first row contains the columns' names
def load_dataset_csv(path: str):
    return pd.read_csv(path, sep=',', index_col=0, header=0, encoding="cp1252")


# filters the dataframe df by the columnName column
def filter_by(df: DataFrame, column_name: str, column_value: str):
    return df.loc[df[column_name] == column_value]
