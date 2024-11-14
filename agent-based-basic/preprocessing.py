import numpy as np
import pandas as pd
from pandas import DataFrame


# reads the csv file at the pointed location
# index_col = 0 means that it uses the first column as index for the rows
# header = 0 means that the first row contains the columns' names
# encoding = cp1252 since some values cannot be read by default utf8
def load_dataset_csv(path: str, index: bool):
    print(f"Loading the dataset from {path}")
    dataset = pd.read_csv(path, sep=',', header=0, encoding="cp1252")
    dataset.replace("", np.nan, inplace=True)  # Replace empty strings with NaN
    if not index:
        dataset.reset_index(drop=True, inplace=True)
    # removes rows wit null values
    dataset.dropna(inplace=True)
    # removes duplicated rows
    dataset.drop_duplicates(inplace=True)
    # typos fix in column names
    dataset.rename(columns={
        "Age_c": "Age",
        "Location_of_resudence": "Location_of_residence",
        "Would_subsribe_car_sharing_if_available": "Would_subscribe_car_sharing_if_available"},
        inplace=True)

    # casts the values of numerical columns into numbers
    # dataset = dataset.astype({
    #     'Education_DNA': "float64",
    #     'Income_level_DNA': "float64",
    #     'Profession_DNA': "float64",
    #     'Age_DNA': "float64",
    #     'Considering_electric_or_hybrid_vehicle_next_purchase_DNA': "float64",
    #     'Concern_environmental_impacts_DNA': "float64",
    #     'Country_DNA': "float64",
    #     'sha_ind': "float64",
    #     'sha_ind_norm': "float64"
    # })
    # # and the string values into strings
    # dataset = dataset.convert_dtypes()

    print(f"Loaded {len(dataset)} records")
    return dataset


# filters the dataframe df by the column_name column and also can return only a specific number of records
def filter_by(df: DataFrame, column_name: str, column_value: str, number_of_rows=-1):
    print(f"Filtering by {column_name}={column_value}")
    filtered = df.loc[df[column_name] == column_value]

    # if I want to get the first n values
    if number_of_rows != -1:
        filtered = filtered.head(number_of_rows)
    print(f"Filtered {len(filtered)} data out of {len(df)} rows")
    filtered.reset_index(drop=True, inplace=True)
    return filtered
