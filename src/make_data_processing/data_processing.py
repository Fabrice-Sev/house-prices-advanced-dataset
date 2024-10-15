# from conf import config
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle

def load_data(path_to_dataset_folder, dataset_file):
    df = pd.read_csv(path_to_dataset_folder+"\\"+dataset_file)
    return df


def split_data(df: pd.DataFrame) :
    X_train, X_test, Y_train, Y_test = train_test_split(df.iloc[:,:-1], df.iloc[:,-1:], test_size=0.2)
    return X_train, X_test, Y_train, Y_test


def data_processing(df: pd.DataFrame) :
    # Here we will handle missing variables and transformations on the dataset
    # This will be to handle na_values for LotFrontage
    df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(
        lambda x: x.fillna(x.median()))

    # This will be to handle na_values for MasVnrArea
    df["MasVnrArea"] = df["MasVnrArea"].fillna(0)

    # This will be to handle na_values for Basement
    df["BsmtExposure"] = df["BsmtExposure"].fillna(df["BsmtExposure"].mode()[0])
    df["BsmtFinType2"] = df["BsmtFinType2"].fillna(df["BsmtFinType2"].mode()[0])

    # # This will be to handle na_values for Electrical
    df["Electrical"] = df["Electrical"].fillna(df["Electrical"].mode()[0])
    # Or to drop the observation
    # df.drop(df_train.loc[df['Electrical'].isnull()].index, inplace=True)

    df["PoolQC"] = df["PoolQC"].fillna("None")

    df["MiscFeature"] = df["MiscFeature"].fillna("None")

    df["Alley"] = df["Alley"].fillna("None")

    df["Fence"] = df["Fence"].fillna("None")

    df["FireplaceQu"] = df["FireplaceQu"].fillna("None")

    df["MasVnrType"] = df["MasVnrType"].fillna("None")

    for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
        df[col] = df[col].fillna(0)

    for col in ('BsmtQual', 'BsmtCond', 'BsmtFinType1'):
        df[col] = df[col].fillna('None')
        
    for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
        df[col] = df[col].fillna('None')
    
    if 'SalePrice' in df.columns:
        df['SalePrice'] = np.log(df['SalePrice'])
    
    df['GrLivArea'] = np.log(df['GrLivArea'])
    
    # df['HasBsmt'] = pd.Series(len(df['TotalBsmtSF']), index=df.index)
    df['HasBsmt'] = 0 
    df.loc[df['TotalBsmtSF'] > 0,'HasBsmt'] = 1
    
    # Convert TotalBsmtSF to float if it is not already
    df['TotalBsmtSF'] = df['TotalBsmtSF'].astype(float)
    
    # To handle column types that disturb during inference
    df['BsmtFinSF1'] = df['BsmtFinSF1'].astype(float)
    df['BsmtFinSF2'] = df['BsmtFinSF2'].astype(float)
    df['BsmtUnfSF'] = df['BsmtUnfSF'].astype(float)
    df['BsmtFullBath'] = df['BsmtFullBath'].astype(float)
    df['BsmtHalfBath'] = df['BsmtHalfBath'].astype(float)
    df['GarageCars'] = df['GarageCars'].astype(float)
    df['GarageArea'] = df['GarageArea'].astype(float)

    # df.loc[df['HasBsmt']==1,'TotalBsmtSF'] = np.log(df['TotalBsmtSF'])
    df.loc[df['HasBsmt'] == 1, 'TotalBsmtSF'] = np.log(df.loc[df['HasBsmt'] == 1, 'TotalBsmtSF'])
    
    if "Id" in df.columns:
        df.drop(columns="Id", inplace= True)
    return df


def save_data_processing(X_train, X_test, Y_train, Y_test) :
    path_processed_train_data = '.\\files\\data.pickle'
    with open(path_processed_train_data, 'wb') as handle:
        pickle.dump((X_train, X_test, Y_train, Y_test), handle, protocol=pickle.HIGHEST_PROTOCOL)
    return None


def run_data_processing():
    # Correct data_processing call
    path_to_dataset_folder = "E:\Work\ML\Databases\csv_files\house-prices-advanced-dataset"
    train_file = "train.csv"
    
    data = load_data(path_to_dataset_folder, train_file)
    X_train, X_valid, y_train, y_valid = split_data(data)
    X_train, X_valid = data_processing(X_train), data_processing(X_valid)
    save_data_processing(X_train, X_valid, y_train, y_valid)
    print("Train data saved successfully!")
