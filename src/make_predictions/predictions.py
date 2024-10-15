import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import mlflow
from mlflow import MlflowClient
# from conf import config

TRACKING_URI = "http://127.0.0.1:8080/"
TARGET_COLUMN = "SalePrice"


def get_test_data(path_to_dataset_folder, dataset_file) :
    df = pd.read_csv(path_to_dataset_folder+"\\"+dataset_file)
    return df


def make_data_processing(df_test: pd.DataFrame) :
    df_cat = df_test.select_dtypes(include = ['O'])
    df_num = df_test.select_dtypes(include = ['float64', 'int64'])
    categorical_features = list(df_cat.columns)
    numeric_features = list(df_num.columns)
    
    new_df_test = df_test.copy()
    # This will be to handle na_values for LotFrontage
    new_df_test["LotFrontage"] = df_test.groupby("Neighborhood")["LotFrontage"].transform(
        lambda x: x.fillna(x.median()))

    # This will be to handle na_values for MasVnrArea
    new_df_test["MasVnrArea"] = new_df_test["MasVnrArea"].fillna(0)

    # This will be to handle na_values for Basement
    new_df_test["BsmtExposure"] = df_test["BsmtExposure"].fillna(df_test["BsmtExposure"].mode()[0])
    new_df_test["BsmtFinType2"] = df_test["BsmtFinType2"].fillna(df_test["BsmtFinType2"].mode()[0])

    # # This will be to handle na_values for Electrical
    new_df_test["Electrical"] = df_test["Electrical"].fillna(df_test["Electrical"].mode()[0])
    # Or to drop the observation
    # new_df_train.drop(df_test.loc[df_test['Electrical'].isnull()].index, inplace=True)
    
    new_df_test["PoolQC"] = new_df_test["PoolQC"].fillna("None")

    new_df_test["MiscFeature"] = new_df_test["MiscFeature"].fillna("None")

    new_df_test["Alley"] = new_df_test["Alley"].fillna("None")

    new_df_test["Fence"] = new_df_test["Fence"].fillna("None")

    new_df_test["FireplaceQu"] = new_df_test["FireplaceQu"].fillna("None")

    new_df_test["MasVnrType"] = new_df_test["MasVnrType"].fillna("None")
    
    for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
        new_df_test[col] = new_df_test[col].fillna(0)
        
    for col in ('BsmtQual', 'BsmtCond', 'BsmtFinType1'):
        new_df_test[col] = new_df_test[col].fillna('None')
        
    for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
        new_df_test[col] = new_df_test[col].fillna('None')
    
    new_df_test.isnull().sum().max() # just checking that there's no missing data missing...
    
    # data transformation
    new_df_test['GrLivArea'] = np.log(new_df_test['GrLivArea'])
    
    # create column for new variable (one is enough because it's a binary categorical feature)
    # if area>0 it gets 1, for area==0 it gets 0
    new_df_test['HasBsmt'] = 0 
    new_df_test.loc[new_df_test['TotalBsmtSF']>0,'HasBsmt'] = 1
    
    # Convert TotalBsmtSF to float if it is not already
    new_df_test['TotalBsmtSF'] = new_df_test['TotalBsmtSF'].astype(float)
    
    # transform data
    # new_df_test.loc[new_df_test['HasBsmt']==1,'TotalBsmtSF'] = np.log(new_df_test['TotalBsmtSF'])
    new_df_test.loc[new_df_test['HasBsmt'] == 1, 'TotalBsmtSF'] = np.log(new_df_test.loc[new_df_test['HasBsmt'] == 1, 'TotalBsmtSF'])
    
    if "Id" in new_df_test.columns:
        new_df_test.drop(columns="Id", inplace= True)
        
    if TARGET_COLUMN in new_df_test.columns:
        new_df_test.drop(columns=TARGET_COLUMN, inplace= True)
    
    # To get all column with empty elts
    na_columns = []
    for column in new_df_test:
        num_elts_in_col = new_df_test[column].isna().sum()
        if num_elts_in_col > 0 :
            na_columns.append(column)
      
    # To replace na values with values based on the type of the column  
    for column in na_columns:
        if column in numeric_features:
            new_df_test[column] = new_df_test[column].fillna(0)
        if column in categorical_features:
            new_df_test[column] = new_df_test[column].fillna("None")
            
    return new_df_test


def make_predictions(df_test, model_name:str="xgb_model") :
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()
    
    versions = client.search_model_versions(f"name='{model_name}'")
    latest_version = versions[0]
    model = mlflow.pyfunc.load_model(f"runs:/{latest_version.run_id}/model")
    Y_predict = pd.DataFrame(model.predict(df_test), columns=[TARGET_COLUMN])
    
    # Code to get not the last model but the one with the stage tag.
    # for version in client.search_model_versions(
    # f"name='{model_name}'"
    # ):
        # version_dict = dict(version)
    #     if "stage" in version_dict['tags']:
    #         model = mlflow.pyfunc.load_model(f"runs:/{version.run_id}/model")
    #         return pd.DataFrame(model.predict(df_test), columns=[TARGET_COLUMN])
    # return pd.DataFrame(model.predict(df_test), columns=[TARGET_COLUMN])
    return Y_predict


def save_predictions(predictions: pd.DataFrame) :
    path_output_predictions = '.\\files\\test_predictions.csv'
    predictions.to_csv(path_output_predictions, index=False)
    return None


def run_make_predictions():
    path_to_dataset_folder = "E:\Work\ML\Databases\csv_files\house-prices-advanced-dataset"
    test_file = "test.csv"
    
    X_test = get_test_data(path_to_dataset_folder, test_file)
    X_test = make_data_processing(X_test)
    y_predictions = make_predictions(X_test, model_name='Random_Forest_model')
    save_predictions(y_predictions)
    print("Predictions done and saved successfully!!")