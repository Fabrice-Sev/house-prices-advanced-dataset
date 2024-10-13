# from conf import config
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
import mlflow
from mlflow.models import infer_signature
import uuid
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor


def get_processed_data():
    with open('.\\files\\data.pickle', 'rb') as handle:
        X_train, X_test, Y_train, Y_test = pickle.load(handle)
    return X_train, X_test, Y_train, Y_test


def log_metrics(model, data_set, Y_expected) -> None :
    Y_pred = model.predict(data_set)
    
    # Root Mean Squared Error
    rmse = root_mean_squared_error (Y_expected, Y_pred)
    
    # Mean Squared Error
    mse = rmse*rmse

    # Mean Absolute Error
    mae = mean_absolute_error(Y_expected, Y_pred)

    # R-squared
    r2 = r2_score(Y_expected, Y_pred)
    
    # Log the error metrics that were calculated during validation
    metrics = {"mae":mae, "rmse":rmse, "mse":mse, "r2":r2}
    mlflow.log_metrics(metrics)


# def train(X_train, X_valid, y_train, y_valid):
#    return None

def train(model_pipeline, X_train, Y_train, model_params) -> None:
    model_pipeline.fit(X_train, Y_train)
    
    # Sets the current active experiment to the "house... bla bla" experiment and
    # returns the Experiment metadata
    mlflow.set_tracking_uri("http://127.0.0.1:8080")
    mlflow_experiment_name = "House Pricing Exp"
    experiment = mlflow.set_experiment(mlflow_experiment_name)
    # artifact_path = "xgb_model_for_house_pricing"
    
    
    # Define a run name for this iteration of training.
    # If this is not set, a unique name will be auto-generated for your run.
    # Note that the model here is a pipeline. If it is directly the model, then the following line will break 
    name_of_algorithm_used_in_pipeline = list(model_pipeline.named_steps) [len(list(model_pipeline.named_steps))-1]
    run_name = "adv_house_pred_run_"+name_of_algorithm_used_in_pipeline+str(uuid.uuid4())[:8]

    # Define an artifact path that the model will be saved to.
    artifact_path = name_of_algorithm_used_in_pipeline + "_model_for_house_pricing"
    
    # Initiate the MLflow run context
    with mlflow.start_run(run_name=run_name):
        # Log the parameters used for the model fit
        mlflow.log_params(model_params)

        # Log the error metrics that were calculated during validation
        # mlflow.log_metrics(metrics)
        
        # Set a tag that we can use to remind ourselves what this run was for
        # mlflow.set_tag("Training Info", "Recommendation system Model")

        # Infer the model signature
        signature = infer_signature(X_train.sample(3, random_state=111), model_pipeline.predict(X_train.sample(3, random_state=111)))

        # Log the model
        model_info = mlflow.sklearn.log_model(
            sk_model=model_pipeline,
            artifact_path=artifact_path,
            signature=signature,
            input_example=X_train.sample(3, random_state=111),
            # registered_model_name=collaboratif_filtering_type+"_recommendation_model",
        )
        log_metrics(model=model_pipeline, data_set=X_train, Y_expected=Y_train)
        mlflow.end_run()

def prepare_model_pipeline(df: pd.DataFrame, model, model_params: dict):
    df_cat = df.select_dtypes(include = ['O'])
    df_num = df.select_dtypes(include = ['float64', 'int64'])
    categorical_features = list(df_cat.columns)
    numeric_features = list(df_num.columns)
    # Define preprocessing for numeric features (with missing values handled)
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())                 # Scale numeric features
    ])

    # Define preprocessing for categorical features (one-hot encoding, without handling missing values here)
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One-hot encode categorical features
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    pipeline = Pipeline([
    ('preprocessor', preprocessor), 
    ("model", model(**model_params))
    ])
    
    return pipeline

def run_train():
    X_train, X_valid, y_train, y_valid = get_processed_data()
    # xgb_params = {"booster":"gbtree"}
    # xgb_pipeline = prepare_model_pipeline(X_train, model=xgb.XGBRegressor, model_params=xgb_params)
    
    random_forest_params = {
    'n_estimators': 100,        # Number of trees in the forest
    'max_depth': 10,            # Maximum depth of the tree
    'min_samples_split': 2,     # Minimum number of samples required to split an internal node
    'min_samples_leaf': 1,      # Minimum number of samples required to be at a leaf node
    'random_state': 42          # Seed for reproducibility
    }
    random_forest_pipeline = prepare_model_pipeline(X_train, RandomForestRegressor, random_forest_params)
    
    train(random_forest_pipeline, X_train, y_train, random_forest_params)
    print("Training and logging completed successfully!!")
