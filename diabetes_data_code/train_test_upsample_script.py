import os

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from diabetes_data_code.train_test_upsample import (
    create_cross_validation,
    upsample_cross_validation_data,
    upsample_training_data,
)

if __name__ == "__main__":
    # Read in the data, parmaeters
    df_processed_file = os.path.join("data", "processed", "df_processed.pkl")
    df_processed = pd.read_pickle(df_processed_file)
    data_params_file = "data_params.yaml"
    with open(data_params_file, "r") as f:
        data_params = yaml.safe_load(f)

    # train test split
    df_train, df_test = train_test_split(
        df_processed,
        test_size=data_params["xgboost_training"]["training_percentage"],
        random_state=42,
    )

    features = [col for col in df_train.columns if col != "Diabetes"]
    target = "Diabetes"
    n_folds = data_params["xgboost_training"]["num_cross_validation_sets"]

    X_train = df_train[features]
    y_train = df_train[target]

    X_test = df_test[features]
    y_test = df_test[features]

    # Create cross validation data for hyperparameter tuning
    (
        X_train_cv_list,
        y_train_cv_list,
        X_val_cv_list,
        y_val_cv_list,
    ) = create_cross_validation(X_train, y_train, n_folds)

    # Upsample the training (not validation) data in the cross validation set for hyperparameter tuning
    X_train_cv_list_upsample, y_train_cv_list_upsample = upsample_cross_validation_data(
        X_train_cv_list, y_train_cv_list
    )

    # Upsample the training data to train on after we have configured the hyperparameters
    X_train_upsample, y_train_upsample = upsample_training_data(X_train, y_train)

    # Output the created dataframes
    training_data_dir = os.path.join("data", "training")
    os.makedirs(training_data_dir, exist_ok=True)

    # Output training and upsample training
    X_train.to_pickle(os.path.join(training_data_dir, "X_train.pkl"))
    X_train_upsample.to_pickle(os.path.join(training_data_dir, "X_train_upsample.pkl"))

    # Output testing data
    X_test.to_pickle(os.path.join(training_data_dir, "X_test.pkl"))
    y_test.to_pickle(os.path.join(training_data_dir, "y_test.pkl"))

    # Output cross validation and upsamples cross validation
    cross_val_data_dir = os.path.join(training_data_dir, "cv")
    os.makedirs(cross_val_data_dir, exist_ok=True)

    for idx, (X_train_cv, y_train_cv, X_val_cv, y_val_cv) in enumerate(
        zip(X_train_cv_list, y_train_cv_list, X_val_cv_list, y_val_cv_list)
    ):
        X_train_cv.to_pickle(os.path.join(cross_val_data_dir, f"X_train_{idx}.pkl"))
        y_train_cv.to_pickle(os.path.join(cross_val_data_dir, f"y_train_{idx}.pkl"))
        X_val_cv.to_pickle(os.path.join(cross_val_data_dir, f"X_val_cv{idx}.pkl"))
        y_val_cv.to_pickle(os.path.join(cross_val_data_dir, f"y_val_cv{idx}.pkl"))

    for idx, (X_train_cv_upsample, y_train_cv_upsample) in enumerate(
        zip(X_train_cv_list_upsample, y_train_cv_list_upsample)
    ):
        X_train_cv_upsample.to_pickle(
            os.path.join(cross_val_data_dir, f"X_train_{idx}_upsample.pkl")
        )
        y_train_cv_upsample.to_pickle(
            os.path.join(cross_val_data_dir, f"y_train_{idx}_upsample.pkl")
        )
