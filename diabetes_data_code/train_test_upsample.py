import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import KFold


def create_cross_validation(
    X_train: pd.DataFrame, y_train: pd.Series, n_folds: int
) -> tuple[list[pd.DataFrame], list[pd.Series], list[pd.DataFrame], list[pd.Series]]:
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    split = kf.split(X_train)

    X_train_cv_list: list[pd.DataFrame] = []
    y_train_cv_list: list[pd.Series] = []

    X_val_cv_list: list[pd.DataFrame] = []
    y_val_cv_list: list[pd.Series] = []

    for _, (train_index, test_index) in enumerate(split):
        X_train_cv, y_train_cv = (
            X_train.iloc[train_index],
            y_train.iloc[train_index],
        )
        X_val_cv, y_val_cv = X_train.iloc[test_index], y_train.iloc[test_index]
        X_train_cv_list.append(X_train_cv)
        y_train_cv_list.append(y_train_cv)

        X_val_cv_list.append(X_val_cv)
        y_val_cv_list.append(y_val_cv)
    return X_train_cv_list, y_train_cv_list, X_val_cv_list, y_val_cv_list


def upsample_cross_validation_data(
    X_train_cv_list: list[pd.DataFrame], y_train_cv_list: list[pd.DataFrame]
) -> tuple[list[pd.DataFrame], list[pd.Series]]:
    smt = SMOTE(random_state=42, k_neighbors=3)

    X_train_cv_list_upsample: list[pd.DataFrame] = []
    y_train_cv_list_upsample: list[pd.Series] = []
    for X_train_cv, y_train_cv in zip(X_train_cv_list, y_train_cv_list):
        X_train_cv_upsample, y_train_cv_upsample = smt.fit_resample(
            X_train_cv, y_train_cv
        )
        X_train_cv_list_upsample.append(X_train_cv_upsample)
        y_train_cv_list_upsample.append(y_train_cv_upsample)
    return X_train_cv_list_upsample, y_train_cv_list_upsample


def upsample_training_data(
    X_train: pd.DataFrame, y_train: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series]:
    smt = SMOTE(random_state=42, k_neighbors=3)

    X_train_upsample, y_train_upsample = smt.fit_resample(X_train, y_train)
    return X_train_upsample, y_train_upsample
