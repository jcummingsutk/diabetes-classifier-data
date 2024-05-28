import pandas as pd


def all_cols_to_int(df_raw: pd.DataFrame) -> pd.DataFrame:
    for col in df_raw.columns:
        df_raw[col] = df_raw[col].astype(int)
    return df_raw


def rename_diabetes_column(df_raw: pd.DataFrame) -> pd.DataFrame:
    df_raw = df_raw.rename(columns={"Diabetes_binary": "Diabetes"})
    return df_raw


def remove_unimportant_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=["AnyHealthcare", "HvyAlcoholConsump"])
    return df
