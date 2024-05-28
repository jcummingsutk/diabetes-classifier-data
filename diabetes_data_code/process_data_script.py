import os

import pandas as pd

from diabetes_data_code.process_data import (
    all_cols_to_int,
    remove_unimportant_cols,
    rename_diabetes_column,
)

if __name__ == "__main__":
    raw_data_file = os.path.join("data", "raw", "diabetes.csv")
    df_raw = pd.read_csv(raw_data_file)
    df_processed = all_cols_to_int(df_raw)
    df_processed = rename_diabetes_column(df_processed)
    df_processed = remove_unimportant_cols(df_processed)
    output_folder = os.path.join("data", "processed")

    os.makedirs(output_folder, exist_ok=True)
    df_processed.to_pickle(os.path.join(output_folder, "df_processed.pkl"))
