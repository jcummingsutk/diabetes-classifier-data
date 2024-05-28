import os

import pandas as pd

from diabetes_data_code.process_data import all_cols_to_int, rename_diabetes_column

if __name__ == "__main__":
    raw_data_file = os.path.join("data", "raw", "diabetes.csv")
    df_raw = pd.read_csv(raw_data_file)
    df_processed = all_cols_to_int(df_raw)
    df_processed = rename_diabetes_column(df_processed)
