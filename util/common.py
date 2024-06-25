import pandas as pd
import re


def camel_to_snake(name):
    name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()


def separate_numeric_and_string_columns(df: pd.DataFrame):
    numeric_columns = [f for f in df.columns if pd.api.types.is_numeric_dtype(df[f])]
    string_columns = [f for f in df.columns if pd.api.types.is_string_dtype(df[f])]
    return numeric_columns, string_columns
