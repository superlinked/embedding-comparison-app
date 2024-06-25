import numpy as np
import pandas as pd

from omegaconf import DictConfig

from util.sbert import embed_with_model


def create_naive_embeddings(df: pd.DataFrame, config: DictConfig) -> np.ndarray:
    naive_input = get_naive_text_repr(df, config.data.target_colname)
    return embed_with_model(
        text_to_embed=naive_input, embedding_model=config.embedding_model.model_name
    )


def get_naive_text_repr(df: pd.DataFrame, target_colname: str) -> list[str]:
    return df.copy().apply(
        lambda x: get_naive_text_repr_for_row(
            row_series=x, target_colname=target_colname
        ),
        axis=1,
    )


def get_naive_text_repr_for_row(
    row_series: pd.Series,
    target_colname: str,
    colname_prefix_mapping: dict[str, str] | None = None,
) -> str:
    if colname_prefix_mapping is None:
        colname_prefix_mapping = {
            col: col for col in row_series.index if not col == target_colname
        }
    row_series = row_series.loc[colname_prefix_mapping.keys()]
    return ", ".join(
        [
            f"{colname_prefix_mapping[col]}: {val}"
            for col, val in zip(row_series.index, row_series.values)
        ]
    )
