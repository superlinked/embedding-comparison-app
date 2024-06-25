import altair as alt
import numpy as np
import pandas as pd

from omegaconf import DictConfig
from sklearn.decomposition import PCA


def pca_transform(vectors: np.ndarray) -> np.ndarray:
    return PCA().fit_transform(vectors)


def create_pca_component_scatter(
    df: pd.DataFrame, pca_vectors: np.ndarray, color_by: str, config: DictConfig
) -> alt.Chart:
    chart_data = pd.DataFrame(pca_vectors, columns=["Component 1", "Component 2"])
    chart_data[config.data.target_colname] = df[config.data.target_colname]
    chart_data[color_by] = df[color_by]
    return (
        alt.Chart(chart_data)
        .mark_circle()
        .encode(
            x="Component 1",
            y="Component 2",
            color=color_by,
            tooltip=["Component 1", "Component 2", config.data.target_colname, color_by],
        )
        .properties(width=config.chart.width, height=config.chart.height)
    )
