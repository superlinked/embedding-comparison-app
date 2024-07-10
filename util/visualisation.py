import altair as alt
import numpy as np
import pandas as pd
import plotly_express as px

from omegaconf import DictConfig
from plotly.graph_objs import Figure


def create_altair_2d_scatter(
    df: pd.DataFrame,
    component_vectors: np.ndarray,
    color_by_colname: str,
    config: DictConfig,
) -> alt.Chart:
    chart_data = pd.DataFrame(component_vectors[:, :2], columns=["Component 1", "Component 2"])
    chart_data[config.data.target_colname] = df[config.data.target_colname]
    if color_by_colname not in chart_data.columns:
        chart_data[color_by_colname] = df[color_by_colname]
    return (
        alt.Chart(chart_data)
        .mark_circle()
        .encode(
            x="Component 1",
            y="Component 2",
            color=color_by_colname,
            tooltip=[
                "Component 1",
                "Component 2",
                config.data.target_colname,
                color_by_colname,
            ],
        )
        .properties(width=config.chart.width, height=config.chart.height)
    ).interactive()


def create_plotly_3d_scatter(
    df: pd.DataFrame,
    component_vectors: np.ndarray,
    color_by_colname: str,
    config: DictConfig,
) -> Figure:
    chart_data = pd.DataFrame(
        component_vectors, columns=["Component 1", "Component 2", "Component 3"]
    )
    chart_data[config.data.target_colname] = df[config.data.target_colname]
    chart_data[color_by_colname] = df[color_by_colname]
    return px.scatter_3d(
        data_frame=chart_data,
        x="Component 1",
        y="Component 2",
        z="Component 3",
        color=color_by_colname,
        width=config.chart.width,
        height=config.chart.height,
    )
