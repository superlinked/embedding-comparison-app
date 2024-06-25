import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from omegaconf import OmegaConf, DictConfig

from util.naive_embeddings import create_naive_embeddings
from util.pca import pca_transform, create_pca_component_scatter
from util.superlinked_embeddings import create_superlinked_embeddings


def main(config: DictConfig) -> None:
    st.set_page_config(layout="wide")

    # Title of the app
    st.title("Hey, wanna check some vector embeddings?")

    # File upload
    user_file = st.file_uploader(
        label="Upload the inputs to your vectors here.", type=["csv"]
    )

    user_input = st.text_input("Enter a URL to download data from")

    # Init state if not already initialized
    if "df" not in st.session_state:
        st.session_state["df"] = None
    if "df_to_use" not in st.session_state:
        st.session_state["df_to_use"] = None
    if "naive_embeddings" not in st.session_state:
        st.session_state["naive_embeddings"] = None
    if "superlinked_embeddings" not in st.session_state:
        st.session_state["superlinked_embeddings"] = None
    if "naive_pca" not in st.session_state:
        st.session_state["naive_pca"] = None
    if "superlinked_pca" not in st.session_state:
        st.session_state["superlinked_pca"] = None
    if "cols_to_use" not in st.session_state:
        st.session_state["cols_to_use"] = None
    if "coloring_attribute" not in st.session_state:
        st.session_state["coloring_attribute"] = None

    if user_file is not None:
        if user_input != "":
            st.write("File uploaded, ignoring URL.")
        st.session_state["df"] = pd.read_csv(user_file)
        st.write("Uploaded file read successfully.")
    elif user_input != "":
        st.session_state["df"] = pd.read_csv(user_input)
        st.write("File downloaded from URL and read successfully.")
    else:
        st.write(
            "Neither a file is uploaded, nor a URL is filled, please do one of those."
        )

    if st.session_state["df"] is not None:
        st.markdown("## The dataset", unsafe_allow_html=True)
        st.write(f'Shape of read dataset: `{st.session_state["df"].shape}`')
        st.write(st.session_state["df"].head())

        st.markdown("## Select columns to use")
        cols_to_use = st.multiselect(
            label="Select columns to use",
            options=[
                f
                for f in st.session_state["df"].columns
                if not f == config.data.target_colname
            ],
        )
        st.session_state["cols_to_use"] = cols_to_use
        if st.session_state["cols_to_use"] is not None:
            if st.button("Use selected columns!"):
                st.session_state["df_to_use"] = st.session_state["df"].drop(
                    [
                        f
                        for f in st.session_state["df"].columns
                        if f not in cols_to_use + [config.data.target_colname]
                    ],
                    axis=1,
                )

        if st.session_state["df_to_use"] is not None:
            st.markdown("# Create vectors", unsafe_allow_html=True)
            embedding_column_1, embedding_column_2 = st.columns(2)
            with embedding_column_1:
                st.markdown("## Create LLM embeddings")
                if st.session_state["naive_embeddings"] is None:
                    st.session_state["naive_embeddings"] = create_naive_embeddings(
                        st.session_state["df_to_use"], config=config
                    )
                if st.session_state["naive_embeddings"] is not None:
                    st.write(
                        f'LLM embeddings are created with shape: `{st.session_state["naive_embeddings"].shape}`'
                    )

            with embedding_column_2:
                st.markdown("## Create Superlinked embeddings")
                if st.session_state["superlinked_embeddings"] is None:
                    st.session_state[
                        "superlinked_embeddings"
                    ] = create_superlinked_embeddings(
                        st.session_state["df_to_use"], config=config
                    )
                if st.session_state["superlinked_embeddings"] is not None:
                    st.write(
                        f'Superlinked embeddings are created with shape: `{st.session_state["superlinked_embeddings"].shape}`'
                    )

            st.markdown("# Visualisation")
            if st.session_state["naive_pca"] is None:
                st.session_state["naive_pca"]: np.ndarray = pca_transform(
                    st.session_state["naive_embeddings"]
                )[:, :2]
            st.write("LLM embedding PCA transformation ready.")
            if st.session_state["superlinked_pca"] is None:
                st.session_state["superlinked_pca"]: np.ndarray = pca_transform(
                    st.session_state["superlinked_embeddings"]
                )[:, :2]
            st.write("Superlinked embedding PCA transformation ready.")

        if st.session_state["superlinked_pca"] is not None:
            st.session_state["coloring_attribute"] = st.selectbox(
                "Color by", st.session_state["df_to_use"].columns
            )
            if st.session_state["coloring_attribute"] is not None:
                if st.button("Create scatter plots!"):
                    column_1, column_2 = st.columns(2)
                    with column_1:
                        st.header("LLM embeddings")
                        naive_chart = create_pca_component_scatter(
                            st.session_state["df_to_use"],
                            st.session_state["naive_pca"],
                            st.session_state["coloring_attribute"],
                            config,
                        )
                        st.altair_chart(naive_chart)

                    with column_2:
                        st.header("Superlinked embeddings")
                        superlinked_chart = create_pca_component_scatter(
                            st.session_state["df_to_use"],
                            st.session_state["superlinked_pca"],
                            st.session_state["coloring_attribute"],
                            config,
                        )
                        st.altair_chart(superlinked_chart)


if __name__ == "__main__":
    cfg = OmegaConf.load("conf/config.yaml")
    main(cfg)
