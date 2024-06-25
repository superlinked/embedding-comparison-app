import numpy as np
import umap


def umap_transform(vectors: np.ndarray) -> np.ndarray:
    return umap.UMAP().fit_transform(vectors)
