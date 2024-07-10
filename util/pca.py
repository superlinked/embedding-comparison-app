import numpy as np

from sklearn.decomposition import PCA


def pca_transform(vectors: np.ndarray, num_factors: int = 2) -> np.ndarray:
    return PCA().fit_transform(vectors)[:, :num_factors]
