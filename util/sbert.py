import numpy as np

from sentence_transformers import SentenceTransformer


def embed_with_model(text_to_embed: list[str], embedding_model: str) -> np.ndarray:
    model = SentenceTransformer(embedding_model)
    return model.encode(text_to_embed)
