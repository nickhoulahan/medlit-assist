from sentence_transformers import SentenceTransformer


class SBertEmbeddingsService:
    """
    Service to generate sentence embeddings using SBERT
    """

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    @classmethod
    def get_embeddings(cls, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a list of texts
        """
        return cls.model.encode(texts).tolist()

    @classmethod
    def get_embedding(cls, text: str) -> list[float]:
        """
        Generate embedding for a single text
        """
        return cls.model.encode(text).tolist()
