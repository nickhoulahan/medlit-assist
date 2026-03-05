from pathlib import Path
from typing import Dict, List

from src.medlit_agent.pmc_service.embeddings_service import SBertEmbeddingsService


class ChromaDB:
    """
    ChromaDB vector database service for storing and querying document embeddings and metadata.
    """

    def __init__(
        self,
        collection_name: str = "pubmed_central_articles",
        persist_directory: str | Path | None = None,
    ):
        from chromadb import PersistentClient

        default_cache_dir = Path(__file__).resolve().parent / "chromadb"
        db_path = (
            Path(persist_directory)
            if persist_directory is not None
            else default_cache_dir
        )
        db_path.mkdir(parents=True, exist_ok=True)

        self.client = PersistentClient(path=str(db_path))
        self.collection = self.client.get_or_create_collection(name=collection_name)

    @staticmethod
    def _split_text(
        text: str, chunk_size: int = 1000, chunk_overlap: int = 200
    ) -> List[str]:
        if not text:
            return []

        chunks: List[str] = []
        step = max(1, chunk_size - chunk_overlap)
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += step
        return chunks

    def add(self, pmcid: str, texts: List[Dict[str, str]]):
        metadatas = []
        documents = []
        for text in texts:
            body = text["body"]
            chunks = self._split_text(body, chunk_size=1000, chunk_overlap=200)
            for chunk in chunks:
                documents.append(chunk)
                metadatas.append(
                    {"title": text["title"], "text": chunk, "pmcid": pmcid}
                )

        embeddings_service = SBertEmbeddingsService()
        embeddings = embeddings_service.get_embeddings([doc for doc in documents])
        self.collection.add(
            ids=[f"{pmcid}_{i}" for i in range(len(documents))],
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def query(
        self, query_embedding: List[float], n_results: int
    ) -> List[Dict[str, str]]:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["metadatas"],
        )
        metadatas = results.get("metadatas", [])
        if not metadatas:
            return []
        return metadatas[0]

    def _document_exists(self, pmcid: str) -> bool:
        # Check if any document with the given PMCID exists in the collection
        results = self.collection.query(
            query_embeddings=[[0] * 384],  # Dummy embedding for existence check
            n_results=1,
            include=["metadatas"],
        )
        for metadata in results["metadatas"]:
            if metadata.get("pmcid") == pmcid:
                return True
        return False
