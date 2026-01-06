"""Wrapper for creating and loading Qdrant-backed vector stores."""

from __future__ import annotations

from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException

from src.config import VectorStoreConfig


class ProjectsVectorStore:
    """Wrapper for creating or loading a Qdrant-backed vector store.
    
    Attributes:
        config: Vector store configuration.
        embedding: OpenAI embeddings model.
    """

    def __init__(
        self,
        config: VectorStoreConfig,
    ) -> None:
        """Initialize vector store configuration.
        
        Args:
            config: Vector store configuration.
        """
        self.config = config
        self.embedding = OpenAIEmbeddings(model=config.embedding_model)

    def get_vector_store(self) -> QdrantVectorStore:
        """Load an existing vector store or create a new one if it does not exist.
        
        Returns:
            Initialized QdrantVectorStore instance.
        """
        client = QdrantClient(url=self.config.url, api_key=self.config.api_key)
        try:
            if client.collection_exists(self.config.collection_name):
                return self._load()
            else:
                raise Exception("Collection does not exist!")
        except ResponseHandlingException as e:
            raise Exception(f"Qdrant DB not running! {e}")

    def _load(self) -> QdrantVectorStore:
        """Load an existing Qdrant collection.
        
        Returns:
            Loaded QdrantVectorStore.
        """
        return QdrantVectorStore.from_existing_collection(
            embedding=self.embedding,
            collection_name=self.config.collection_name,
            url=self.config.url,
            api_key=self.config.api_key
        )