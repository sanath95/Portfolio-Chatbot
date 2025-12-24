"""Wrapper for creating and loading Qdrant-backed vector stores."""

from __future__ import annotations

from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from src.config import ProcessorConfig, VectorStoreConfig
from src.vector_store.processor import FileProcessor


class ProjectsVectorStore:
    """Wrapper for creating or loading a Qdrant-backed vector store.
    
    Attributes:
        config: Vector store configuration.
        embedding: OpenAI embeddings model.
    """

    def __init__(
        self,
        config: VectorStoreConfig,
        processor_config: ProcessorConfig | None = None,
    ) -> None:
        """Initialize vector store configuration.
        
        Args:
            config: Vector store configuration.
            processor_config: Document processor configuration (for creating new stores).
        """
        self.config = config
        self.processor_config = processor_config
        self.embedding = OpenAIEmbeddings(model=config.embedding_model)

    def get_vector_store(self) -> QdrantVectorStore:
        """Load an existing vector store or create a new one if it does not exist.
        
        Returns:
            Initialized QdrantVectorStore instance.
        """
        client = QdrantClient(url=self.config.url, port=self.config.port)

        if client.collection_exists(self.config.collection_name):
            return self._load()
        else:
            return self._create()

    def _create(self) -> QdrantVectorStore:
        """Create a new Qdrant collection from processed documents.
        
        Returns:
            Newly created QdrantVectorStore.
            
        Raises:
            ValueError: If processor_config is not provided.
        """
        if self.processor_config is None:
            raise ValueError(
                "processor_config must be provided to create a new vector store"
            )

        processor = FileProcessor(self.processor_config)
        documents = processor.build_documents()

        return QdrantVectorStore.from_documents(
            documents=documents,
            embedding=self.embedding,
            collection_name=self.config.collection_name,
            url=self.config.url,
            port=self.config.port,
        )

    def _load(self) -> QdrantVectorStore:
        """Load an existing Qdrant collection.
        
        Returns:
            Loaded QdrantVectorStore.
        """
        return QdrantVectorStore.from_existing_collection(
            embedding=self.embedding,
            collection_name=self.config.collection_name,
            url=self.config.url,
            port=self.config.port,
        )