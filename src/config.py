"""Configuration management for the portfolio chatbot system."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from os import getenv

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class VectorStoreConfig:
    """Configuration for Qdrant vector store.
    
    Attributes:
        url: Qdrant service URL.
        port: Qdrant service port.
        collection_name: Name of the collection to use.
        embedding_model: OpenAI embedding model identifier.
    """
    url: str = "http://localhost:6333"
    port: int = 6333
    collection_name: str = "sanath_projects"
    embedding_model: str = "text-embedding-3-small"


@dataclass(frozen=True)
class ProcessorConfig:
    """Configuration for document processing.
    
    Attributes:
        input_glob: Glob pattern for input files.
        config_path: Path to JSON metadata configuration.
        num_threads: Number of threads for PDF processing.
    """
    input_glob: str = "./data/*"
    config_path: Path = Path("./configs/data_config.json")
    num_threads: int = 4


@dataclass(frozen=True)
class AgentConfig:
    """Configuration for AI agents.
    
    Attributes:
        orchestrator_model: Model for orchestrator agent.
        professional_info_model: Model for professional info agent.
        final_presentation_model: Model for final presentation agent.
        orchestrator_instructions_path: Path to orchestrator instructions.
        professional_info_instructions_path: Path to professional info instructions.
        final_presentation_instructions_path: Path to final presentation instructions.
        resume_path: Path to resume file.
        about_me_path: Path to about me file.
    """
    orchestrator_model: str = "o3-mini"
    professional_info_model: str = "gpt-5.2"
    final_presentation_model: str = "gpt-5-mini"
    orchestrator_instructions_path: Path = Path("./prompts/orchestrator.txt")
    professional_info_instructions_path: Path = Path("./prompts/professional_info.txt")
    final_presentation_instructions_path: Path = Path("./prompts/final_presentation.txt")
    resume_path: Path = Path("./prompts/Sanath Vijay Haritsa - CV.tex")
    about_me_path: Path = Path("./prompts/Sanath Vijay Haritsa - About Me.md")


@dataclass(frozen=True)
class RetrievalConfig:
    """Configuration for retrieval and reranking.
    
    Attributes:
        search_type: Type of search to perform.
        retrieval_k: Number of documents to retrieve.
        reranker_model: Model for reranking.
        reranker_threshold: Minimum score threshold for reranked documents.
    """
    search_type = "similarity"
    retrieval_k: int = 10
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_threshold: float = 0.0

    
@dataclass(frozen=True)
class LangfuseConfig:
    """Configuration for Langfuse tracking.
    
    Attributes:
        enabled: Whether to enable Langfuse tracking.
        public_key: Langfuse public key.
        secret_key: Langfuse secret key.
        host: Langfuse host URL.
    """
    enabled: bool = True
    public_key: str = getenv("LANGFUSE_PUBLIC_KEY", "")
    secret_key: str = getenv("LANGFUSE_SECRET_KEY", "")
    host: str = getenv("LANGFUSE_HOST", "http://localhost:3000")
    
    def is_configured(self) -> bool:
        """Check if Langfuse is properly configured."""
        return bool(self.public_key and self.secret_key)
    

@dataclass(frozen=True)
class AppConfig:
    """Main application configuration.
    
    Attributes:
        vector_store: Vector store configuration.
        processor: Document processor configuration.
        agent: Agent configuration.
        retrieval: Retrieval configuration.
        langfuse: Langfuse tracking configuration.
    """
    vector_store: VectorStoreConfig = VectorStoreConfig()
    processor: ProcessorConfig = ProcessorConfig()
    agent: AgentConfig = AgentConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    langfuse: LangfuseConfig = LangfuseConfig()