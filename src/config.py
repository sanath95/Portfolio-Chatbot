"""Configuration management for the portfolio chatbot system."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from os import getenv

@dataclass(frozen=True)
class VectorStoreConfig:
    """Configuration for Qdrant vector store.
    
    Attributes:
        url: Qdrant service URL.
        port: Qdrant service port.
        collection_name: Name of the collection to use.
        embedding_model: OpenAI embedding model identifier.
    """
    url: str = getenv("QDRANT_URL", "")
    api_key: str = getenv("QDRANT_API_KEY", "")
    collection_name: str = "sanath_projects"
    embedding_model: str = "text-embedding-3-small"

@dataclass(frozen=True)
class AgentProfile:
    model: str
    gcs_bucket: str
    blob_name: str
    langfuse_key: str
    
@dataclass(frozen=True)
class ProfessionalInfoProfile(AgentProfile):
    tool_config: ProfessionalInfoToolConfig
    
@dataclass(frozen=True)
class PublicPersonaProfile(AgentProfile):
    tool_config: PublicPersonaToolConfig

@dataclass(frozen=True)
class ProfessionalInfoToolConfig:
    resume_path: str = "documents/Sanath Vijay Haritsa - CV.tex"
    about_me_path: str = "documents/Sanath Vijay Haritsa - About Me.md"
    old_resume_path: str = "documents/Sanath Vijay Haritsa - Old CV.pdf"
    transcript_of_records_path: str = "documents/Sanath Vijay Haritsa - Transcript of Records.pdf"
    github_repos_endpoint: str = "https://api.github.com/users/sanath95/repos"
    
@dataclass(frozen=True)
class PublicPersonaToolConfig:
    instagram_account_info_endpoint = "https://graph.instagram.com/me"
    instagram_media_endpoint = "https://graph.instagram.com/me/media"
    instagram_account_info_fields = "username,media_count"
    instagram_media_fields = "id,caption,like_count,comments_count,media_type,timestamp,permalink"

@dataclass(frozen=True)
class AgentConfig:
    """Configuration for AI agents."""

    orchestrator: AgentProfile = AgentProfile(
        model="o3-mini",
        gcs_bucket=getenv("GCS_BUCKET", "my-portfolio-chatbot-bucket"),
        blob_name="prompts/orchestrator.txt",
        langfuse_key="orchestrator"
    )

    professional_info: ProfessionalInfoProfile = ProfessionalInfoProfile(
        model="gpt-5.2",
        gcs_bucket=getenv("GCS_BUCKET", "my-portfolio-chatbot-bucket"),
        blob_name="prompts/professional_info.txt",
        langfuse_key="professional_info",
        tool_config=ProfessionalInfoToolConfig()
    )
    
    public_persona: PublicPersonaProfile = PublicPersonaProfile(
        model="gpt-5.2",
        gcs_bucket=getenv("GCS_BUCKET", "my-portfolio-chatbot-bucket"),
        blob_name="prompts/public_persona.txt",
        langfuse_key="public_persona",
        tool_config=PublicPersonaToolConfig()
    )

    final_presentation: AgentProfile = AgentProfile(
        model="gpt-5-mini",
        gcs_bucket=getenv("GCS_BUCKET", "my-portfolio-chatbot-bucket"),
        blob_name="prompts/final_presentation.txt",
        langfuse_key="final_presentation",
    )

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
    agent: AgentConfig = AgentConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    
@dataclass(frozen=True)
class GradioConfig:
    """Gradio UI configuration
    
    Attributes:
        header_html_path: Path for the chatbot header html.
        footer_html_path: Path for the chatbot footer html.
        image_path: Path for Sanath's photo.
    """
    header_html_path: Path = Path("./static/header.html")
    footer_html_path: Path = Path("./static/footer.html")
    image_path: Path = Path("./static/sanath_vijay_haritsa.png")