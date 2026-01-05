"""Langfuse client singleton for reuse across the application."""

from langfuse import Langfuse
from typing import Optional

_langfuse_client: Optional[Langfuse] = None

def get_langfuse_client() -> Langfuse:
    """Get or create the Langfuse client singleton.
    
    Returns:
        Langfuse client instance.
    """
    global _langfuse_client
    if _langfuse_client is None:
        _langfuse_client = Langfuse()
    return _langfuse_client