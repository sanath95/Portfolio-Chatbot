"""Langfuse tracking utilities for observability."""

from __future__ import annotations

from typing import Optional

from langfuse import Langfuse

from src.config import LangfuseConfig


class LangfuseTracker:
    """Wrapper for Langfuse tracking operations.
    
    Attributes:
        config: Langfuse configuration.
        client: Langfuse client instance.
    """
    
    def __init__(self, config: LangfuseConfig) -> None:
        """Initialize Langfuse tracker.
        
        Args:
            config: Langfuse configuration.
        """
        self.config = config
        self.client = None
        
        if config.enabled and config.is_configured():
            self.client = Langfuse(
                public_key=config.public_key,
                secret_key=config.secret_key,
                host=config.host,
            )
    
    def is_enabled(self) -> bool:
        """Check if tracking is enabled and configured."""
        return self.client is not None
    
    def flush(self) -> None:
        """Flush pending traces to Langfuse."""
        if self.client:
            self.client.flush()

    def update_current_trace(self, **kwargs) -> None:
        """Update current trace with given keyword arguments"""
        if self.client:
            self.client.update_current_trace(**kwargs)

# Global tracker instance
_tracker: Optional[LangfuseTracker] = None


def init_tracker(config: LangfuseConfig) -> LangfuseTracker:
    """Initialize the global Langfuse tracker.
    
    Args:
        config: Langfuse configuration.
        
    Returns:
        Initialized tracker instance.
    """
    global _tracker
    _tracker = LangfuseTracker(config)
    return _tracker