"""Orchestrator agent for intent interpretation and routing."""

from __future__ import annotations

from typing import List, Dict, cast
from openai import AsyncOpenAI
from openai.types.responses import ResponseInputParam

from src.config import AgentConfig
from src.models.schemas import OrchestratorRoute
from src.utils.gcp_buckets import GCSHelper
from src.utils.langfuse_client import get_langfuse_client


class OrchestratorAgent:
    """Intent interpreter and router for the chatbot system.
    
    This agent analyzes user queries, classifies evaluation intent,
    and routes requests to appropriate downstream agents.
    
    Attributes:
        config: Agent configuration.
        client: AsyncOpenAI client for API calls.
        langfuse_client: Langfuse client.
        instructions: System instructions for the agent.
    """

    def __init__(self, config: AgentConfig) -> None:
        """Initialize the orchestrator agent.
        
        Args:
            config: Agent configuration.
        """
        self.config = config.orchestrator
        self.client = AsyncOpenAI()
        self.gcs = GCSHelper()
        self.langfuse_client = get_langfuse_client()
        self.instructions = self._load_instructions()

    async def run(self, conversation: List[Dict[str, str]], trace_id: str) -> OrchestratorRoute:
        """Process a user message and determine routing.
        
        Args:
            conversation: List of user queries and generated responses over the past.
            
        Returns:
            Routing decision with downstream requests.
            
        Raises:
            Exception: If the agent fails to parse the response.
        """
        generation = self.langfuse_client.generation(
            trace_id=trace_id,
            name="orchestrator_routing",
            model=self.config.model,
            input=conversation,
            metadata={"agent": "orchestrator", "instructions": self.instructions}
        )
        response = await self.client.responses.parse(
            model=self.config.model,
            instructions=self.instructions,
            input=cast(ResponseInputParam, conversation),
            text_format=OrchestratorRoute,
            reasoning={"effort": "high"}
        )

        result = response.output_parsed
        if result:
            generation.end(
                output=result.model_dump(),
            )
            return result

        raise Exception(f"OrchestratorAgent failed to parse response: {response}")

    def _load_instructions(self) -> str:
        """Load system instructions from langfuse or file.
        
        Returns:
            System instructions text.
        """
        try:
            return self.langfuse_client.get_prompt(self.config.langfuse_key).prompt
        except:
            return self.gcs.download_as_text(
                bucket_name=self.config.gcs_bucket,
                blob_path=self.config.blob_name
            )