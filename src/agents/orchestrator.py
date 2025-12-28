"""Orchestrator agent for intent interpretation and routing."""

from __future__ import annotations

from typing import List, Dict, cast
from openai import AsyncOpenAI
from openai.types.responses import ResponseInputParam
from langfuse import observe, get_client

from src.config import AgentConfig
from src.models.schemas import OrchestratorRoute


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
        self.langfuse_client = get_client()
        self.instructions = self._load_instructions()

    @observe(name="orchestrator_agent", capture_input=True, capture_output=True)
    async def run(self, conversation: List[Dict[str, str]]) -> OrchestratorRoute:
        """Process a user message and determine routing.
        
        Args:
            conversation: List of user queries and generated responses over the past.
            
        Returns:
            Routing decision with downstream requests.
            
        Raises:
            Exception: If the agent fails to parse the response.
        """
        self.langfuse_client.update_current_span(metadata={"orchestrator_instructions": self.instructions})
        response = await self.client.responses.parse(
            model=self.config.model,
            instructions=self.instructions,
            input=cast(ResponseInputParam, conversation),
            text_format=OrchestratorRoute,
            reasoning={"effort": "high"}
        )

        if response.output_parsed:
            return response.output_parsed

        raise Exception(f"OrchestratorAgent failed to parse response: {response}")

    def _load_instructions(self) -> str:
        """Load system instructions from langfuse or file.
        
        Returns:
            System instructions text.
        """
        try:
            return self.langfuse_client.get_prompt(self.config.langfuse_key).prompt
        except:
            return self.config.instructions_path.read_text(encoding="utf-8")