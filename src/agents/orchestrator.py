"""Orchestrator agent for intent interpretation and routing."""

from __future__ import annotations

from openai import AsyncOpenAI
from langfuse import observe

from src.config import AgentConfig
from src.models.schemas import OrchestratorRoute


class OrchestratorAgent:
    """Intent interpreter and router for the chatbot system.
    
    This agent analyzes user queries, classifies evaluation intent,
    and routes requests to appropriate downstream agents.
    
    Attributes:
        config: Agent configuration.
        client: AsyncOpenAI client for API calls.
        instructions: System instructions for the agent.
    """

    def __init__(self, config: AgentConfig) -> None:
        """Initialize the orchestrator agent.
        
        Args:
            config: Agent configuration.
        """
        self.config = config
        self.client = AsyncOpenAI()
        self.instructions = self._load_instructions()

    @observe(name="orchestrator_agent")
    async def run(self, user_message: str) -> OrchestratorRoute:
        """Process a user message and determine routing.
        
        Args:
            user_message: The user's query.
            
        Returns:
            Routing decision with downstream requests.
            
        Raises:
            Exception: If the agent fails to parse the response.
        """
        response = await self.client.responses.parse(
            model=self.config.orchestrator_model,
            instructions=self.instructions,
            input=[
                {
                    "role": "user",
                    "content": user_message,
                },
            ],
            text_format=OrchestratorRoute,
        )

        if response.output_parsed:
            return response.output_parsed

        raise Exception(f"OrchestratorAgent failed to parse response: {response}")

    def _load_instructions(self) -> str:
        """Load system instructions from file.
        
        Returns:
            System instructions text.
        """
        return self.config.orchestrator_instructions_path.read_text(encoding="utf-8")