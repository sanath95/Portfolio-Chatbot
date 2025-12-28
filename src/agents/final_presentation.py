"""Final presentation agent for generating responses."""

from __future__ import annotations
from typing import AsyncGenerator

from openai import AsyncOpenAI
from langfuse import observe, get_client

from src.config import AgentConfig
from src.models.schemas import EvidenceBundle, OrchestratorRoute


class FinalPresentationAgent:
    """Agent that transforms evidence into persuasive responses.
    
    This agent represents Sanath professionally by converting evidence
    bundles into polished, accurate, and persuasive responses.
    
    Attributes:
        config: Agent configuration.
        client: AsyncOpenAI client for API calls.
        langfuse_client: Langfuse client.
        instructions: System instructions for the agent.
    """

    def __init__(self, config: AgentConfig) -> None:
        """Initialize the final presentation agent.
        
        Args:
            config: Agent configuration.
        """
        self.config = config.final_presentation
        self.client = AsyncOpenAI()
        self.langfuse_client = get_client()
        self.instructions = self._load_instructions()

    @observe(name="final_presentation_agent", capture_input=True, capture_output=True)
    async def run(
        self,
        user_query: str,
        evidence_bundle: EvidenceBundle | None,
        orchestrator_output: OrchestratorRoute,
    ) -> AsyncGenerator[str, None]:
        """Stream the final response from evidence and routing decisions.
        
        Args:
            user_query: The original user query.
            evidence_bundle: Evidence gathered by professional info agent.
            orchestrator_output: Routing decisions from orchestrator.
            
        Returns:
            Final response text.
        """
        # Format input for the agent
        input_content = self._format_input(
            user_query, evidence_bundle, orchestrator_output
        )
        self.langfuse_client.update_current_span(metadata={"final_presentation_instructions": self.instructions, "user_prompt": input_content})
        async with self.client.responses.stream(
            model=self.config.model,
            instructions=self.instructions,
            input=[
                {
                    "role": "user",
                    "content": input_content
                }
            ],
            reasoning={"effort": "medium"}
        ) as stream:
            async for event in stream:
                if event.type == "response.output_text.delta":
                    yield event.delta
                if event.type == "response.completed":
                    break

    def _format_input(
        self,
        user_query: str,
        evidence_bundle: EvidenceBundle | None,
        orchestrator_output: OrchestratorRoute,
    ) -> str:
        """Format input for the final presentation agent.
        
        Args:
            user_query: The original user query.
            evidence_bundle: Evidence gathered by professional info agent.
            orchestrator_output: Routing decisions from orchestrator.
            
        Returns:
            Formatted input string.
        """
        if evidence_bundle:
            coverage = evidence_bundle.coverage_assessment.model_dump_json()
            claims = [claim.model_dump_json() for claim in evidence_bundle.claims]
            projects = ", ".join(evidence_bundle.project_leads or [])
            redirect = evidence_bundle.safe_redirect_if_missing or "none"
        else:
            coverage = "none"
            claims = "none"
            projects = "none"
            redirect = "none"
            
        user_input = (
            f"Original user query: {user_query}\n"
            f"Coverage assessment: {coverage}\n"
            f"Claims: {claims}\n"
            f"Relevant projects: {projects}\n"
            f"Safe redirect: {redirect}\n"
            f"Refusal directive: {orchestrator_output.refusal_directive.model_dump_json()}\n"
        )
        
        if orchestrator_output.downstream_requests:
            user_input += f"Task directive: {orchestrator_output.downstream_requests[-1].task}"
            
        return user_input

    def _load_instructions(self) -> str:
        """Load system instructions from file.
        
        Returns:
            System instructions text.
        """
        try:
            return self.langfuse_client.get_prompt(self.config.langfuse_key).prompt
        except:
            return self.config.instructions_path.read_text(encoding="utf-8")