"""Final presentation agent for generating responses."""

from __future__ import annotations

from openai import AsyncOpenAI
from langfuse import observe

from src.config import AgentConfig
from src.models.schemas import EvidenceBundle, OrchestratorRoute
from src.tools.tracking import get_tracker


class FinalPresentationAgent:
    """Agent that transforms evidence into persuasive responses.
    
    This agent represents Sanath professionally by converting evidence
    bundles into polished, accurate, and persuasive responses.
    
    Attributes:
        config: Agent configuration.
        client: AsyncOpenAI client for API calls.
        tracker: Langfuse tracker instance.
        instructions: System instructions for the agent.
    """

    def __init__(self, config: AgentConfig) -> None:
        """Initialize the final presentation agent.
        
        Args:
            config: Agent configuration.
        """
        self.config = config
        self.client = AsyncOpenAI()
        self.tracker = get_tracker()
        self.instructions = self._load_instructions()

    @observe(name="final_presentation_agent")
    async def run(
        self,
        user_query: str,
        evidence_bundle: EvidenceBundle | None,
        orchestrator_output: OrchestratorRoute,
    ) -> str:
        """Generate a final response from evidence and routing decisions.
        
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

        response = await self.client.responses.create(
            model=self.config.final_presentation_model,
            instructions=self.instructions,
            input=[
                {
                    "role": "user",
                    "content": input_content,
                },
            ],
        )

        return response.output_text

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

        return (
            f"Original user query: {user_query}\n"
            f"Coverage assessment: {coverage}\n"
            f"Claims: {claims}\n"
            f"Relevant projects: {projects}\n"
            f"Safe redirect: {redirect}\n"
            f"Refusal directive: {orchestrator_output.refusal_directive.model_dump_json()}"
        )

    def _load_instructions(self) -> str:
        """Load system instructions from file.
        
        Returns:
            System instructions text.
        """
        if self.tracker and self.tracker.client:
            return self.tracker.client.get_prompt(self.config.final_presentation_instructions_langfuse_path).compile()
        return self.config.final_presentation_instructions_path.read_text(
            encoding="utf-8"
        )