"""Main entry point for the portfolio chatbot system."""

from __future__ import annotations

import asyncio
from langfuse import observe, get_client

from src.agents.final_presentation import FinalPresentationAgent
from src.agents.orchestrator import OrchestratorAgent
from src.agents.professional_info import ProfessionalInfoAgent
from src.config import AppConfig
from src.models.schemas import DownstreamAgent
from src.tools.retrieval import create_retrieval_deps
from src.tools.tracking import init_tracker
from src.vector_store.store import ProjectsVectorStore


class PortfolioChatbot:
    """Main chatbot orchestration system.
    
    This class coordinates the three-agent system to process user queries
    and generate responses about Sanath's professional profile.
    
    Attributes:
        config: Application configuration.
        orchestrator: Orchestrator agent for routing.
        professional_info: Professional info agent for evidence gathering.
        final_presentation: Final presentation agent for response generation.
        retrieval_deps: Dependencies for retrieval operations.
    """

    def __init__(self, config: AppConfig | None = None) -> None:
        """Initialize the chatbot system.
        
        Args:
            config: Application configuration. Uses defaults if not provided.
        """
        self.config = config or AppConfig()
        
        # Initialize Langfuse tracking
        self.tracker = init_tracker(self.config.langfuse)
        if self.tracker.is_enabled():
            print("✓ Langfuse tracking enabled")
            print(f"  Dashboard: {self.config.langfuse.host}")
        else:
            print("✗ Langfuse tracking disabled (not configured)")

        # Initialize agents
        self.orchestrator = OrchestratorAgent(self.config.agent)
        self.professional_info = ProfessionalInfoAgent(self.config.agent)
        self.final_presentation = FinalPresentationAgent(self.config.agent)

        # Initialize retrieval system
        self.retrieval_deps = self._setup_retrieval()

    def _setup_retrieval(self):
        """Set up the vector store and retrieval dependencies.
        
        Returns:
            Configured retrieval dependencies.
        """
        # Initialize vector store
        vector_store_manager = ProjectsVectorStore(
            self.config.vector_store,
            self.config.processor,
        )
        vector_store = vector_store_manager.get_vector_store()

        # Create retriever
        retriever = vector_store.as_retriever(
            search_type=self.config.retrieval.search_type,
            search_kwargs={"k": self.config.retrieval.retrieval_k},
        )

        # Create retrieval dependencies
        return create_retrieval_deps(retriever, self.config.retrieval)

    @observe(name="process_query")
    async def process_query(self, user_query: str) -> str:
        """Process a user query through the agent pipeline.
        
        Args:
            user_query: The user's question or request.
            
        Returns:
            Final response text.
        """
        # Add custom metadata to the trace
        if self.tracker.is_enabled():
            self.tracker.update_current_trace(
                user_id="demo_user"
            )
            
        # Step 1: Route the query
        orchestrator_output = await self.orchestrator.run(user_query)

        # Step 2: Gather evidence if needed
        evidence_bundle = None
        for request in orchestrator_output.downstream_requests:
            if request.agent == DownstreamAgent.PROFESSIONAL_INFO:
                user_prompt = self._format_professional_info_prompt(
                    user_query, request
                )
                evidence_bundle = await self.professional_info.run(
                    user_prompt, self.retrieval_deps
                )
            elif request.agent == DownstreamAgent.PUBLIC_PERSONA:
                # TODO: Implement public persona agent
                print("Public persona agent not yet implemented")

        # Step 3: Generate final response
        final_response = await self.final_presentation.run(
            user_query, evidence_bundle, orchestrator_output
        )
        
        # Flush traces to Langfuse
        if self.tracker.is_enabled():
            self.tracker.flush()
            
        return final_response

    @staticmethod
    def _format_professional_info_prompt(
        user_query: str, request
    ) -> str:
        """Format the prompt for the professional info agent.
        
        Args:
            user_query: The original user query.
            request: The downstream request from orchestrator.
            
        Returns:
            Formatted prompt string.
        """
        return (
            f"Original user query: {user_query}\n"
            f"Task: {request.task}\n"
            f"Constraints: {request.constraints or 'none'}"
        )


async def main() -> None:
    """Main function to run example queries."""
    # Example queries
    queries = [
        "what are the contributions of his thesis?",
        "llm content evaluation",
        "bikes data sources",
        "introduce yourself",
        "magic trick",
        "tell me a joke",
    ]

    # Initialize chatbot
    chatbot = PortfolioChatbot()

    # Process each query
    for query in queries:
        print(f"\nQuery: {query}")
        print("=" * 80)
        response = await chatbot.process_query(query)
        print(response)
        print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())