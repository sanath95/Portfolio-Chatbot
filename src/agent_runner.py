from __future__ import annotations

from typing import AsyncGenerator, List, Dict
from langfuse import Langfuse, propagate_attributes

from src.agents.final_presentation import FinalPresentationAgent
from src.agents.orchestrator import OrchestratorAgent
from src.agents.professional_info import ProfessionalInfoAgent
from src.config import AppConfig
from src.models.schemas import DownstreamAgent, AgentEventUnion, OrchestratorEvent, ProfessionalInfoEvent, FinalPresentationEvent, AgentSource
from src.tools.retrieval import create_retrieval_deps
from src.vector_store.store import ProjectsVectorStore


class AgentRunner:
    """Main agent orchestration system.
    
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

    async def process_query(self, user_query: str, conversation: List[Dict[str, str]], session_id: str) -> AsyncGenerator[AgentEventUnion, None]:
        """Process a user query through the agent pipeline.
        
        Args:
            user_query: The user's question or request.
            conversation: List of user queries and generated responses over the past.
            
        Returns:
            Final response text.
        """
        langfuse = Langfuse()
        with langfuse.start_as_current_span(name="process_query", input={"user_query": user_query}, end_on_exit=True) as root_span:
            with propagate_attributes(session_id=session_id):
                # Step 1: Route the query
                orchestrator_output = await self.orchestrator.run(conversation)
                yield OrchestratorEvent(from_=AgentSource.ORCHESTRATOR, output=orchestrator_output.model_dump_json(),)

                # Step 2: Gather evidence if needed
                evidence_bundle = None
                for request in orchestrator_output.downstream_requests:
                    if request.agent == DownstreamAgent.PROFESSIONAL_INFO:
                        user_prompt_for_professional_info = self._format_professional_info_prompt(
                            user_query, request
                        )
                        evidence_bundle = await self.professional_info.run(
                            user_prompt_for_professional_info, self.retrieval_deps
                        )
                        yield ProfessionalInfoEvent(from_=AgentSource.PROFESSIONAL_INFO, output=evidence_bundle.model_dump_json(),)
                    elif request.agent == DownstreamAgent.PUBLIC_PERSONA:
                        # TODO: Implement public persona agent
                        print("Public persona agent not yet implemented")

                # Step 3: Stream final response
                final_response = ""
                async for partial_response in self.final_presentation.run(user_query, evidence_bundle, orchestrator_output):
                    final_response += partial_response
                    yield FinalPresentationEvent(from_=AgentSource.FINAL_PRESENTATION, output=partial_response,)
            root_span.update(output=final_response)

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