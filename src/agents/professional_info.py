"""Professional information retrieval agent."""

from __future__ import annotations

from pydantic_ai import Agent, RunContext
from aiofiles import open as aopen

from src.config import AgentConfig
from src.models.schemas import EvidenceBundle
from src.tools.retrieval import RetrievalDeps, retrieve_and_rerank


class ProfessionalInfoAgent:
    """Agent for retrieving and summarizing professional information.
    
    This agent accesses Sanath's resume, about document, and project
    documentation to gather evidence for answering queries.
    
    Attributes:
        config: Agent configuration.
        agent: PydanticAI agent instance.
    """

    def __init__(self, config: AgentConfig) -> None:
        """Initialize the professional info agent.
        
        Args:
            config: Agent configuration.
        """
        self.config = config
        self.instructions = self._load_instructions()
        self.agent: Agent[RetrievalDeps, EvidenceBundle] = self._create_agent()
        self._register_tools()

    async def run(
        self, user_prompt: str, deps: RetrievalDeps
    ) -> EvidenceBundle:
        """Run the agent with a user prompt.
        
        Args:
            user_prompt: The user's prompt including task and constraints.
            deps: Retrieval dependencies for document access.
            
        Returns:
            Evidence bundle with claims and coverage assessment.
        """
        result = await self.agent.run(
            user_prompt=user_prompt,
            deps=deps,
        )
        return result.output

    def _create_agent(self) -> Agent[RetrievalDeps, EvidenceBundle]:
        """Create the PydanticAI agent.
        
        Returns:
            Configured Agent instance.
        """
        return Agent(
            self.config.professional_info_model,
            instructions=self.instructions,
            deps_type=RetrievalDeps,
            output_type=EvidenceBundle
        )

    def _register_tools(self) -> None:
        """Register tools with the agent."""

        @self.agent.tool_plain
        async def read_resume() -> str:
            """Return the full text of Sanath’s resume.

            Use this tool when you need:
            - Accurate facts about his education, roles, dates, locations, and employers.
            - A list of his technical skills, tools, and technologies.
            - High level project summaries.

            Treat the returned text as the single source of truth for all
            resume-level information. When answering questions about his
            background, always align your statements with this content.
            
            Returns:
                resume: Full text of Sanath’s resume.
            """
            async with aopen(self.config.resume_path, encoding="utf-8") as f:
                return await f.read()

        @self.agent.tool_plain
        async def read_about_sanath() -> str:
            """Return the narrative 'About Sanath' profile text.

            Use this tool when you need:
            - A more story-like description of his technical background.
            - Details about responsibilities, ways of working, and collaboration style.
            - Material to answer questions about motivation, culture fit, and soft skills.
            - Phrasing that explains how he approaches problems and delivers value.

            Use this content to make answers sound human, nuanced, and persuasive,
            while still staying faithful to what is written. Do not invent new
            achievements; build your explanations from this text.
            
            Returns:
                about_me: Narrative profile text about Sanath.
            """
            async with aopen(self.config.about_me_path, encoding="utf-8") as f:
                return await f.read()

        @self.agent.tool
        async def retrieve(
            context: RunContext[RetrievalDeps], search_query: str
        ) -> list[str]:
            """Fetch project-specific, detailed evidence from internal project reports or documentation.

            **When to use**
            - The user explicitly asks about a specific project, system, or implementation
            - Resume/About-Sanath information is insufficiently detailed
            - You need concrete technical depth (architecture, methods, decisions)
            
            **How to formulate the query**
            - Use one precise, narrowly scoped query
            - Focus strictly on the exact information requested
            - Use only essential keywords
            - Keep the query short and specific
            - Never include Sanath’s name
            - Do not include model names, or technology that are not specified in the user query

            **How NOT to formulate the query**
            - Do not broaden the query beyond what was asked
            - Do not add assumptions, context, or inferred goals
            - Do not add extra context or details
            - Do not include model names, or technology buzzwords unless explicitly mentioned in the user query
            
            **What this tool provides**
            - Deep, project-level factual evidence
            - Technical details not present in resume or narrative descriptions

            **Failure handling**
            - If no relevant information is found, explicitly report: “No evidence found.”

            Do not guess, infer, or fill gaps with general knowledge.
            
            Args:
                search_query: The search query.
                
            Returns:
                ranked_docs: List of relevant text chunks from Sanath’s portfolio documents.
            """
            return await retrieve_and_rerank(context.deps, search_query)

    def _load_instructions(self) -> str:
        """Load system instructions from file.
        
        Returns:
            System instructions text.
        """
        return self.config.professional_info_instructions_path.read_text(
            encoding="utf-8"
        )