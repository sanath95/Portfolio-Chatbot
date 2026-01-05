"""Professional information retrieval agent."""

from __future__ import annotations

from pydantic_ai import Agent, RunContext, ModelSettings
from pymupdf import open as pdfopen

from src.config import AgentConfig
from src.models.schemas import EvidenceBundle
from src.tools.retrieval import RetrievalDeps, retrieve_and_rerank
from src.tools.github_repos import fetch_project_repos
from src.utils.gcp_buckets import GCSHelper
from src.utils.langfuse_client import get_langfuse_client


class ProfessionalInfoAgent:
    """Agent for retrieving and summarizing professional information.
    
    This agent accesses Sanath's resume, about document, and project
    documentation to gather evidence for answering queries.
    
    Attributes:
        config: Agent configuration.
        langfuse_client: Langfuse client.
        instructions: System instructions for the agent.
        agent: PydanticAI agent instance.
    """

    def __init__(self, config: AgentConfig) -> None:
        """Initialize the professional info agent.
        
        Args:
            config: Agent configuration.
        """
        self.config = config.professional_info
        self.gcs = GCSHelper()
        self.langfuse_client = get_langfuse_client()
        self.instructions = self._load_instructions()
        self.agent: Agent[RetrievalDeps, EvidenceBundle] = self._create_agent()
        self._register_tools()

    async def run(
        self, user_prompt: str, deps: RetrievalDeps, trace_id: str
    ) -> EvidenceBundle:
        """Run the agent with a user prompt.
        
        Args:
            user_prompt: The user's prompt including task and constraints.
            deps: Retrieval dependencies for document access.
            
        Returns:
            Evidence bundle with claims and coverage assessment.
        """
        generation = self.langfuse_client.generation(
            trace_id=trace_id,
            name="professional_info_retrieval",
            model=self.config.model,
            input={"user_prompt": user_prompt},
            metadata={"agent": "professional_info", "instructions": self.instructions}
        )
        deps.trace_id = trace_id
        result = await self.agent.run(
            instructions=self.instructions,
            user_prompt=user_prompt,
            deps=deps,
        )
        generation.end(
            output=result.output.model_dump()
        )
        return result.output

    def _create_agent(self) -> Agent[RetrievalDeps, EvidenceBundle]:
        """Create the PydanticAI agent.
        
        Returns:
            Configured Agent instance.
        """
        return Agent(
            self.config.model,
            deps_type=RetrievalDeps,
            output_type=EvidenceBundle,
            model_settings=ModelSettings(temperature=0, parallel_tool_calls=True)
        )

    def _register_tools(self) -> None:
        """Register tools with the agent."""

        @self.agent.tool
        def read_resume(context: RunContext[RetrievalDeps]) -> str:
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
            trace_id = context.deps.trace_id
            span = self.langfuse_client.span(
                trace_id=trace_id,
                name="resume",
                input={},
                metadata={"tool": "read_resume"}
            )
            resume = self.gcs.download_as_text(
                bucket_name=self.config.gcs_bucket,
                blob_path=self.config.tool_config.resume_path
            )
            span.end(output=resume)
            return resume
        
        @self.agent.tool
        def read_old_resume(context: RunContext[RetrievalDeps]) -> str:
            """Return the full text of Sanath's old resume.

            Use this tool when you need:
            - More details about Sanath's work experience in Nichesolv Private Limited, Bangalore, India.
            
            Returns:
                old_resume: Full text of Sanath's old resume.
            """
            trace_id = context.deps.trace_id
            span = self.langfuse_client.span(
                trace_id=trace_id,
                name="old_resume",
                input={},
                metadata={"tool": "read_old_resume"}
            )
            pdf_stream = self.gcs.download_to_stream(
                bucket_name=self.config.gcs_bucket,
                blob_path=self.config.tool_config.old_resume_path
            )
            with pdfopen(stream=pdf_stream) as doc:
                old_resume = "\n".join(str(page.get_text("text")) for page in doc)
            span.end(output=old_resume)
            return old_resume
        
        @self.agent.tool
        def read_transcript_of_records(context: RunContext[RetrievalDeps]) -> str:
            """Return the full text of Sanath's master's degree transcript of records.

            Use this tool when you need:
            - Information about Sanath's master's degree curriculum and his grades.
            
            Returns:
                transcript_of_records: Full text of Sanath's transcript of records.
            """
            trace_id = context.deps.trace_id
            span = self.langfuse_client.span(
                trace_id=trace_id,
                name="transcript",
                input={},
                metadata={"tool": "read_transcript_of_records"}
            )
            pdf_stream = self.gcs.download_to_stream(
                bucket_name=self.config.gcs_bucket,
                blob_path=self.config.tool_config.transcript_of_records_path
            )
            with pdfopen(stream=pdf_stream) as doc:
                transcript_of_records = "\n".join(str(page.get_text("text")) for page in doc)
            span.end(output=transcript_of_records)
            return transcript_of_records

        @self.agent.tool
        def read_about_sanath(context: RunContext[RetrievalDeps]) -> str:
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
            trace_id = context.deps.trace_id
            span = self.langfuse_client.span(
                trace_id=trace_id,
                name="about_sanath",
                input={},
                metadata={"tool": "read_about_sanath"}
            )
            about_me = self.gcs.download_as_text(
                bucket_name=self.config.gcs_bucket,
                blob_path=self.config.tool_config.about_me_path
            )
            span.end(output=about_me)
            return about_me

        
        @self.agent.tool
        async def fetch_github_repos(context: RunContext[RetrievalDeps]) -> str:
            """Fetch Sanath's GitHub repositories for project linking.
    
            This tool returns a JSON list of repository metadata including name, description, and GitHub URL.
            """
            trace_id = context.deps.trace_id
            span = self.langfuse_client.span(
                trace_id=trace_id,
                name="github",
                input={},
                metadata={"tool": "fetch_github_repos"}
            )
            repos = await fetch_project_repos(self.config.tool_config.github_repos_endpoint)
            span.end(output=repos)
            return repos

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
            trace_id = context.deps.trace_id
            span = self.langfuse_client.span(
                trace_id=trace_id,
                name="retrieve",
                input=search_query,
                metadata={"tool": "retrieve"}
            )
            ranked_docs = await retrieve_and_rerank(context.deps, search_query)
            span.end(output=ranked_docs)
            return ranked_docs

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