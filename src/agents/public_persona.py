"""Public persona retrieval agent."""

from __future__ import annotations

from pydantic_ai import Agent, ModelSettings, RunContext
from typing import Dict

from src.config import AgentConfig
from src.models.schemas import PublicArtifacts
from src.tools.social_media_retrieval import get_instagram_posts, get_youtube_videos
from src.utils.gcp_buckets import GCSHelper
from src.utils.langfuse_client import get_langfuse_client

class PublicPersonaAgent:
    def __init__(self, config: AgentConfig) -> None:
        """Initialize the public persona agent.
        
        Args:
            config: Agent configuration.
        """
        self.config = config.public_persona
        self.gcs = GCSHelper()
        self.langfuse_client = get_langfuse_client()
        self.instructions = self._load_instructions()
        self.agent: Agent[Dict[str, str], PublicArtifacts] = self._create_agent()
        self._register_tools()
        
    def _create_agent(self) -> Agent[Dict[str, str], PublicArtifacts]:
        """Create the PydanticAI agent.
        
        Returns:
            Configured Agent instance.
        """
        return Agent(
            self.config.model,
            output_type=PublicArtifacts,
            model_settings=ModelSettings(temperature=0, parallel_tool_calls=True),
            deps_type=Dict[str, str]
        )
    
    async def run(self, user_prompt: str, trace_id: str) -> PublicArtifacts:
        """Run the agent with a user prompt.
        
        Args:
            user_prompt: The user's prompt including task and constraints.
            
        Returns:
            Public artifacts with account metadata and retrieved artifacts.
        """
        generation = self.langfuse_client.generation(
            trace_id=trace_id,
            name="public_persona_retrieval",
            model=self.config.model,
            input={"user_prompt": user_prompt},
            metadata={"agent": "public_persona", "instructions": self.instructions}
        )
        result = await self.agent.run(
            instructions=self.instructions,
            user_prompt=user_prompt,
            deps={"trace_id": trace_id}
        )
        generation.end(
            output=result.output.model_dump()
        )
        return result.output
        
    def _register_tools(self):
        """Register tools with the agent."""
        
        @self.agent.tool
        async def get_instagram_content(context: RunContext[Dict[str, str]]) -> str:
            """Fetch public Instagram content for Sanath.

            Retrieves Instagram media items using the Instagram Basic Display API.
            Returns posts sorted by engagement indicators (likes and comments).

            Args:
                None

            Returns:
                str: A JSON-serialized list of post objects. Each object includes:
                    - id (str): The Instagram media ID.
                    - caption (str): The post title, if available.
                    - media_type (str): The type of media (e.g., IMAGE, VIDEO, CAROUSEL).
                    - timestamp (str): Normalized publish timestamp ISO8601.
                    - like_count (int): Number of likes.
                    - comments_count (int): Number of comments.
                    - permalink (str): Public URL to the post.

            Raises:
                requests.HTTPError: If an HTTP request to the Instagram API fails.
            """            
            trace_id = context.deps.get("trace_id")
            span = self.langfuse_client.span(
                trace_id=trace_id,
                name="instagram",
                input={},
                metadata={"tool": "get_instagram_content"}
            )
            account_info_endpoint = self.config.tool_config.instagram_account_info_endpoint
            media_endpoint = self.config.tool_config.instagram_media_endpoint
            account_info_fields = self.config.tool_config.instagram_account_info_fields
            media_fields = self.config.tool_config.instagram_media_fields
            result = await get_instagram_posts(account_info_endpoint, media_endpoint, account_info_fields, media_fields)
            span.end(output=result)
            return result
        
        @self.agent.tool
        async def get_youtube_content(context: RunContext[Dict[str, str]]):
            """Fetch all uploaded YouTube videos and channel metadata for Sanath.

            Uses the YouTube Data API to list a user’s channel uploads, retrieve
            engagement statistics for each video, and sort videos by views and engagement.

            Args:
                None

            Returns:
                str: A JSON-serialized object with:
                    - channel (dict): Channel metadata with keys:
                        - id (str): YouTube channel ID.
                        - title (str): Channel title.
                        - description (str): Channel description.
                        - subscriberCount (str | int): Subscriber count.
                    - videos (List[dict]): A list of video objects. Each video includes:
                        - videoId (str): The video ID.
                        - url (str): Full YouTube watch URL.
                        - title (str): Video title.
                        - description (str): Video description.
                        - published_at (str): Normalized publish timestamp ISO8601.
                        - viewCount (int | str): Number of views.
                        - likeCount (int | str): Number of likes.
                        - commentCount (int | str): Number of comments.

            Raises:
                google.auth.exceptions.GoogleAuthError: On authentication failure.
                HttpError: If an API call to YouTube fails.
            """
            trace_id = context.deps.get("trace_id")
            span = self.langfuse_client.span(
                trace_id=trace_id,
                name="youtube",
                input={},
                metadata={"tool": "get_youtube_content"}
            )
            result = await get_youtube_videos()
            span.end(output=result)
            return result
        
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