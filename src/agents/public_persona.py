"""Public persona retrieval agent."""

from __future__ import annotations

from pydantic_ai import Agent, ModelSettings
from langfuse import observe, get_client

from src.config import AgentConfig
from src.models.schemas import PublicArtifacts
from src.tools.social_media_retrieval import get_instagram_posts, get_youtube_videos

class PublicPersonaAgent:
    def __init__(self, config: AgentConfig) -> None:
        """Initialize the public persona agent.
        
        Args:
            config: Agent configuration.
        """
        self.config = config.public_persona
        self.langfuse_client = get_client()
        self.instructions = self._load_instructions()
        self.agent: Agent[None, PublicArtifacts] = self._create_agent()
        self._register_tools()
        
    def _create_agent(self) -> Agent[None, PublicArtifacts]:
        """Create the PydanticAI agent.
        
        Returns:
            Configured Agent instance.
        """
        return Agent(
            self.config.model,
            output_type=PublicArtifacts,
            model_settings=ModelSettings(temperature=0, parallel_tool_calls=True)
        )
    
    @observe(name="public_persona_agent", capture_input=True, capture_output=True)
    async def run(self, user_prompt: str) -> PublicArtifacts:
        """Run the agent with a user prompt.
        
        Args:
            user_prompt: The user's prompt including task and constraints.
            
        Returns:
            Public artifacts with account metadata and retrieved artifacts.
        """
        self.langfuse_client.update_current_span(metadata={"public_persona_instructions": self.instructions, "user_prompt": user_prompt})
        result = await self.agent.run(
            instructions=self.instructions,
            user_prompt=user_prompt,
        )
        return result.output
        
    def _register_tools(self):
        """Register tools with the agent."""
        
        @self.agent.tool_plain
        @observe(name="instagram", capture_input=True, capture_output=True)
        async def get_instagram_content() -> str:
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
            account_info_endpoint = self.config.tool_config.instagram_account_info_endpoint
            media_endpoint = self.config.tool_config.instagram_media_endpoint
            account_info_fields = self.config.tool_config.instagram_account_info_fields
            media_fields = self.config.tool_config.instagram_media_fields
            return await get_instagram_posts(account_info_endpoint, media_endpoint, account_info_fields, media_fields)
        
        @self.agent.tool_plain
        @observe(name="youtube", capture_input=True, capture_output=True)
        async def get_youtube_content():
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
            return await get_youtube_videos()
        
    def _load_instructions(self) -> str:
        """Load system instructions from langfuse or file.
        
        Returns:
            System instructions text.
        """
        try:
            return self.langfuse_client.get_prompt(self.config.langfuse_key).prompt
        except:
            return self.config.instructions_path.read_text(encoding="utf-8")