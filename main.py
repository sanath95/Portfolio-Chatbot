"""Main entry point for the portfolio chatbot system."""

from __future__ import annotations

from typing import AsyncGenerator, Any
from random import choice
import base64
import gradio as gr
from string import Template

from src.agent_runner import AgentRunner
from src.config import GradioConfig
from src.models.schemas import AgentSource

ChatHistory = list[dict[str, str]]
GradioOutputs = tuple[gr.Textbox | dict[Any, Any], ChatHistory | dict[Any, Any], ChatHistory | dict[Any, Any]]


class ChatbotUI:
    """Manages the Gradio UI for the portfolio chatbot."""

    def __init__(self, agent: AgentRunner, config: GradioConfig | None = None) -> None:
        """Initialize the chatbot UI.

        Args:
            config: Gradio configuration. Uses defaults if not provided.
            agent: The agent runner instance for processing queries.
        """
        self.config = config or GradioConfig()
        self.agent = agent

    async def stream_response(
        self,
        prompt: str,
        chatbot: ChatHistory,
        conversation: ChatHistory,
        request: gr.Request,
    ) -> AsyncGenerator[GradioOutputs, None]:
        """Stream chatbot responses to the UI.

        Args:
            prompt: User input message.
            chatbot: Chat history for display in the UI.
            conversation: Full conversation history for the agent.

        Yields:
            Tuple of (textbox state, chatbot history, conversation history).
        """
        session_id = request.session_hash
        if not prompt or not prompt.strip():
            chatbot.append({"role": "assistant", "content": "Please enter a question."})
            yield gr.Textbox(interactive=True, value=""), chatbot, conversation
            return

        # Show user message
        chatbot.append({"role": "user", "content": prompt})
        yield gr.Textbox(interactive=False, value=""), chatbot, gr.skip()
        
        # Disable input and update UI
        thinking_messages = [
            "🤔 Let me think about that...",
            "💭 Gathering information...",
            "📚 Checking my knowledge base...",
            "⚡ Processing your question..."
        ]
        chatbot.append({"role": "assistant", "content": choice(thinking_messages)})
        yield gr.skip(), chatbot, gr.skip()
        
        # Stream response chunks
        chatbot[-1]["content"] = ""
        conversation.append({"role": "user", "content": prompt})
        async for event in self.agent.process_query(prompt, conversation, str(session_id)):
            match event.from_:
                case AgentSource.ORCHESTRATOR | AgentSource.PROFESSIONAL_INFO | AgentSource.PUBLIC_PERSONA:
                    conversation.append({"role": "assistant", "content": event.output})
                case AgentSource.FINAL_PRESENTATION:
                    chatbot[-1]["content"] += event.output
                    yield gr.skip(), chatbot, gr.skip()

        conversation.append({"role": "assistant", "content": chatbot[-1]["content"]})
        
        # Re-enable input
        yield gr.Textbox(interactive=True), gr.skip(), conversation

    def build_interface(self) -> gr.Blocks:
        """Build and configure the Gradio interface.

        Returns:
            Configured Gradio Blocks interface.
        """
        with gr.Blocks(title="Sanath's Portfolio") as demo:
            self._add_header()
            
            conversation = gr.State([])
            chatbot = self._create_chatbot()
            prompt = self._create_input()
            self._add_examples(prompt)
            self._add_footer()
            
            # Connect event handlers
            prompt.submit(
                self.stream_response,
                inputs=[prompt, chatbot, conversation],
                outputs=[prompt, chatbot, conversation],
                show_progress="full"
            )


        return demo

    def _add_header(self) -> None:
        """Render the header section of the UI.

        This includes the profile image, title, and subtitle displayed
        at the top of the chatbot interface.
        """
        image_path = self.config.image_path
        if image_path.exists():
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode()
                image_src = f"data:image/png;base64,{image_data}"
        else:
            image_src = ""
        template = Template(self.config.header_html_path.read_text())
        html = template.safe_substitute(image_src=image_src)
        gr.HTML(html)

    def _create_chatbot(self) -> gr.Chatbot:
        """Create the chatbot display component.

        Returns:
            Configured Gradio Chatbot component.
        """
        intro = \
            """
            Welcome 👋

            Responses are grounded in verified sources, delivered through a multi-agent interactive chatbot, and transparently traced.

            You can ask about:
            - Technical skills
            - Project details
            - Career background and professional focus
            - Hobbies and interests
            """
        return gr.Chatbot(
            [
                {"role": "assistant", "content": intro}
            ],
            label="Sanath’s Portfolio Chatbot",
            avatar_images=(None, self.config.image_path)
        )

    @staticmethod
    def _create_input() -> gr.Textbox:
        """Create the user input textbox.

        Returns:
            Configured Gradio Textbox component.
        """
        with gr.Row():
            return gr.Textbox(
                lines=1,
                show_label=False,
                placeholder="Ask something about Sanath",
                submit_btn=True
            )
            
    @staticmethod
    def _add_examples(input_textbox: gr.Textbox) -> None:
        """Add example prompts below the input box.

        Args:
            input_textbox: The textbox component that will be populated
                when an example is selected.
        """
        gr.Examples(
            examples=[
                "Introduce Sanath in a few short sentences.",
                "What was his final grade in his master's program?",
                "Summarize his generative AI work and list the tools used.",
                "Show me a card trick.",
                "How can I contact Sanath?"
            ],
            inputs=input_textbox,
        )
        
    def _add_footer(self) -> None:
        """Render the footer section of the UI.

        The footer contains metadata about the chatbot as well as
        external links (GitHub and LinkedIn).
        """
        footer_html = self.config.footer_html_path.read_text(encoding="utf-8")
        gr.HTML(footer_html)


if __name__ == "__main__":
    """Launch the portfolio chatbot."""
    agent = AgentRunner()
    ui = ChatbotUI(agent)
    demo = ui.build_interface()
    demo.launch()