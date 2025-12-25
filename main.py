"""Main entry point for the portfolio chatbot system."""

from __future__ import annotations

from typing import AsyncGenerator, Any
from random import choice
import base64
from pathlib import Path
import gradio as gr

from src.agent_runner import AgentRunner


ChatHistory = list[dict[str, str]]
GradioOutputs = tuple[gr.Textbox | dict[Any, Any], ChatHistory | dict[Any, Any], ChatHistory | dict[Any, Any]]


class ChatbotUI:
    """Manages the Gradio UI for the portfolio chatbot."""

    def __init__(self, agent: AgentRunner) -> None:
        """Initialize the chatbot UI.

        Args:
            agent: The agent runner instance for processing queries.
        """
        self.agent = agent

    async def stream_response(
        self,
        prompt: str,
        chatbot: ChatHistory,
        conversation: ChatHistory,
    ) -> AsyncGenerator[GradioOutputs, None]:
        """Stream chatbot responses to the UI.

        Args:
            prompt: User input message.
            chatbot: Chat history for display in the UI.
            conversation: Full conversation history for the agent.

        Yields:
            Tuple of (textbox state, chatbot history, conversation history).
        """
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
        async for chunk in self.agent.process_query(prompt, conversation):
            chatbot[-1]["content"] += chunk
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

    @staticmethod
    def _add_header() -> None:
        """Render the header section of the UI.

        This includes the profile image, title, and subtitle displayed
        at the top of the chatbot interface.
        """
        # Load and encode the local image
        image_path = Path("assets/sanath_vijay_haritsa.png")
        if image_path.exists():
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode()
                image_src = f"data:image/png;base64,{image_data}"
        else:
            image_src = ""
        gr.HTML(
            f"""
            <div style="display: flex; justify-content: center; align-items: center; gap: 2rem; padding: 1rem; width: 100%">
                <img src={image_src} style="width:120px;border-radius:50%" alt="Sanath Vijay Haritsa"/>
                <div>
                    <h1 style="margin:0;font-size:1.8rem">Sanath’s Portfolio Chatbot</h1>
                    <h3 style="margin:0;color:#9aa0a6;font-size:0.95rem">
                        Ask about my work, projects, and research.
                    </h3>
                </div>
            </div>
            """
        )

    @staticmethod
    def _create_chatbot() -> gr.Chatbot:
        """Create the chatbot display component.

        Returns:
            Configured Gradio Chatbot component.
        """
        return gr.Chatbot(
            [
                {"role": "assistant", "content": "I am happy to provide you that report and plot."}
            ],
            label="Sanath’s Portfolio Chatbot",
            avatar_images=(None, "assets/sanath_vijay_haritsa.png")
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
                "Introduce Sanath in 2 lines.",
                "Tell me about his work with LLMs.",
                "What are the contributions of the master's thesis?",
            ],
            inputs=input_textbox,
        )
        
    @staticmethod
    def _add_footer() -> None:
        """Render the footer section of the UI.

        The footer contains metadata about the chatbot as well as
        external links (GitHub and LinkedIn).
        """
        gr.HTML(
            """
            <div style="text-align:center;color:#777;font-size:0.8rem;margin-top:1rem">
                <div>
                    Multi-agent chatbot · Factually grounded responses · Built with Gradio · Traced on Langfuse
                </div>
                <div style="margin-top:0.3rem">
                    <a href="https://github.com/sanath95" target="_blank" style="color:#8ab4f8;text-decoration:none">
                        GitHub
                    </a>
                    &nbsp;·&nbsp;
                    <a href="https://www.linkedin.com/in/sanath-haritsa-974686315/" target="_blank" style="color:#8ab4f8;text-decoration:none">
                        LinkedIn
                    </a>
                </div>
            </div>
            """
        )


if __name__ == "__main__":
    """Launch the portfolio chatbot."""
    agent = AgentRunner()
    ui = ChatbotUI(agent)
    demo = ui.build_interface()
    demo.launch()