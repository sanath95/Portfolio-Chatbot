from dataclasses import dataclass
import asyncio

from langchain_qdrant import QdrantVectorStore
from pydantic_ai import Agent, ModelResponse, RunContext, TextPart

from utils.vector_store import VectorStore

from dotenv import load_dotenv
load_dotenv()

    

vs = VectorStore()
qdrant_store = vs.get_vector_store()


@dataclass
class Deps:
    resume: str
    about_me: str
    instructions_template: str
    vector_store: QdrantVectorStore

with open("./prompts/Sanath Vijay Haritsa - CV.tex", encoding="utf8") as f:
    resume = f.read()
with open("./prompts/Sanath Vijay Haritsa - About Me.md", encoding="utf8") as f:
    about_me = f.read()
with open("./prompts/system_instructions_template.txt", encoding="utf8") as f:
    instructions_template = f.read()


deps = Deps(resume=resume, about_me=about_me, instructions_template=instructions_template, vector_store=qdrant_store)

agent = Agent('openai:gpt-5', deps_type=Deps)


@agent.instructions
def get_system_instructions(ctx: RunContext[Deps]) -> str:
    system_instructions = f"{ctx.deps.instructions_template}\n---\n## Resume\n{ctx.deps.resume}\n---\n## About Sanath\n{ctx.deps.about_me}"
    print(system_instructions)
    return system_instructions


async def run_agent(prompt):
    resp = await agent.run(prompt, deps=deps)
    print(resp)

if __name__ == '__main__':
    q = "what are the major contributions of the thesis?"
    asyncio.run(run_agent(q))