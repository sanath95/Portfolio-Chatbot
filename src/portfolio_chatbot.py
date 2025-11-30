from dataclasses import dataclass
import asyncio
from typing import Dict

from langchain_qdrant import QdrantVectorStore
from pydantic_ai import Agent, RunContext
from sentence_transformers import CrossEncoder

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
    return system_instructions

@agent.tool
async def retrieve(context: RunContext[Deps], search_query: str) -> Dict[str, float]:
    """Retrieve documentation sections based on a search query.

    Args:
        context: The call context.
        search_query: The search query.
    """
    retriever = context.deps.vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 10})
    results = await retriever.ainvoke(search_query)

    input_for_cross_encoder = [(search_query, r.page_content) for r in results]
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    scores = model.predict(input_for_cross_encoder)
    result = {s: float(v) for s, v in zip([r.page_content for r in results], scores) if v > 0}
    return result

async def run_agent(prompt):
    resp = await agent.run(prompt, deps=deps)
    print(resp)

if __name__ == '__main__':
    q = "what are the major contributions of the thesis?"
    asyncio.run(run_agent(q))