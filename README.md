# Sanath's Portfolio Chatbot

## Introduction

**Problem Statement:** Recruiters and hiring managers often need to sift through multiple documents such as resumes, cover letters, academic transcripts, and certificates to understand a candidate’s profile and assess role fit. This process is inherently time-consuming, fragmented, and inefficient, especially when the relevant information is spread across several static files.

To address this problem, I built an agentic portfolio chatbot that transforms the traditional candidate evaluation process into a natural, conversational experience. Instead of manually scanning documents, recruiters can interact with a chatbot to learn about my technical expertise, soft skills, professional experience, academic background, and interests outside of work through targeted questions in plain language.

Beyond solving a real hiring workflow problem, this project also serves as a practical demonstration of my data science and AI skills. The system implements a full end-to-end Retrieval-Augmented Generation (RAG) pipeline, including document ingestion, intelligent retrieval, grounded response generation, and a production-ready user interface. It further incorporates evaluation, tracing, observability, and cloud deployment, showcasing how modern LLM-based systems can be built, monitored, and scaled in real-world settings.

In essence, this project reframes a static portfolio into an interactive AI system, simultaneously improving the recruiter experience and demonstrating applied AI engineering competence.

## Architecture

![chatbot architecture](./assets/architecture.drawio.png)

The portfolio chatbot is deployed on Google Cloud Platform and follows an agentic, modular architecture designed for clarity, traceability, and controlled information flow.

User interactions begin through a Gradio-based chat application, which handles query input, response streaming, and UI-level interaction. Each user query is forwarded to a central Orchestrator Agent, which analyzes the intent of the conversation and determines which specialized agents should be invoked. The Orchestrator Agent also performs lightweight context checks before invoking downstream agents. If sufficient evidence is already available within the ongoing conversation or previously retrieved context, the orchestrator can bypass additional retrieval and route the request directly to the Final Presentation Agent. Similarly, if the orchestrator determines that a query is unrelated to me or my work, it routes the request directly to the final agent with an enforced refusal policy, ensuring consistent, safe, and scope-limited responses without unnecessary agent execution.

The Professional Info Agent is responsible for retrieving evidence related to my technical skills, academic background, and project experience. It accesses embedded documents stored in a Qdrant Cloud vector store for retrieval-augmented generation and optionally enriches responses using metadata fetched via the GitHub API. Supporting documents and artifacts, like resume and academic transcripts, are stored in a cloud storage bucket.

In parallel, the Public Persona Agent handles questions related to interests, public-facing activities, and non-professional context. This agent integrates with external services such as the Instagram API and YouTube API to fetch relevant public data.

Both specialized agents return grounded evidence to the Final Presentation Agent, which is responsible for synthesizing a coherent, user-facing response. This final agent ensures that answers are factual, consistent with retrieved evidence, and aligned with the system’s safety and refusal policies.

The generated response is then streamed back to the user via the Gradio interface. Throughout this flow, the architecture supports observability and traceability, allowing each step of the interaction to be monitored, evaluated, and debugged without tightly coupling components.

Overall, the design emphasizes separation of concerns, controlled agent hand-offs, and production readiness, while remaining flexible enough to extend with additional agents or data sources in the future.

## Key Features

* **Agentic architecture**
  An orchestrator agent dynamically routes queries to specialized downstream agents such as professional profile and public persona agents, before handing off to a final presentation agent. This enables structured reasoning, clear responsibility separation, and controllable behavior.

* **End-to-end Retrieval-Augmented Generation (RAG)**
  Academic project reports and technical documentation are processed, embedded using OpenAI models, and stored in a Qdrant vector store. Responses are grounded in retrieved evidence and reranked to ensure factual accuracy and relevance.

* **Multi-source knowledge ingestion**
  Supports structured and unstructured data from PDFs, Markdown files, GitHub repositories, and optionally social platforms such as Instagram and YouTube. This allows a unified conversational view over heterogeneous data sources.

* **Factually grounded and safe responses**
  The system is explicitly instructed to avoid hallucination, overselling, or inventing experience. Answers are strictly limited to retrieved context, with clear acknowledgment when information is missing. A refusal policy is enforced for queries unrelated to the candidate or their work.

* **Short-term conversational memory**
  Maintains session-level context to support coherent multi-turn conversations without leaking information across users or sessions.

* **Observability and tracing**
  Full tracing, span tracking, prompt management, and user feedback capture are implemented using Langfuse. Thumbs up and thumbs down signals are collected directly from the chat UI to support quality monitoring.

* **Evaluation framework**
  Retrieval quality is assessed using an LLM-as-a-Judge approach aligned with custom quality criteria. Manual regression testing is performed at each development step to ensure consistent system behavior.

* **Interactive user interface**
  A Gradio-based front end provides response streaming, example prompts, and personal branding, enabling a smooth and recruiter-friendly interaction experience.

* **Production-ready deployment**
  Fully containerized using Docker and deployed on Google Cloud Run, demonstrating real-world readiness with scalable infrastructure and clean separation between build and runtime stages.

> NOTE: This branch focuses on local deployment. For code changes, containerization and cloud deployment, refer to [deploy/gcp](https://github.com/sanath95/Portfolio-Chatbot/tree/deploy/gcp) branch.

## Folder Structure

```
📦Portfolio-Chatbot
 ┣ 📂configs                    # configuration for data files used for indexing
 ┣ 📂data                       # data files (project reports and documentation)
 ┣ 📂Notebooks                  # experiment notebooks
 ┣ 📂prompts                    # system instructions for all the agents (resume, academic transcripts and other narrative documents go here)
 ┣ 📂src                        # heart of the application
 ┃ ┣ 📂agents
 ┃ ┃ ┣ 📜final_presentation.py
 ┃ ┃ ┣ 📜orchestrator.py
 ┃ ┃ ┣ 📜professional_info.py
 ┃ ┃ ┣ 📜public_persona.py
 ┃ ┣ 📂models
 ┃ ┃ ┣ 📜schemas.py
 ┃ ┣ 📂tools
 ┃ ┃ ┣ 📜github_repos.py
 ┃ ┃ ┣ 📜retrieval.py
 ┃ ┃ ┣ 📜social_media_retrieval.py
 ┃ ┣ 📂vector_store
 ┃ ┃ ┣ 📜processor.py
 ┃ ┃ ┣ 📜store.py
 ┃ ┣ 📜agent_runner.py
 ┃ ┣ 📜config.py                # models to use, file paths, and other configurations
 ┣ 📂static                     # static files for the UI
 ┃ ┣ 📜footer.html
 ┃ ┣ 📜header.html
 ┃ ┗ 📜sanath_vijay_haritsa.png
 ┣ 📜.env
 ┣ 📜.gitignore
 ┣ 📜.python-version
 ┣ 📜LICENSE
 ┣ 📜main.py                    # entry point for the gradio chatbot
 ┣ 📜pyproject.toml
 ┣ 📜README.md
 ┗ 📜uv.lock
```

## Prerequisites

- Python 3.11
- The packages are maintained with UV. [Install UV](https://docs.astral.sh/uv/getting-started/installation/) and then run `uv sync` to install dependencies and create a virtual environment.
- Qdrant running locally on docker: [Get Started with Qdrant Locally](https://qdrant.tech/documentation/quickstart/)
- Langfuse running locally on docker: [Self-host Langfuse with Docker Compose](https://langfuse.com/self-hosting/deployment/docker-compose)

## Environment Variables

| Variable | Purpose |
| --- | --- |
| `OPENAI_API_KEY` | Required for all OpenAI model and embedding calls. |
| `QDRANT_URL`, `QDRANT_PORT` | URL and port where qdrant is running |
| `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST` | Enable tracing and prompt retrieval via Langfuse. |
| `GITHUB_TOKEN` | Fetches repositories for project linking. |
| `INSTAGRAM_ACCESS_TOKEN` | Optional, enables Instagram media retrieval for the public persona agent. |
| `YOUTUBE_ACCESS_TOKEN`, `YOUTUBE_REFRESH_TOKEN`, `YOUTUBE_CLIENT_ID`, `YOUTUBE_CLIENT_SECRET`, `YOUTUBE_TOKEN_URI` | Optional, enable YouTube channel/video retrieval. |

## Running the app

```bash
uv run main.py
```

## 𝗧𝗼𝗼𝗹𝘀
Docling, EasyOCR, PydanticAI, LangChain, OpenAI API, Qdrant Vector Store, Gradio, Langfuse, Docker, and Google Cloud Platform.

## 𝗙𝘂𝘁𝘂𝗿𝗲 𝘀𝗰𝗼𝗽𝗲
1. Automated offline & online evaluation
2. Long-term memory for a continuously evolving system

## Try It Out
* [Click here](https://chatbot-566360013332.europe-west3.run.app/) to have a chat.
* Responses take ~30–60 seconds — please be patient
* UI is not responsive yet; best viewed on a laptop

## Deployment

The deployment setup introduces several changes compared to local development to support scalability, security, and cloud-native operation:

1. **Qdrant Cloud integration**
   Qdrant is deployed as a managed cloud service instead of running locally. This requires a `QDRANT_API_KEY` rather than a local `QDRANT_PORT`.

2. **Langfuse deployment on Google Cloud Platform**
   Langfuse is deployed on GCP and backed by **Cloud SQL (PostgreSQL)** for trace and metadata storage.

3. **Minimal Langfuse setup**
   The deployment uses the Langfuse v2 Docker image along with the Python SDK (v2.60.10) to keep the observability stack lightweight.

4. **Cloud-based document storage**
   Supporting documents such as resumes and academic transcripts are stored and read from **Google Cloud Storage buckets**.

5. **Cloud-specific utilities**
   Additional utility modules are included for interacting with Google Cloud Storage and for initializing the Langfuse client in a cloud environment.

6. **Separated indexing pipeline**
   A dedicated `index.py` entry point is introduced to decouple document indexing from chatbot runtime deployment.

7. **Secure build-time secrets for indexing**
   Environment variables required for indexing are injected using **Docker BuildKit secrets** to avoid leaking credentials into image layers.

8. **Runtime secrets management**
   Secrets required by the chatbot at runtime are sourced from **Google Secret Manager**.

9. **Indexing documents into Qdrant Cloud**
   Run the following command to build the indexing image and populate the Qdrant Cloud vector store:

   ```bash
   docker build \
     --secret id=QDRANT_URL,env=QDRANT_URL \
     --secret id=QDRANT_API_KEY,env=QDRANT_API_KEY \
     --secret id=OPENAI_API_KEY,env=OPENAI_API_KEY \
     --no-cache \
     --target indexer .
   ```

10. **Building the chatbot runtime image**
    After indexing is complete, build the chatbot runtime image:

    ```bash
    docker build --no-cache --target runtime -t portfolio-chatbot:latest .
    ```

---