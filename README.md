# Portfolio Chatbot

A multi-agent RAG system that represents Sanath Vijay Haritsa professionally during conversations with recruiters, hiring managers, and employers.

## Architecture

The system consists of three coordinated agents:

1. **Orchestrator Agent**: Interprets user intent and routes requests
2. **Professional Info Agent**: Retrieves evidence from resume, profile, and project documents
3. **Final Presentation Agent**: Transforms evidence into persuasive responses

## Project Structure

```
portfolio_chatbot/
├── src/
│   ├── agents/              # AI agent implementations
│   ├── vector_store/        # Document processing and storage
│   ├── models/              # Pydantic schemas
│   ├── tools/               # RAG utilities
│   └── config.py            # Configuration management
├── data/                    # Raw documents (PDFs, markdown)
├── configs/                 # Configuration files
├── prompts/                 # System prompts and profile documents
├── main.py                  # Entry point
└── requirements.txt
```

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables:**
   Create a `.env` file with:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

3. **Start Qdrant (using Docker):**
   ```bash
   docker run -p 6333:6333 qdrant/qdrant
   ```

4. **Prepare your data:**
   - Place documents in `data/` directory
   - Configure metadata in `configs/data_config.json`:
     ```json
     {
       "project1.pdf": {
         "tools_used": ["Python", "FastAPI"],
         "skills": ["API Development", "Testing"]
       }
     }
     ```

5. **Add system prompts:**
   Place your agent instructions in `prompts/`:
   - `orchestrator.txt`
   - `professional_info.txt`
   - `final_presentation.txt`
   - `Sanath Vijay Haritsa - CV.tex`
   - `Sanath Vijay Haritsa - About Me.md`

## Usage

### Basic Usage

```python
from main import PortfolioChatbot
import asyncio

async def main():
    chatbot = PortfolioChatbot()
    response = await chatbot.process_query("What are his technical skills?")
    print(response)

asyncio.run(main())
```

### Running Example Queries

```bash
python main.py
```

### Custom Configuration

```python
from src.config import AppConfig, VectorStoreConfig, AgentConfig
from main import PortfolioChatbot

config = AppConfig(
    vector_store=VectorStoreConfig(
        collection_name="my_collection",
        embedding_model="text-embedding-3-large"
    ),
    agent=AgentConfig(
        orchestrator_model="gpt-4",
        professional_info_model="gpt-4"
    )
)

chatbot = PortfolioChatbot(config)
```

## Key Features

- **Evidence-Based Responses**: All claims are grounded in actual documents
- **Smart Routing**: Orchestrator determines the best agent for each query
- **Semantic Search**: Vector store with reranking for relevant document retrieval
- **Modular Design**: Easy to extend with new agents or data sources
- **Type Safety**: Full type hints and Pydantic validation

## Configuration

All configuration is centralized in `src/config.py`:

- **VectorStoreConfig**: Qdrant connection and embedding settings
- **ProcessorConfig**: Document processing parameters
- **AgentConfig**: Model selection and prompt paths
- **RetrievalConfig**: Search and reranking parameters

## Development

### Adding a New Agent

1. Create agent class in `src/agents/`
2. Define output schema in `src/models/schemas.py`
3. Add agent to orchestrator routing logic
4. Register in `src/agents/__init__.py`

### Adding New Document Types

Extend `FileProcessor._to_markdown()` in `src/vector_store/processor.py`

### Running Tests

```bash
pytest tests/
```

## License

MIT