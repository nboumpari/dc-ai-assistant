# DC AI Assistant

A chat assistant for the control tower of a fictional FMCG manufacturer (one plant, three distribution centres in Greece). A supervisor sends each question to one of two agents:

| | Data agent | Rules agent |
|---|---|---|
| Answers | How many, which, when, where | What the rule says, what is allowed, what is required |
| Source | Synthetic 2025 warehouse database: orders, transfers, returns, shipments, pallets and SSCCs, inventory, dock appointments (SQLite, 18 tables) | Real public documents: GS1 General Specifications and Logistic Label Guideline, ADR 2025 (limited quantities, aerosols), CMR, Regulation (EC) 561/2006, EU GPSR, HSE HSG76 |
| Technique | Text-to-SQL with a semantic layer and read-only guardrails | Advanced RAG: query reformulation, vector search, cross-encoder reranking, cited answers |

```
question -> supervisor (router) --> data agent   (LLM writes and runs SQL)
                                +--> rules agent  (reformulate -> retrieve -> rerank -> answer with citations)
                                +--> fallback     (what the assistant can help with)
```

Everything runs locally in Docker: the chat page, the API, the vector database (Qdrant) and the local embedding and reranking models (on the CPU). The LLM is OpenAI's gpt-5.4-mini; evaluations are tracked in LangSmith.

## Results

Every part was improved on a dev split and scored once, at the end, on a test split it had never seen. Each question is run 3 times.

| Part | Metric | Dev | Test | Details |
|---|---|---|---|---|
| Data agent (prompt v4) | answer matches the expected query result | 96.8% | 94.4% | [CHANGELOG](src/dc_ai_assistant/data_assistant/prompts/CHANGELOG.md) |
| Rules agent (prompt v2) | correct answer (LLM judge: Gemini) or correct not-found / decline | 94.6% | 80.4% | [CHANGELOG](src/dc_ai_assistant/rules_assistant/prompts/CHANGELOG.md) |
| Supervisor router (prompt v3) | routed to the right place | 100% | 100% | [CHANGELOG](src/dc_ai_assistant/supervisor/prompts/CHANGELOG.md) |

The rules agent's retriever was chosen in a separate benchmark of 45 configurations (3 chunking strategies, BM25 / vector / hybrid search, 2 embedding models, 2 rerankers or none), scored on finding the evidence passages: [retrieval results](evals/rules/retrieval_results.md).

## Setup

You need Docker Desktop and these keys in a `.env` file (copy `.env.example`):

- `OPENAI_API_KEY`: the agents' LLM
- `LANGSMITH_API_KEY`: tracing and evaluation
- `HF_TOKEN`: a Hugging Face read token, to download the gated `google/embeddinggemma-300m` (accept its licence on huggingface.co first)
- `GEMINI_API_KEY`: only for evaluating the rules agent (the LLM judge)

Then, from the project root:

```powershell
docker compose build
```

Generate the warehouse database (deterministic, about 1.4 million rows):

```powershell
docker compose run --rm app python -m scripts.data_generator.generate
```

Get the rules documents. Most are downloaded from their official sources; UNECE blocks automated downloads, so save ADR 2025 Volume I and II (English PDFs) by hand as `data/documents/raw/adr_2025_vol1.pdf` and `adr_2025_vol2.pdf` (the download script prints the link).

```powershell
docker compose run --rm app python -m scripts.rules_corpus.download
```

```powershell
docker compose run --rm app python -m scripts.rules_corpus.extract
```

```powershell
docker compose run --rm app python -m scripts.rules_corpus.chunk
```

Start Qdrant, download the local models and build the vector index:

```powershell
docker compose up -d qdrant
```

```powershell
docker compose run --rm app python -m scripts.rules_corpus.check_models
```

```powershell
docker compose run --rm app python -m scripts.rules_corpus.index
```

The documents, the database, the index and the models stay on your machine and are mounted into the containers; they are never copied into an image (the documents may not be redistributed).

## Run

```powershell
docker compose up -d --build
```

Open http://localhost:8080. The answer streams each step of the graph while it works ("Searching the documents", "Running SQL", ...), rules answers cite their sources, and each citation opens the passage it refers to.

For development (mounted code, auto-reload, the page on http://localhost:5173):

```powershell
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d
```

## Evaluate

```powershell
docker compose run --rm app python -m evals.rules.run_retrieval_grid
```

```powershell
docker compose run --rm app python -m evals.rules.run_rag_eval --label v2
```

```powershell
uv run python -m evals.rules.judge --label v2
```

```powershell
uv run python -m evals.supervisor.run_router_eval --label v3
```

```powershell
uv run python -m evals.run_eval --label v4
```

Unit tests (no API calls):

```powershell
uv run pytest
```

## Project layout

```
src/dc_ai_assistant/
  data_assistant/     data agent: text-to-SQL, semantic layer, SQL guardrails
  rules_assistant/    rules agent: retrieval, local models, Qdrant client, RAG graph
  supervisor/         router and the graph that joins both agents
  api.py              FastAPI with streamed answers (Server-Sent Events)
frontend/             the chat page (React + Vite, served by nginx)
scripts/
  data_generator/     builds the synthetic warehouse database
  rules_corpus/       downloads, extracts, chunks and indexes the documents
evals/                golden sets and evaluation runners for every part
tests/
data/                 database, documents, index and saved runs (not committed)
```
