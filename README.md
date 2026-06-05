# RAG-as-a-Service

RAG-as-a-Service is a multi-tenant Retrieval-Augmented Generation API built with FastAPI. Each tenant can register with the service, connect a Qdrant collection, upload or provide documents, build an isolated index, and query that knowledge base through standard RAG or agentic ReAct-style reasoning.

The project is designed for AWS-backed operation, with DynamoDB for tenant and memory state, S3 for document storage, optional Lambda-based index builds, CloudWatch metrics, Qdrant for vector search, and Groq Llama 3.3 70B for generation.

## What This Codebase Does

- Registers tenant organizations and stores tenant metadata in DynamoDB.
- Creates managed S3 storage and scoped IAM upload credentials, or accepts tenant-owned S3 buckets.
- Downloads tenant documents, extracts PDF and web content, chunks it, embeds it, and stores it in Qdrant.
- Runs hybrid retrieval with dense Qdrant search and sparse BM25 search.
- Fuses results with Reciprocal Rank Fusion, reranks with a cross-encoder, and generates cited answers with Groq.
- Supports agentic queries using a ReAct loop and tool registry.
- Supports tenant-specific custom tools.
- Stores conversation memory and learned navigation patterns in DynamoDB.
- Exposes operational metrics through CloudWatch.
- Includes endpoint tests under `tests/`.

## Architecture

```text
Client
  |
  v
FastAPI app
  |
  +-- Tenant auth and tenant metadata
  |     +-- DynamoDB: rag-tenants
  |
  +-- Document storage
  |     +-- S3: managed service bucket or tenant-owned bucket
  |     +-- IAM: scoped upload users for managed storage
  |
  +-- Index building
  |     +-- Local thread mode or AWS Lambda
  |     +-- PDFLoader / WebLoader / Chunker
  |     +-- SentenceTransformer embeddings
  |     +-- Qdrant collection per tenant config
  |
  +-- Query pipeline
  |     +-- Dense retrieval from Qdrant
  |     +-- Sparse retrieval with BM25
  |     +-- Reciprocal Rank Fusion
  |     +-- Cross-encoder reranking with optional diversity
  |     +-- Groq Llama 3.3 70B answer generation
  |
  +-- Agentic pipeline
  |     +-- ReActAgent
  |     +-- Built-in and custom tools
  |     +-- Conversation and pattern memory
  |
  +-- Monitoring
        +-- CloudWatch metrics and logs
```

## Project Structure

```text
api/
  main.py                         FastAPI app, lifespan, router registration
  auth.py                         X-API-Key authentication and ready-state checks
  app_state.py                    Shared model instances for endpoint modules
  pipeline_helper.py              Tenant pipeline cache
  endpoints/
    tenant_endpoints.py           Tenant registration, status, credentials, metrics
    document_endpoints.py         Upload and index-build endpoints
    query_endpoints.py            Standard and agentic query endpoints
    tool_endpoints.py             Custom tool registration and listing
    memory_endpoints.py           Memory stats, clear, export, conversations

agent/
  react_agent.py                  ReAct loop for multi-step retrieval
  prompts.py                      Agent reasoning and answer prompts
  tools/                          Built-in tools, custom tool wrapper, registry, store
  memory/                         Conversation memory, pattern memory, DynamoDB store

ingestion/
  base_loader.py                  Document model and loader base class
  chunker.py                      Text chunking
  pdf_loader.py                   PDF extraction with PyMuPDF
  web_loader.py                   URL scraping with requests, cloudscraper, Selenium fallback

retrieval/
  sparse_retriever.py             BM25 sparse retriever
  hybrid.py                       Reciprocal Rank Fusion

reranker/
  cross_encoder.py                Cross-encoder reranking and diversity selection

generation/
  generator.py                    Groq-backed answer generation

vectorstore/
  qdrant_store.py                 Qdrant collection, upsert, search, stats

mlops/
  pipeline.py                     Tenant orchestration, status updates, metrics
  storage_manager.py              S3 and IAM management
  build_index.py                  Tenant document download and index building

lambda/
  build_index.py                  Lambda handler for async builds

schemas/
  tenant.py                       Tenant and query Pydantic schemas
  tool.py                         Tool schemas
  memory.py                       Memory schemas

tests/
  conftest.py                     Endpoint test fixtures and local stubs
  root_tests/                     Root and health endpoint tests
  tenant_tests/                   Tenant endpoint tests
  document_tests/                 Document endpoint tests
  query_tests/                    Query endpoint tests
  tool_tests/                     Tool endpoint tests
  memory_tests/                   Memory endpoint tests

scripts/
  setup_memory_tables.py          DynamoDB setup for memory tables
  setup_custom_tools_table.py     DynamoDB setup for custom tools
```

## Runtime Components

### Tenant Registry

Tenant records are stored in DynamoDB through `MLOpsPipeline`. The tenant record includes:

- `tenant_id`
- API key
- company and contact information
- storage information
- Qdrant config
- status
- build statistics

Tenant API keys must start with `rsk_`.

### Document Storage

The service supports two storage modes:

| Mode | Description |
| --- | --- |
| `managed` | The service creates/uses a managed S3 bucket and scoped IAM credentials for tenant uploads. |
| `own_s3` | The tenant provides an existing S3 bucket that the service can access. |

### Index Building

Index builds are triggered through `POST /api/tenant/{tenant_id}/build`.

Depending on `USE_LAMBDA`, the build is handled by:

- AWS Lambda (`USE_LAMBDA=true`)
- a local background thread (`USE_LAMBDA=false`)

The build process:

1. Loads the tenant record from DynamoDB.
2. Lists files in the tenant's S3 location.
3. Downloads files to a temporary directory.
4. Loads `.pdf` files with `PDFLoader`.
5. Treats `.txt` files as lists of URLs and loads them with `WebLoader`.
6. Chunks extracted content.
7. Embeds chunks with `all-MiniLM-L6-v2`.
8. Upserts vectors and payloads into Qdrant.
9. Updates tenant status and CloudWatch metrics.

### Standard Query Pipeline

`POST /api/query` runs the standard RAG pipeline:

1. Dense retrieval from Qdrant.
2. Sparse retrieval with BM25.
3. Reciprocal Rank Fusion.
4. Cross-encoder reranking, with optional MMR-style diversity.
5. Answer generation using Groq `llama-3.3-70b-versatile`.

### Agentic Query Pipeline

`POST /api/query/agentic` enables a ReAct agent that can reason over which tools to use before producing an answer.

The core principle in this codebase is that reasoning and memory are for navigation only. Final answers are generated from retrieved observations, not from the hidden reasoning trace or memory alone.

Agentic query supports:

- `max_iterations` from 1 to 5
- `tool_mode` of `strict` or `relaxed`
- optional `conversation_id`
- optional streaming with `?stream=true`

### Tools

Built-in tools are managed by `ToolRegistry`:

- `retrieve`
- `web_search`
- `calculator`

Tenants can register additional HTTP-backed tools through `/api/tenant/tools/register`.

Tool modes:

| Mode | Behavior |
| --- | --- |
| `strict` | Only faithful tools are available. Faithful tools are grounded in tenant data. |
| `relaxed` | All tools are available, including external/unfaithful tools. |

### Memory

Agent memory is managed by `MemoryManager` and persisted through DynamoDB.

Memory types:

- Conversation memory: short-term context for follow-up questions.
- Pattern memory: long-term navigation hints learned from prior successful interactions.

Memory is exposed through `/api/tenant/memory/*` endpoints for stats, clearing, export, and conversation listing.

## API Endpoints

All tenant, query, tool, memory, and document endpoints require:

```http
X-API-Key: rsk_...
```

`GET /api` and `GET /api/health` do not require tenant credentials.

### Service

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/api` | Service description and workflow summary. |
| `GET` | `/api/health` | Health check for DynamoDB and S3 connectivity. |

### Tenant Management

| Method | Endpoint | Description |
| --- | --- | --- |
| `POST` | `/api/tenant/register` | Register a tenant with a tenant JSON payload and Qdrant config file. |
| `GET` | `/api/tenant/status` | Return tenant status, storage type, chunks indexed, and build stats. |
| `GET` | `/api/tenant/credentials` | Return managed S3 upload credentials. |
| `POST` | `/api/tenant/credentials/rotate` | Rotate managed S3 upload credentials. |
| `PUT` | `/api/tenant/config` | Update tenant Qdrant config and clear cached pipelines. |
| `DELETE` | `/api/tenant` | Delete tenant metadata and managed tenant data. |
| `GET` | `/api/tenant/metrics` | Return CloudWatch metric series and latest values. |

### Document Management

| Method | Endpoint | Description |
| --- | --- | --- |
| `POST` | `/api/tenant/{tenant_id}/upload` | Direct file upload for managed-storage tenants. |
| `POST` | `/api/tenant/{tenant_id}/build` | Trigger index build or rebuild. |

The direct upload endpoint accepts `.pdf` and `.txt` files and limits uploads to 50 files, 100 MB per file. Tenants using `own_s3` should upload directly to their own bucket using aws cli.

### Query

| Method | Endpoint | Description |
| --- | --- | --- |
| `POST` | `/api/query` | Standard RAG query. |
| `POST` | `/api/query/agentic` | Agentic ReAct query. |
| `POST` | `/api/query/agentic?stream=true` | Stream agent reasoning and answer events as Server-Sent Events. |

### Tool Management

| Method | Endpoint | Description |
| --- | --- | --- |
| `POST` | `/api/tenant/tools/register` | Register a tenant custom tool. |
| `GET` | `/api/tenant/tools?mode=strict` | List tools available in strict mode. |
| `GET` | `/api/tenant/tools?mode=relaxed` | List tools available in relaxed mode. |
| `DELETE` | `/api/tenant/tools/{tool_name}` | Unregister a custom tool. |

Built-in tool names cannot be registered or unregistered by tenants.

### Memory Management

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/api/tenant/memory/stats` | Return memory stats for all conversations. |
| `GET` | `/api/tenant/memory/stats?conversation_id={id}` | Return memory stats for one conversation. |
| `POST` | `/api/tenant/memory/clear` | Clear `conversation`, `patterns`, or `all` memory. |
| `GET` | `/api/tenant/memory/export` | Export learned patterns. |
| `GET` | `/api/tenant/memory/conversations` | List tenant conversations. |

## Request Examples

### Tenant Config File

Create a JSON or YAML config file for Qdrant:

```yaml
COLLECTION_NAME: acme_docs
QDRANT_URL: https://your-qdrant-cluster.example
QDRANT_API_KEY: your_qdrant_api_key
```

### Register a Tenant

```bash
curl -X POST "http://localhost:8000/api/tenant/register" \
  -F 'tenant={"company_name":"Acme Corp","contact_email":"ops@acme.test","storage_type":"managed"}' \
  -F "config_file=@config.yaml"
```

Response includes:

- `tenant_id`
- `api_key`
- storage details
- next-step instructions

### Upload Documents

For both managed and own_s3 storage, upload directly to S3 with the credentials returned at registration:

```bash
aws s3 cp ./documents/ s3://<bucket>/<prefix> --recursive
```

For managed-storage tenants, direct API upload is also available:

```bash
curl -X POST "http://localhost:8000/api/tenant/<tenant_id>/upload" \
  -H "X-API-Key: rsk_..." \
  -F "files=@./documents/policy.pdf" \
  -F "files=@./documents/urls.txt;type=text/plain"
```

### Build the Index

```bash
curl -X POST "http://localhost:8000/api/tenant/<tenant_id>/build" \
  -H "X-API-Key: rsk_..."
```

### Check Tenant Status

```bash
curl "http://localhost:8000/api/tenant/status" \
  -H "X-API-Key: rsk_..."
```

### Standard Query

```bash
curl -X POST "http://localhost:8000/api/query" \
  -H "X-API-Key: rsk_..." \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the refund policy?","return_metadata":true}'
```

### Agentic Query

```bash
curl -X POST "http://localhost:8000/api/query/agentic" \
  -H "X-API-Key: rsk_..." \
  -H "Content-Type: application/json" \
  -d '{"question":"Compare the pricing and support terms.","max_iterations":3,"return_metadata":true,"tool_mode":"strict"}'
```

### Continue a Conversation

Send the returned `conversation_id` on later agentic requests:

```bash
curl -X POST "http://localhost:8000/api/query/agentic" \
  -H "X-API-Key: rsk_..." \
  -H "Content-Type: application/json" \
  -d '{"question":"What about enterprise support?","conversation_id":"<conversation_id>"}'
```

If no `conversation_id` is provided, the server starts a new conversation and returns a new ID.

### Streaming Agentic Query

```bash
curl -N -X POST "http://localhost:8000/api/query/agentic?stream=true" \
  -H "X-API-Key: rsk_..." \
  -H "Content-Type: application/json" \
  -d '{"question":"Walk through the relevant policy details.","max_iterations":3}'
```

The response is `text/event-stream` with events such as:

- `reasoning`
- `answer_start`
- `answer_chunk`
- `answer_complete`
- `complete`
- `error`

### Register a Custom Tool

```bash
curl -X POST "http://localhost:8000/api/tenant/tools/register" \
  -H "X-API-Key: rsk_..." \
  -H "Content-Type: application/json" \
  -d '{
    "name": "enterprise_search",
    "description": "Search internal enterprise data for additional context.",
    "faithful": true,
    "endpoint_url": "https://tools.example.com/search",
    "method": "POST",
    "headers": {"Content-Type": "application/json"}
  }'
```

### Clear Memory

```bash
curl -X POST "http://localhost:8000/api/tenant/memory/clear" \
  -H "X-API-Key: rsk_..." \
  -H "Content-Type: application/json" \
  -d '{"memory_type":"all"}'
```

## Local Development

### Prerequisites

- Python 3.10+
- AWS credentials with access to S3, DynamoDB, IAM, Lambda, and CloudWatch
- Qdrant Cloud or reachable Qdrant instance
- Groq API key
- Docker, if building container images

### Virtual Environment

Create a virtal environment and install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=eu-west-1
ACCOUNT_ID=your_aws_account_id
USE_LAMBDA=false
```

Set `USE_LAMBDA=true` in deployed environments when Lambda should handle index builds.

### Run the API

```powershell
.venv\Scripts\activate
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Open:

```text
http://localhost:8000/docs
```

## DynamoDB Setup

The tenant table is created by `MLOpsPipeline` if needed. Memory and custom tool tables have setup scripts:

```powershell
.venv\Scripts\activate
python scripts\setup_memory_tables.py
python scripts\setup_custom_tools_table.py
```

Expected tables include:

- `rag-tenants`
- `RAG-ConversationMemory`
- `RAG-PatternMemory`
- `RAG-CustomTools`

## Testing

Endpoint tests live under `tests/` and are grouped by endpoint family:

```text
tests/
  root_tests/
  tenant_tests/
  document_tests/
  query_tests/
  tool_tests/
  memory_tests/
```

Run the tests:

```powershell
.venv\Scripts\Activate.ps1
pytest tests/ -v
```

The endpoint test harness stubs AWS, model, memory, and tool dependencies so tests run locally without hitting external services.


## Tenant Statuses

| Status | Meaning |
| --- | --- |
| `awaiting_data` | Tenant is registered but no index has been built. |
| `building` | Index build is in progress. |
| `ready` | Index is available and query endpoints can run. |
| `failed` | Last index build failed. Check tenant status for the error. |

## Monitoring

API metrics are logged through `MLOpsPipeline` to CloudWatch. Tenant metrics can be queried through:

```text
GET /api/tenant/metrics?hours=24&period=300
```

Tracked metric categories include:

- query latency
- tokens used
- query count
- index build duration
- chunks indexed
- files processed
- build success



## Important Implementation Notes

- Query endpoints require tenant status to be `ready`.
- `POST /api/tenant/{tenant_id}/upload` is intended for managed-storage tenants. `own_s3` tenants should upload directly to their own bucket.
- `.txt` files are interpreted as URL lists during index building.
- The project currently uses `all-MiniLM-L6-v2` embeddings, which produce 384-dimensional vectors.
- Qdrant collections are created automatically if they do not already exist.
- The API lifespan initializes shared embedder, reranker, and generator instances.
- Tenant pipelines are cached in `api.pipeline_helper.tenant_pipelines` and cleared when tenant config changes.
- Memory is intentionally used for retrieval/navigation context, not as a source of final answer facts.

