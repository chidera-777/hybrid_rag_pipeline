# RAG-as-a-Service

A multi-tenant Retrieval-Augmented Generation (RAG) API service. Each tenant registers, uploads their documents, builds a vector index, and queries their own isolated knowledge base.

## Architecture

```
Client
  │
  ▼
ALB (port 80)
  │
  ▼
ECS Fargate (FastAPI)
  ├── DynamoDB        → tenant registry & status
  ├── S3              → document storage
  ├── AWS Lambda      → async index building
  ├── Qdrant Cloud    → vector store (per tenant collection)
  ├── Groq (LLaMA 3.3 70B) → answer generation
  └── CloudWatch      → query & build metrics
```

**Query pipeline per request:**
1. Dense retrieval (Qdrant — `all-MiniLM-L6-v2` embeddings)
2. Sparse retrieval (BM25)
3. Reciprocal Rank Fusion
4. Reranking with diversity (Cross-Encoder `ms-marco-MiniLM-L-6-v2` + MMR)
5. Answer generation (LLaMA 3.3 70B via Groq)

## Project Structure

```
├── api/
│   ├── main.py               # FastAPI routes
│   └── auth.py               # API key authentication
├── ingestion/
│   ├── pdf_loader.py         # PDF extraction & chunking
│   ├── web_loader.py         # Web scraping (requests + Selenium fallback)
│   ├── chunker.py            # Text chunking
│   └── base_loader.py        # Document base class
├── vectorstore/
│   └── qdrant_store.py       # Qdrant vector store operations
├── retrieval/
│   ├── hybrid.py             # Reciprocal Rank Fusion
│   └── sparse_retriever.py   # BM25 sparse retrieval
├── reranker/
│   └── cross_encoder.py      # Cross-encoder reranking with MMR diversity
├── generation/
│   └── generator.py          # LLM answer generation (Groq)
├── mlops/
│   ├── pipeline.py           # MLOpsPipeline — tenant management & orchestration
│   ├── build_index.py        # IndexBuilder — document ingestion logic
│   └── storage_manager.py    # S3 & IAM management
├── lambda/
│   └── build_index.py        # Lambda handler for async index builds
├── schemas/
│   └── tenant.py             # Pydantic request/response models
├── pipeline.py               # RAGPipeline — query execution
├── config.py                 # Project-wide config & paths
├── Dockerfile                # API container image
├── Dockerfile.lambda         # Lambda container image
├── task-definition.json      # ECS task definition
└── requirements.txt
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Service health check |
| `POST` | `/api/tenant/register` | Register a new tenant |
| `GET` | `/api/tenant/status` | Get tenant status & index stats |
| `GET` | `/api/tenant/credentials` | Get S3 upload credentials (managed storage) |
| `POST` | `/api/tenant/credentials/rotate` | Rotate S3 credentials |
| `POST` | `/api/tenant/{tenant_id}/upload` | Upload documents directly via API |
| `POST` | `/api/tenant/{tenant_id}/build` | Trigger index build |
| `POST` | `/api/query` | Query the tenant's knowledge base |
| `GET` | `/api/tenant/metrics` | Get CloudWatch metrics |
| `DELETE` | `/api/tenant` | Delete tenant and all associated data |

All endpoints except `/health` and `/register` require an `X-API-Key` header.

## Storage Options

**Managed** — the service creates an S3 bucket and a scoped IAM user for the tenant. Credentials are returned at registration.

**Own S3** — tenant provides their own S3 bucket. The bucket must be accessible to the service's IAM role.

## Supported Document Types

| File | Ingestion Method |
|------|-----------------|
| `.pdf` | PyMuPDF extraction + chunking |
| `.txt` | Treated as a list of URLs, scraped via `WebLoader` |

## Getting Started

### Prerequisites

- Python 3.10+
- Docker
- AWS account with permissions for: ECS, ECR, Lambda, S3, DynamoDB, IAM, CloudWatch, Secrets Manager
- [Qdrant Cloud](https://cloud.qdrant.io) account
- [Groq](https://console.groq.com) API key

### Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=eu-west-1
ACCOUNT_ID=your_aws_account_id
USE_LAMBDA=true   # false to use threading locally
```

These same variables must be stored in AWS Secrets Manager under `rag-pipeline/env` for production use.

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the API
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Set `USE_LAMBDA=false` in `.env` to use threading instead of Lambda for index builds locally.

### Deploying to AWS

**1. Build and push the API image**
```bash
docker build -f Dockerfile -t rag-pipeline .
docker tag rag-pipeline:latest <account_id>.dkr.ecr.eu-west-1.amazonaws.com/rag-pipeline:latest
docker push <account_id>.dkr.ecr.eu-west-1.amazonaws.com/rag-pipeline:latest
```

**2. Build and push the Lambda image**
```bash
docker build -f Dockerfile.lambda -t rag-index-builder .
docker tag rag-index-builder:latest <account_id>.dkr.ecr.eu-west-1.amazonaws.com/rag-index-builder:latest
docker push <account_id>.dkr.ecr.eu-west-1.amazonaws.com/rag-index-builder:latest
```

**3. Deploy Lambda function** (first time only)
```bash
aws lambda create-function \
  --function-name RAG-IndexBuild \
  --package-type Image \
  --code ImageUri=<account_id>.dkr.ecr.eu-west-1.amazonaws.com/rag-index-builder:latest \
  --role arn:aws:iam::<account_id>:role/<lambda_execution_role> \
  --timeout 900 \
  --memory-size 3008 \
  --region eu-west-1
```

**4. Update Lambda after code changes**
```bash
aws lambda update-function-code \
  --function-name RAG-IndexBuild \
  --image-uri <account_id>.dkr.ecr.eu-west-1.amazonaws.com/rag-index-builder:latest \
  --region eu-west-1
```

**5. Deploy ECS service**
```bash
aws ecs update-service \
  --cluster <cluster-name> \
  --service <service-name> \
  --force-new-deployment \
  --region eu-west-1
```

## Usage Workflow

```bash
# 1. Register your organisation
curl -X POST "http://<alb-url>/api/tenant/register" \
  -F 'tenant={"company_name":"Acme","contact_email":"you@acme.com","storage_type":"managed"}' \
  -F "config_file=@config.yaml"

# Response includes: tenant_id, api_key, S3 credentials

# 2. Upload documents to S3
aws s3 cp ./documents/ s3://<bucket>/<prefix> --recursive

# 3. Build the index
curl -X POST "http://<alb-url>/api/tenant/<tenant_id>/build" \
  -H "X-API-Key: rsk_..."

# 4. Poll status until "ready"
curl "http://<alb-url>/api/tenant/status" \
  -H "X-API-Key: rsk_..."

# 5. Query
curl -X POST "http://<alb-url>/api/query" \
  -H "X-API-Key: rsk_..." \
  -F 'request={"question":"What is the refund policy?"}'
```

### Tenant Config File (`config.yaml`)

```yaml
QDRANT_URL: https://your-cluster.qdrant.io
QDRANT_API_KEY: your_qdrant_api_key
COLLECTION_NAME: your_collection_name
```

## Tenant Statuses

| Status | Meaning |
|--------|---------|
| `awaiting_data` | Registered, no documents uploaded yet |
| `building` | Index build in progress |
| `ready` | Index built, queries accepted |
| `failed` | Build failed — check `/api/tenant/status` for error details |

## Monitoring

Lambda execution logs:
```
CloudWatch → Log Groups → /aws/lambda/RAG-IndexBuild
```

API logs:
```
CloudWatch → Log Groups → /ecs/rag-pipeline
```

Tenant metrics (query latency, tokens used, build duration, etc.):
```
GET /api/tenant/metrics?hours=24&period=300
```
