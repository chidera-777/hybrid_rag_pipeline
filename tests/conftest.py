import importlib
import json
import sys
import types
from datetime import datetime
from enum import Enum
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class DummyTable:
    def __init__(self):
        self.table_status = "ACTIVE"
        self.deleted_keys = []
        self.updated_items = []

    def update_item(self, **kwargs):
        self.updated_items.append(kwargs)
        return {"ResponseMetadata": {"HTTPStatusCode": 200}}

    def delete_item(self, **kwargs):
        self.deleted_keys.append(kwargs.get("Key"))
        return {"ResponseMetadata": {"HTTPStatusCode": 200}}


class DummyCloudWatch:
    def get_metric_data(self, **kwargs):
        now = datetime(2026, 6, 5, 12, 0, 0)
        return {
            "MetricDataResults": [
                {"Id": "querylatency", "Timestamps": [now], "Values": [0.42]},
                {"Id": "tokensused", "Timestamps": [now], "Values": [12]},
            ]
        }


class DummyMLOpsPipeline:
    tenants_table = DummyTable()
    cloudwatch = DummyCloudWatch()

    def __init__(self):
        self.tenants_table = self.__class__.tenants_table
        self.cloudwatch = self.__class__.cloudwatch
        self.s3 = types.SimpleNamespace(list_buckets=lambda: {"Buckets": []})

    def validate_tenant_config(self, config):
        return True

    def store_tenant(self, tenant_id, api_key, company_name, contact_email, storage_info, config):
        self.stored_tenant = {
            "tenant_id": tenant_id,
            "api_key": api_key,
            "company_name": company_name,
            "contact_email": contact_email,
            "storage_info": storage_info,
            "config": config,
            "status": "awaiting_data",
            "created_at": "2026-06-05T12:00:00",
        }

    def get_tenant_by_tenant_id(self, tenant_id):
        return {
            "tenant_id": tenant_id,
            "created_at": "2026-06-05T12:00:00",
        }

    def get_tenant_by_api_key(self, api_key):
        return None

    def trigger_index_build(self, tenant_id):
        self.triggered_tenant_id = tenant_id

    def update_tenant_status(self, tenant_id, status, **kwargs):
        self.updated_status = (tenant_id, status, kwargs)

    def log_query_metrics(self, tenant_id, latency, tokens_used):
        self.logged_metrics = (tenant_id, latency, tokens_used)


class DummyStorageManager:
    def __init__(self, *args, **kwargs):
        self.s3 = types.SimpleNamespace(list_buckets=lambda: {"Buckets": []})

    def create_managed_storage(self, tenant_id):
        return {
            "type": "managed",
            "bucket": "service-bucket",
            "prefix": f"tenants/{tenant_id}/",
            "region": "eu-west-1",
            "iam_user": f"tenant-{tenant_id}",
            "access_credentials": {
                "access_key_id": "AKIA_TEST",
                "secret_access_key": "SECRET_TEST",
            },
        }

    def validate_tenant_bucket(self, bucket, region):
        return True

    def rotate_tenant_credentials(self, tenant_id):
        return {"access_key_id": "AKIA_ROTATED", "secret_access_key": "SECRET_ROTATED"}

    def delete_tenant_data(self, tenant_id, storage_info):
        self.deleted = (tenant_id, storage_info)

    def list_tenant_files(self, tenant_id, storage_info):
        return ["doc.txt"]


class DummyMemoryStore:
    def get_conversation_history(self, tenant_id, conversation_id):
        return [{"question": "first"}, {"question": "second"}]

    def list_conversations(self, tenant_id):
        return [
            {
                "conversation_id": "conv-1",
                "turn_count": 2,
                "first_question": "What is RAG?",
                "last_updated": "2026-06-05T12:00:00",
            }
        ]

    def get_pattern_statistics(self, tenant_id):
        return {"pricing": {"query_count": 3}}

    def clear_conversation_history(self, tenant_id, conversation_id=None):
        self.cleared_conversation = (tenant_id, conversation_id)

    def clear_patterns(self, tenant_id):
        self.cleared_patterns = tenant_id

    def get_patterns(self, tenant_id):
        return [{"category": "pricing", "query_count": 3}]


class DummyToolMode(str, Enum):
    STRICT = "strict"
    RELAXED = "relaxed"


class DummyToolRegistry:
    def __init__(self, mode=DummyToolMode.STRICT):
        self.mode = mode
        self.tools = {
            "retrieve": {"name": "retrieve", "faithful": True},
            "calculator": {"name": "calculator", "faithful": False},
        }

    def set_mode(self, mode):
        self.mode = mode

    def register_custom_tool(self, name, description, faithful, endpoint_url, method="POST", headers=None, auth_token=None):
        self.tools[name] = {"name": name, "description": description, "faithful": faithful}

    def unregister_tool(self, name):
        return self.tools.pop(name, None) is not None

    def list_tools(self):
        return list(self.tools.values())


class DummyToolStore:
    def __init__(self, *args, **kwargs):
        self.tools = {}

    def get_tool(self, tenant_id, tool_name):
        return self.tools.get((tenant_id, tool_name))

    def list_tools(self, tenant_id):
        return [tool for (tid, _), tool in self.tools.items() if tid == tenant_id]

    def save_tool(self, tenant_id, tool_name, description, faithful, endpoint_url, method="POST", headers=None, auth_token=None):
        self.tools[(tenant_id, tool_name)] = {
            "tenant_id": tenant_id,
            "tool_name": tool_name,
            "description": description,
            "faithful": faithful,
            "endpoint_url": endpoint_url,
            "method": method,
            "headers": headers or {},
            "auth_token": auth_token,
        }

    def delete_tool(self, tenant_id, tool_name):
        return self.tools.pop((tenant_id, tool_name), None) is not None


class DummyPipeline:
    def query(self, question, return_metadata=False):
        return {
            "answer": f"Answer to {question}",
            "sources": [{"document": "doc.txt"}],
            "model": "dummy-model",
            "metadata": {"returned": return_metadata},
        }

    def agentic_query(self, question, max_iterations=3, return_metadata=False, stream=False):
        if stream:
            return iter(
                [
                    {"type": "reasoning", "content": "thinking"},
                    {
                        "type": "answer_complete",
                        "answer": f"Agent answer to {question}",
                        "conversation_id": "conv-1",
                    },
                ]
            )
        return {
            "answer": f"Agent answer to {question}",
            "sources": [{"document": "doc.txt"}],
            "model": "dummy-agent",
            "iterations": max_iterations,
            "conversation_id": "conv-1",
            "metadata": {"trace": return_metadata},
        }


def _install_import_stubs():
    sentence_transformers = types.ModuleType("sentence_transformers")
    sentence_transformers.SentenceTransformer = lambda *args, **kwargs: object()
    sys.modules.setdefault("sentence_transformers", sentence_transformers)

    cross_encoder = types.ModuleType("reranker.cross_encoder")
    cross_encoder.Reranker = lambda *args, **kwargs: object()
    sys.modules["reranker.cross_encoder"] = cross_encoder

    generator = types.ModuleType("generation.generator")
    generator.Generator = lambda *args, **kwargs: object()
    sys.modules["generation.generator"] = generator

    pipeline_module = types.ModuleType("pipeline")
    pipeline_module.RAGPipeline = lambda *args, **kwargs: DummyPipeline()
    sys.modules["pipeline"] = pipeline_module

    tools_module = types.ModuleType("agent.tools")
    tools_module.ToolRegistry = DummyToolRegistry
    tools_module.ToolMode = DummyToolMode
    sys.modules["agent.tools"] = tools_module

    tool_store_module = types.ModuleType("agent.tools.tool_store")
    tool_store_module.ToolStore = DummyToolStore
    sys.modules["agent.tools.tool_store"] = tool_store_module

    memory_store_module = types.ModuleType("agent.memory.dynamodb_store")
    memory_store_module.DynamoDBMemoryStore = DummyMemoryStore
    sys.modules["agent.memory.dynamodb_store"] = memory_store_module


_install_import_stubs()


@pytest.fixture(scope="session")
def app():
    import mlops.pipeline
    import mlops.storage_manager

    mlops.pipeline.MLOpsPipeline = DummyMLOpsPipeline
    mlops.storage_manager.StorageManager = DummyStorageManager

    app_module = importlib.import_module("api.main")
    from api.app_state import set_app_state

    set_app_state(object(), object(), object())
    return app_module.app


@pytest.fixture
def tenant():
    return {
        "tenant_id": "tenant-123",
        "company_name": "Acme Corp",
        "contact_email": "ops@acme.test",
        "status": "ready",
        "storage_info": {
            "type": "managed",
            "bucket": "service-bucket",
            "prefix": "tenants/tenant-123/",
            "region": "eu-west-1",
            "iam_user": "tenant-user",
            "access_credentials": {"access_key_id": "AKIA_TEST", "secret_access_key": "SECRET_TEST"},
        },
        "config": {
            "COLLECTION_NAME": "acme_docs",
            "QDRANT_URL": "http://qdrant.test",
            "QDRANT_API_KEY": "qdrant-key",
        },
        "chunks_indexed": 10,
        "files_processed": 2,
    }


@pytest.fixture
def client(app, tenant):
    from api.auth import authenticate_tenant, require_tenant_ready

    async def override_authenticate_tenant():
        return tenant

    async def override_require_tenant_ready():
        return tenant

    app.dependency_overrides[authenticate_tenant] = override_authenticate_tenant
    app.dependency_overrides[require_tenant_ready] = override_require_tenant_ready

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


@pytest.fixture
def config_upload():
    return {
        "config_file": (
            "config.json",
            json.dumps(
                {
                    "COLLECTION_NAME": "acme_docs",
                    "QDRANT_URL": "http://qdrant.test",
                    "QDRANT_API_KEY": "qdrant-key",
                }
            ),
            "application/json",
        )
    }
