import json


def test_register_tenant_with_managed_storage(client, config_upload):
    tenant_payload = {
        "company_name": "Acme Corp",
        "contact_email": "ops@acme.test",
        "storage_type": "managed",
    }

    response = client.post(
        "/api/tenant/register",
        data={"tenant": json.dumps(tenant_payload)},
        files=config_upload,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["company_name"] == "Acme Corp"
    assert body["api_key"].startswith("rsk_")
    assert body["storage_info"]["type"] == "managed"


def test_get_tenant_status(client):
    response = client.get("/api/tenant/status")

    assert response.status_code == 200
    body = response.json()
    assert body["tenant_id"] == "tenant-123"
    assert body["status"] == "ready"
    assert body["chunks_indexed"] == 10


def test_get_tenant_credentials_for_managed_storage(client):
    response = client.get("/api/tenant/credentials")

    assert response.status_code == 200
    body = response.json()
    assert body["bucket"] == "service-bucket"
    assert body["credentials"]["access_key_id"] == "AKIA_TEST"


def test_rotate_tenant_credentials(client):
    response = client.post("/api/tenant/credentials/rotate")

    assert response.status_code == 200
    body = response.json()
    assert body["new_credentials"]["access_key_id"] == "AKIA_ROTATED"


def test_update_tenant_config(client, config_upload):
    response = client.put("/api/tenant/config", files=config_upload)

    assert response.status_code == 200
    body = response.json()
    assert body["tenant_id"] == "tenant-123"
    assert body["updated_config"]["COLLECTION_NAME"] == "acme_docs"


def test_delete_tenant(client):
    response = client.delete("/api/tenant")

    assert response.status_code == 200
    body = response.json()
    assert body["tenant_id"] == "tenant-123"
    assert "deleted successfully" in body["message"]


def test_get_tenant_metrics(client):
    response = client.get("/api/tenant/metrics?hours=1&period=60")

    assert response.status_code == 200
    body = response.json()
    assert body["tenant_id"] == "tenant-123"
    assert body["latest"]["querylatency"] == 0.42
