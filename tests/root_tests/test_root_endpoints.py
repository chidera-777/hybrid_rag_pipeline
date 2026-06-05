def test_root_returns_service_workflow(client):
    response = client.get("/api")

    assert response.status_code == 200
    body = response.json()
    assert body["service"] == "RAG-as-a-Service"
    assert "workflow" in body


def test_health_returns_component_status(client):
    response = client.get("/api/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["components"]["dynamodb"] == "connected"
