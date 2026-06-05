def test_query_returns_answer(client):
    response = client.post(
        "/api/query",
        json={"question": "What is in the docs?", "return_metadata": True},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Answer to What is in the docs?"
    assert body["tenant_id"] == "tenant-123"
    assert body["metadata"]["returned"] is True


def test_agentic_query_returns_reasoning_response(client):
    response = client.post(
        "/api/query/agentic",
        json={"question": "Plan an answer", "max_iterations": 2, "return_metadata": True},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Agent answer to Plan an answer"
    assert body["iterations"] == 2
    assert body["conversation_id"] == "conv-1"


def test_agentic_query_streams_sse_events(client):
    response = client.post(
        "/api/query/agentic?stream=true",
        json={"question": "Stream an answer"},
    )

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    assert '"type": "reasoning"' in response.text
    assert '"type": "complete"' in response.text
