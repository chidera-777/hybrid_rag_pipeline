def test_get_memory_stats_for_all_conversations(client):
    response = client.get("/api/tenant/memory/stats")

    assert response.status_code == 200
    body = response.json()
    assert body["tenant_id"] == "tenant-123"
    assert body["conversation_turns"] == 2
    assert body["pattern_statistics"]["pricing"]["query_count"] == 3


def test_get_memory_stats_for_specific_conversation(client):
    response = client.get("/api/tenant/memory/stats?conversation_id=conv-1")

    assert response.status_code == 200
    body = response.json()
    assert body["conversation_summary"] == "Conversation conv-1 with 2 turns"


def test_clear_memory_patterns(client):
    response = client.post("/api/tenant/memory/clear", json={"memory_type": "patterns"})

    assert response.status_code == 200
    body = response.json()
    assert body["memory_type"] == "patterns"
    assert body["message"] == "Pattern memory cleared"


def test_export_memory_patterns(client):
    response = client.get("/api/tenant/memory/export")

    assert response.status_code == 200
    body = response.json()
    assert body["tenant_id"] == "tenant-123"
    assert body["total_patterns"] == 1


def test_list_memory_conversations(client):
    response = client.get("/api/tenant/memory/conversations")

    assert response.status_code == 200
    body = response.json()
    assert body["total_conversations"] == 1
    assert body["conversations"][0]["conversation_id"] == "conv-1"
