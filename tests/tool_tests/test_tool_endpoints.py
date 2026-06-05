def test_register_custom_tool(client):
    response = client.post(
        "/api/tenant/tools/register",
        json={
            "name": "enterprise_search",
            "description": "Search internal enterprise data for additional context.",
            "faithful": True,
            "endpoint_url": "https://tools.acme.test/search",
            "method": "POST",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["tool_name"] == "enterprise_search"
    assert body["mode_availability"]["strict"] is True


def test_list_tools_in_strict_mode_filters_unfaithful_tools(client):
    response = client.get("/api/tenant/tools?mode=strict")

    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "strict"
    assert all(tool["faithful"] for tool in body["tools"])


def test_unregister_custom_tool(client):
    client.post(
        "/api/tenant/tools/register",
        json={
            "name": "crm_lookup",
            "description": "Look up CRM records grounded in tenant business systems.",
            "faithful": True,
            "endpoint_url": "https://tools.acme.test/crm",
            "method": "GET",
        },
    )

    response = client.delete("/api/tenant/tools/crm_lookup")

    assert response.status_code == 200
    assert response.json()["tool_name"] == "crm_lookup"
