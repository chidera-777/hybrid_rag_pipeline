class DummyS3Client:
    def __init__(self):
        self.uploaded = []

    def upload_fileobj(self, file_obj, bucket, key):
        self.uploaded.append((file_obj.filename, bucket, key))


def test_upload_tenant_data_accepts_managed_storage_direct_upload(client, monkeypatch):
    from api.endpoints import document_endpoints

    s3_client = DummyS3Client()
    monkeypatch.setattr(document_endpoints.boto3, "client", lambda service_name: s3_client)

    response = client.post(
        "/api/tenant/tenant-123/upload",
        files={"files": ("doc.txt", b"hello", "text/plain")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["uploaded"] == 1
    assert body["failed"] == 0
    assert body["files"][0]["s3_key"] == "tenants/tenant-123/doc.txt"
    assert s3_client.uploaded == [("doc.txt", "service-bucket", "tenants/tenant-123/doc.txt")]


def test_upload_tenant_data_rejects_own_s3_direct_upload(client, tenant):
    tenant["storage_info"] = {
        "type": "own_s3",
        "bucket": "customer-bucket",
        "region": "eu-west-1",
        "prefix": "",
    }

    response = client.post(
        "/api/tenant/tenant-123/upload",
        files={"files": ("doc.txt", b"hello", "text/plain")},
    )

    assert response.status_code == 400
    assert "managed storage" in response.json()["detail"]


def test_build_index_starts_for_tenant_with_files(client):
    response = client.post("/api/tenant/tenant-123/build")

    assert response.status_code == 200
    body = response.json()
    assert body["tenant_id"] == "tenant-123"
    assert body["status"] == "building"
    assert body["files_processed"] == 1
