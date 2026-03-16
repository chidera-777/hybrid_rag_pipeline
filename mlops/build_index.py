import boto3
import os
from pathlib import Path
import time
from typing import Any, Optional

from mlops.storage_manager import StorageManager
from ingestion.pdf_loader import PDFLoader
from ingestion.web_loader import WebLoader
from vectorstore.qdrant_store import QdrantStore
import logging
import shutil

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class IndexBuilder:
    def __init__(self, mlops: Optional[Any] = None, storage_manager: Optional[StorageManager] = None):
        self.s3 = boto3.client("s3")
        self.mlops = mlops
        self.storage_manager = storage_manager or StorageManager()
        
    
    def build_index(self, tenant_id: str):
        start_time = time.time()
        tenant =  self.mlops.get_tenant_by_tenant_id(tenant_id)
        if not tenant:
            raise ValueError(f"Tenant {tenant_id} not found")
        self.mlops.update_tenant_status(tenant_id, "building")
        
        local_dir = None
        try:
            local_dir = f"/tmp/tenant_{tenant_id}"
            os.makedirs(local_dir, exist_ok=True)
            files = self._download_tenant_files(tenant, local_dir)
            if not files:
                raise ValueError(f"No files found for tenant {tenant_id} to index. Please upload documents first.")
            all_chunks = self._load_documents(files)
            if not all_chunks:
                raise ValueError(f"No valid chunks created. Check your document contents and try again.")
            self._create_vector_index(tenant, all_chunks)
            duration = time.time() - start_time
            self.mlops.update_tenant_status(
                tenant_id,
                "ready",
                chunks_indexed=len(all_chunks),
                files_processed=len(files),
                last_build_duration=round(duration, 2),
                last_build_at=time.strftime('%Y-%m-%d %H:%M:%S')
            )
            self.mlops.log_index_build_metrics(
                tenant_id=tenant_id,
                duration=duration,
                chunks_indexed=len(all_chunks),
                files_processed=len(files),
                success=True
            )
            logging.info(f"Indexing completed for tenant {tenant_id} in {duration} seconds")  
        except Exception as e:
            duration = time.time() - start_time
            self.mlops.update_tenant_status(
                tenant_id,
                "failed",
                error=str(e),
                last_build_duration=round(duration, 2),
            )
            self.mlops.log_index_build_metrics(
                tenant_id=tenant_id,
                duration=duration,
                chunks_indexed=0,
                files_processed=0,
                success=False
            )
            raise
        finally:
            if local_dir and os.path.exists(local_dir):
                shutil.rmtree(local_dir)
                logging.info(f"Cleaned up temp files")
        
        
    
    def _download_tenant_files(self, tenant: dict, local_dir: str):
        storage_info = tenant["storage_info"]
        files = self.storage_manager.list_tenant_files(tenant_id=tenant["tenant_id"], storage_info=storage_info)
        if not files:
            return []
        
        downloaded_files = []
        for i, file in enumerate(files, 1):
            try:
                filename = Path(file["key"]).name
                local_path = os.path.join(local_dir, filename)
                logging.info(f"{i}/{len(files)}: Downloading {filename}...")
                self.s3.download_file(storage_info["bucket"], file["key"], local_path)
                downloaded_files.append(local_path)
            except Exception as e:
                logging.error(f"Error downloading {filename}: {e}")
        return downloaded_files
    
    
    def _load_documents(self, files: list):
        all_chunks = []
        pdf_loader = PDFLoader()
        web_loader = WebLoader()
        for path in files:
            suffix = Path(path).suffix.lower()
            try:
                if suffix == ".pdf":
                    chunks = pdf_loader.load(path)
                    all_chunks.extend(chunks)
                elif suffix == ".txt":
                    with open(path, "r", encoding="utf-8") as f:
                        urls = [line.strip() for line in f if line.strip()]
                    web_chunks, failed = web_loader.load_urls(urls)
                    all_chunks.extend(web_chunks)
                    if failed:
                        logging.warning(f"Failed URLs: {len(failed)}")
            except Exception as e:
                logging.error(f"Error loading {path}: {e}")
        return all_chunks
    
    
    def _create_vector_index(self, tenant: dict, chunks: list):
        config = tenant['config']
        vector_store = QdrantStore(
            url=config["QDRANT_URL"],
            api_key=config["QDRANT_API_KEY"],
            collection_name=config["COLLECTION_NAME"]
        )
        stats = vector_store.add_documents(chunks)
        logging.info(f"Indexing stats: {stats}")
