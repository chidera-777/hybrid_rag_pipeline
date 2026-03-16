import boto3
import json
from botocore.exceptions import ClientError
import logging
import dotenv
import os

dotenv.load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class StorageManager:
    def __init__(self, service_bucket: str = f"rag-pipeline-storage-{os.getenv('ACCOUNT_ID')}"):
        self.s3 = boto3.client("s3")
        self.iam = boto3.client("iam")
        self.service_bucket = service_bucket
        self._ensure_bucket_exists()
        
    def _ensure_bucket_exists(self):
        try:
            buckets = self.s3.list_buckets().get("Buckets", [])
            if any(b["Name"] == self.service_bucket for b in buckets):
                return
        except ClientError:
            pass
        try:
            self.s3.head_bucket(Bucket=self.service_bucket)
            return
        except ClientError as e:
            code = str(e.response.get("Error", {}).get("Code", ""))
            if code not in ("404", "NoSuchBucket", "NotFound"):
                if code in ("403", "AccessDenied"):
                    raise
                if code in ("301", "PermanentRedirect"):
                    raise
        try:
            region = os.getenv("AWS_DEFAULT_REGION")
            if region == "us-east-1":
                self.s3.create_bucket(Bucket=self.service_bucket)
            else:
                self.s3.create_bucket(
                    Bucket=self.service_bucket,
                    CreateBucketConfiguration={"LocationConstraint": region}
                )
            self.s3.put_bucket_versioning(
                Bucket=self.service_bucket,
                VersioningConfiguration={
                    "Status": "Enabled"
                }
            )
            self.s3.put_bucket_lifecycle_configuration(
                Bucket=self.service_bucket,
                LifecycleConfiguration={
                    "Rules": [
                        {
                            "ID": "DeleteOldVersions",
                            "Status": "Enabled",
                            "NoncurrentVersionExpiration": {
                                "NoncurrentDays": 30
                            }
                        }
                    ]
                }
            )
            logging.info(f"Created service bucket: {self.service_bucket}")
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code == "BucketAlreadyOwnedByYou":
                return
            if code == "BucketAlreadyExists":
                raise
            logging.error(f"Failed to create bucket: {e}")
            raise
            
    
    def _ensure_tenant_upload_role(self):
        role_name = "TenantUploadRole"
        try:
            self.iam.get_role(RoleName=role_name)
            logging.info(f"Role {role_name} already exists")
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchEntity":
                logging.info(f"Creating role {role_name}")
                trust_policy = {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Principal": {
                                "AWS": "s3.amazonaws.com"
                            },
                            "Action": "sts:AssumeRole"
                        }
                    ]
                }
                try:
                    self.iam.create_role(
                        RoleName=role_name,
                        AssumeRolePolicyDocument=json.dumps(trust_policy),
                        Description="Role for uploading tenant data to S3"
                    )
                    logging.info(f"Created role {role_name}")
                except Exception as create_error:
                    raise Exception(f"Failed to create role {role_name}: {create_error}")
            else:
                raise Exception(f"Unexpected error: {e}")


    def create_managed_storage(self, tenant_id: str):
        prefix = f"tenants/{tenant_id}/"
        iam_user = self._create_iam_user(tenant_id)
        region = os.getenv("AWS_DEFAULT_REGION")
        return {
            "type": "managed",
            "bucket": self.service_bucket,
            "prefix": prefix,
            "region": region,
            "iam_user": iam_user["username"],
            "access_credentials": {
                "access_key_id": iam_user['access_key_id'],
                "secret_access_key": iam_user['secret_access_key'],
                "note": "Store these credentials securely. You can always change them later."
            }
        }
        
        
    def _create_iam_user(self, tenant_id: str):
        username = f"rag-tenant-{tenant_id}"
        try:
            self.iam.create_user(
                UserName=username,
                Tags=[
                    {"Key": "tenant_id", "Value": tenant_id},
                    {"Key": "Service", "Value": "RAG-as-a-Service"},
                    {"Key": "Purpose", "Value": "RAG Tenant Upload"}
                ]
            )
            logging.info(f"Created user {username}")
            policy_docs = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": [
                            "s3:PutObject",
                            "s3:GetObject",
                            "s3:DeleteObject"
                        ],
                        "Resource": [
                            f"arn:aws:s3:::{self.service_bucket}/tenants/{tenant_id}/*"
                        ],
                    },
                    {
                        "Effect": "Allow",
                        "Action": "s3:ListBucket",
                        "Resource": f"arn:aws:s3:::{self.service_bucket}",
                        "Condition": {
                            "StringLike": {
                                "s3:prefix": [f"tenants/{tenant_id}/*"]
                            }
                        }
                    }
                ]
            }
            self.iam.put_user_policy(
                UserName=username,
                PolicyName=f"TenantAccess-{tenant_id}",
                PolicyDocument=json.dumps(policy_docs)
            )
            response = self.iam.create_access_key(UserName=username)
            access_key = response['AccessKey']
            return {
                "username": username,
                "access_key_id": access_key['AccessKeyId'],
                "secret_access_key": access_key['SecretAccessKey'],
                "created_at": access_key['CreateDate'].isoformat()
            }
        except ClientError as e:
            if e.response["Error"]["Code"] == "EntityAlreadyExists":
                raise Exception(
                    f"IAM user {username} already exists. "
                    "This tenant may have been registered before."
                )
            raise Exception(f"Failed to create IAM user: {e}")
    
    
    def rotate_tenant_credentials(self, tenant_id: str):
        username = f"rag-tenant-{tenant_id}"
        try:
            response = self.iam.list_access_keys(UserName=username)
            for access_key in response['AccessKeyMetadata']:
                self.iam.delete_access_key(
                    UserName=username,
                    AccessKeyId=access_key['AccessKeyId']
                )
            response = self.iam.create_access_key(UserName=username)
            access_key = response['AccessKey']
            return {
                "access_key_id": access_key['AccessKeyId'],
                "secret_access_key": access_key['SecretAccessKey']
            }
        except ClientError as e:
            raise Exception(f"Failed to rotate credentials: {e}")
        
        
    def validate_tenant_bucket(self, bucket: str, region: str = "eu-west-1"):
        try:
            s3_client = boto3.client('s3', region_name=region)
            s3_client.head_bucket(Bucket=bucket)
            return True
        except ClientError:
            return False
        
    
    def list_tenant_files(self, tenant_id: str, storage_info: dict):
        if storage_info["type"] == "managed":
            bucket = storage_info["bucket"]
            prefix = storage_info["prefix"]
        else:
            bucket = storage_info["bucket"]
            prefix = storage_info.get("prefix", "")
            
        try:
            paginator = self.s3.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
            files = []
            for page in pages:
                for obj in page.get('Contents', []):
                    if obj["Key"].endswith((".pdf", ".txt", ".md", ".docx")):
                        files.append({
                            "key": obj["Key"],
                            "size": obj["Size"],
                            "last_modified": obj["LastModified"].isoformat()
                        })      
            return files
        except ClientError as e:
            logging.error(f"Failed to list files: {e}")
            return []
    
    
    def delete_tenant_data(self, tenant_id: str, storage_info: dict):
        if storage_info["type"] == "managed":
            bucket = storage_info["bucket"]
            prefix = storage_info["prefix"]
            
            paginator = self.s3.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
            for page in pages:
                objects = [{"Key": obj["Key"]} for obj in page.get('Contents', [])]
                if objects:
                    self.s3.delete_objects(Bucket=bucket, Delete={"Objects": objects})
                    logging.info(f"Deleted {len(objects)} objects from bucket {bucket}")
            self._delete_iam_user(tenant_id)
            

    def _delete_iam_user(self, tenant_id: str):
        username = f"rag-tenant-{tenant_id}"
        try:
            response = self.iam.list_access_keys(UserName=username)
            for key in response['AccessKeyMetadata']:
                self.iam.delete_access_key(
                    UserName=username,
                    AccessKeyId=key['AccessKeyId']
                )
            
            response = self.iam.list_user_policies(UserName=username)
            for policy in response['PolicyNames']:
                self.iam.delete_user_policy(
                    UserName=username,
                    PolicyName=policy
                )
                
            self.iam.delete_user(UserName=username)
            logging.info(f"Deleted IAM user {username}")
        except ClientError as e:
            logging.error(f"Failed to delete IAM user {username}: {e}")
