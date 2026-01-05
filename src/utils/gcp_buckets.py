"""
Google Cloud Storage Helper Class
Provides a singleton client for downloading blobs from GCS buckets.
"""

from google.cloud import storage
from typing import Optional
import io

class GCSHelper:
    """
    Singleton helper class for Google Cloud Storage operations.
    Creates a single client instance and reuses it across the application.
    """
    
    _instance = None
    _client = None
    
    def __new__(cls, project_id: Optional[str] = None):
        """Ensure only one instance of GCSHelper exists (Singleton pattern)."""
        if cls._instance is None:
            cls._instance = super(GCSHelper, cls).__new__(cls)
            cls._instance._initialize_client(project_id)
        return cls._instance
    
    def _initialize_client(self, project_id: Optional[str] = None):
        """Initialize the GCS client."""
        try:
            if project_id:
                self._client = storage.Client(project=project_id)
            else:
                self._client = storage.Client()
            print("GCS client initialized successfully")
        except Exception as e:
            print(f"Error initializing GCS client: {e}")
            raise
    
    @property
    def client(self) -> storage.Client:
        """Get the GCS client instance."""
        if self._client is None:
            raise RuntimeError("GCS client not initialized")
        return self._client
    
    def download_as_text(self, bucket_name: str, blob_path: str, encoding: str = 'utf-8') -> str:
        """
        Download a blob as text.
        
        Args:
            bucket_name: Name of the GCS bucket
            blob_path: Path to the blob within the bucket
            encoding: Text encoding (default: utf-8)
            
        Returns:
            String content of the blob
        """
        try:
            bucket = self._client.bucket(bucket_name) # type: ignore
            blob = bucket.blob(blob_path)
            content = blob.download_as_bytes().decode(encoding)
            return content
        except Exception as e:
            print(f"Error downloading {blob_path} from {bucket_name}: {e}")
            raise
        
    def download_to_stream(self, bucket_name: str, blob_path: str) -> io.BytesIO:
        """
        Download a blob to a BytesIO stream.
        
        Args:
            bucket_name: Name of the GCS bucket
            blob_path: Path to the blob within the bucket
            
        Returns:
            BytesIO stream containing the blob data
        """
        try:
            bucket = self._client.bucket(bucket_name) # type: ignore
            blob = bucket.blob(blob_path)
            stream = io.BytesIO()
            blob.download_to_file(stream)
            stream.seek(0)
            return stream
        except Exception as e:
            print(f"Error downloading {blob_path} from {bucket_name}: {e}")
            raise
        
    def list_blobs(self, bucket_name: str, prefix: Optional[str] = None) -> list:
        """
        List all blobs in a bucket with optional prefix filter.
        
        Args:
            bucket_name: Name of the GCS bucket
            prefix: Optional prefix to filter blobs
            
        Returns:
            List of blob names
        """
        try:
            bucket = self._client.bucket(bucket_name) # type: ignore
            blobs = bucket.list_blobs(prefix=prefix)
            return [blob.name for blob in blobs if not blob.name.endswith("/")]
        except Exception as e:
            print(f"Error listing blobs in {bucket_name}: {e}")
            raise