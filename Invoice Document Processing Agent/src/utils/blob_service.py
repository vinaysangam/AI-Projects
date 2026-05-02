# This module has been removed — the invoice pipeline does not use Blob Storage.

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from azure.storage.blob import BlobServiceClient, ContentSettings
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.config.settings import Settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Allowed upload file extensions
ALLOWED_EXTENSIONS: set[str] = {".json", ".csv"}
MAX_UPLOAD_SIZE_MB: int = 10


class BlobStorageService:
    """Manages Azure Blob Storage operations for document upload/retrieval."""

    def __init__(self, settings: Settings) -> None:
        if not settings.azure_storage_connection_string:
            logger.warning("Azure Blob Storage not configured — upload disabled")
            self._enabled = False
            return

        self._enabled = True
        self._client = BlobServiceClient.from_connection_string(
            settings.azure_storage_connection_string
        )
        self._container_name = settings.azure_storage_container_name
        self._container_client = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    def initialise(self) -> None:
        """Create the blob container if it doesn't exist."""
        if not self._enabled:
            return

        try:
            self._container_client = self._client.create_container(self._container_name)
            logger.info("Created blob container: %s", self._container_name)
        except Exception:
            # Container already exists
            self._container_client = self._client.get_container_client(self._container_name)
            logger.info("Using existing blob container: %s", self._container_name)

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def upload_document(
        self,
        file_content: bytes,
        filename: str,
        content_type: str = "application/json",
        metadata: dict[str, str] | None = None,
    ) -> dict[str, str]:
        """Upload a telemetry document to Blob Storage.

        Args:
            file_content: Raw file bytes.
            filename: Original filename.
            content_type: MIME content type.
            metadata: Optional blob metadata.

        Returns:
            Dict with ``blob_name``, ``url``, and ``upload_id``.

        Raises:
            ValueError: If file type is not allowed or size exceeds limit.
        """
        if not self._enabled or self._container_client is None:
            raise RuntimeError("Blob Storage is not configured")

        # Validate extension
        ext = Path(filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise ValueError(f"File type '{ext}' not allowed. Accepted: {ALLOWED_EXTENSIONS}")

        # Validate size
        size_mb = len(file_content) / (1024 * 1024)
        if size_mb > MAX_UPLOAD_SIZE_MB:
            raise ValueError(f"File size {size_mb:.1f} MB exceeds limit of {MAX_UPLOAD_SIZE_MB} MB")

        upload_id = uuid.uuid4().hex[:12]
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        blob_name = f"uploads/{ts}_{upload_id}_{filename}"

        blob_metadata = {
            "upload_id": upload_id,
            "original_filename": filename,
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
        }
        if metadata:
            blob_metadata.update(metadata)

        blob_client = self._container_client.get_blob_client(blob_name)
        blob_client.upload_blob(
            data=file_content,
            overwrite=True,
            content_settings=ContentSettings(content_type=content_type),
            metadata=blob_metadata,
        )

        logger.info(
            "Uploaded document: blob=%s size=%.2f MB upload_id=%s",
            blob_name,
            size_mb,
            upload_id,
        )

        return {
            "blob_name": blob_name,
            "url": blob_client.url,
            "upload_id": upload_id,
            "size_mb": f"{size_mb:.2f}",
        }

    def download_document(self, blob_name: str) -> bytes:
        """Download a document from Blob Storage.

        Args:
            blob_name: The blob path.

        Returns:
            Raw file bytes.
        """
        if not self._enabled or self._container_client is None:
            raise RuntimeError("Blob Storage is not configured")

        blob_client = self._container_client.get_blob_client(blob_name)
        download = blob_client.download_blob()
        data = download.readall()
        logger.info("Downloaded blob: %s (%d bytes)", blob_name, len(data))
        return data

    def list_uploads(self, prefix: str = "uploads/") -> list[dict[str, str]]:
        """List uploaded documents.

        Args:
            prefix: Blob name prefix to filter.

        Returns:
            List of blob metadata dicts.
        """
        if not self._enabled or self._container_client is None:
            return []

        blobs = []
        for blob in self._container_client.list_blobs(name_starts_with=prefix):
            blobs.append({
                "name": blob.name,
                "size": blob.size,
                "last_modified": blob.last_modified.isoformat() if blob.last_modified else "",
                "content_type": blob.content_settings.content_type if blob.content_settings else "",
            })
        return blobs

    def parse_uploaded_json(self, file_content: bytes) -> list[dict[str, Any]]:
        """Parse uploaded JSON content into telemetry records.

        Accepts a single object or an array.

        Args:
            file_content: Raw JSON bytes.

        Returns:
            List of telemetry dicts.
        """
        raw = json.loads(file_content.decode("utf-8"))
        if isinstance(raw, dict):
            return [raw]
        if isinstance(raw, list):
            return raw
        raise ValueError("Uploaded JSON must be an object or array of objects")
