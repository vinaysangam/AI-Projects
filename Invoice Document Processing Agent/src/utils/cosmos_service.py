# This module has been removed -- the invoice pipeline does not use Cosmos DB.

import json
from datetime import datetime, timezone
from typing import Any, Optional

from azure.cosmos import CosmosClient, PartitionKey, exceptions
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.config.settings import Settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CosmosDBService:
    """Manages Cosmos DB operations for the patient monitoring system.

    Uses a singleton CosmosClient pattern — create one instance and reuse it.
    Partition key: ``/patient_id`` for high cardinality and query-aligned access.
    """

    def __init__(self, settings: Settings) -> None:
        if not settings.cosmos_endpoint or not settings.cosmos_key:
            logger.warning("Cosmos DB not configured — persistence disabled")
            self._enabled = False
            return

        self._enabled = True
        self._client = CosmosClient(
            url=settings.cosmos_endpoint,
            credential=settings.cosmos_key,
            connection_verify=True,
        )
        self._database_name = settings.cosmos_database
        self._container_name = settings.cosmos_container
        self._partition_key_path = settings.cosmos_partition_key

        self._database = None
        self._container = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    def initialise(self) -> None:
        """Create database and containers if they don't exist.

        Called once at application startup.
        """
        if not self._enabled:
            return

        try:
            self._database = self._client.create_database_if_not_exists(
                id=self._database_name
            )
            self._container = self._database.create_container_if_not_exists(
                id=self._container_name,
                partition_key=PartitionKey(path=self._partition_key_path),
                offer_throughput=400,  # Start low; use autoscale in production
            )

            # Create alerts container
            self._alerts_container = self._database.create_container_if_not_exists(
                id="alerts",
                partition_key=PartitionKey(path="/patient_id"),
                offer_throughput=400,
            )

            # Create recommendations container
            self._recommendations_container = self._database.create_container_if_not_exists(
                id="recommendations",
                partition_key=PartitionKey(path="/patient_id"),
                offer_throughput=400,
            )

            logger.info(
                "Cosmos DB initialised: database=%s containers=[%s, alerts, recommendations]",
                self._database_name,
                self._container_name,
            )
        except exceptions.CosmosHttpResponseError as e:
            logger.error("Failed to initialise Cosmos DB: %s (status=%s)", e.message, e.status_code)
            raise

    # --- Telemetry CRUD -------------------------------------------------------

    @retry(
        retry=retry_if_exception_type(exceptions.CosmosHttpResponseError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def store_telemetry(self, telemetry_doc: dict[str, Any]) -> dict[str, Any]:
        """Store a telemetry document.

        Args:
            telemetry_doc: Serialised patient telemetry with ``patient_id``.

        Returns:
            The created document with Cosmos DB metadata.
        """
        if not self._enabled or self._container is None:
            logger.debug("Cosmos DB disabled — skipping telemetry store")
            return telemetry_doc

        doc = self._prepare_document(telemetry_doc, doc_type="telemetry")

        try:
            result = self._container.create_item(body=doc)
            logger.info(
                "Stored telemetry for patient %s (RU charge: %.2f)",
                doc.get("patient_id"),
                result.get("_ts", 0),
            )
            return result
        except exceptions.CosmosResourceExistsError:
            logger.warning("Telemetry document already exists: %s", doc.get("id"))
            return doc

    @retry(
        retry=retry_if_exception_type(exceptions.CosmosHttpResponseError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def store_alert(self, alert_doc: dict[str, Any]) -> dict[str, Any]:
        """Store a patient alert document."""
        if not self._enabled or self._alerts_container is None:
            return alert_doc

        doc = self._prepare_document(alert_doc, doc_type="alert")
        try:
            return self._alerts_container.create_item(body=doc)
        except exceptions.CosmosResourceExistsError:
            logger.warning("Alert document already exists: %s", doc.get("id"))
            return doc

    @retry(
        retry=retry_if_exception_type(exceptions.CosmosHttpResponseError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def store_recommendation(self, rec_doc: dict[str, Any]) -> dict[str, Any]:
        """Store an intervention recommendation document."""
        if not self._enabled or self._recommendations_container is None:
            return rec_doc

        doc = self._prepare_document(rec_doc, doc_type="recommendation")
        try:
            return self._recommendations_container.create_item(body=doc)
        except exceptions.CosmosResourceExistsError:
            logger.warning("Recommendation document already exists: %s", doc.get("id"))
            return doc

    def store_monitoring_summary(self, summary_doc: dict[str, Any]) -> dict[str, str]:
        """Store all parts of a monitoring summary (telemetry, alert, recommendation).

        Returns:
            Dict of document type → document id for stored items.
        """
        stored: dict[str, str] = {}

        self.store_telemetry(summary_doc)
        stored["telemetry"] = summary_doc.get("patient_id", "unknown")

        if summary_doc.get("alert"):
            self.store_alert(summary_doc["alert"])
            stored["alert"] = summary_doc["alert"].get("patient_id", "unknown")

        if summary_doc.get("recommendation"):
            self.store_recommendation(summary_doc["recommendation"])
            stored["recommendation"] = summary_doc["recommendation"].get("patient_id", "unknown")

        return stored

    # --- Query ----------------------------------------------------------------

    def get_patient_telemetry(
        self,
        patient_id: str,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Retrieve recent telemetry for a patient (single-partition query).

        Args:
            patient_id: The patient identifier (partition key).
            limit: Maximum documents to return.

        Returns:
            List of telemetry documents.
        """
        if not self._enabled or self._container is None:
            return []

        query = "SELECT TOP @limit * FROM c WHERE c.patient_id = @pid AND c.doc_type = 'telemetry' ORDER BY c.timestamp DESC"
        parameters: list[dict[str, Any]] = [
            {"name": "@pid", "value": patient_id},
            {"name": "@limit", "value": limit},
        ]

        items = list(
            self._container.query_items(
                query=query,
                parameters=parameters,
                partition_key=patient_id,
            )
        )
        logger.info("Retrieved %d telemetry docs for patient %s", len(items), patient_id)
        return items

    def get_patient_alerts(
        self,
        patient_id: str,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Retrieve recent alerts for a patient (single-partition query)."""
        if not self._enabled or self._alerts_container is None:
            return []

        query = "SELECT TOP @limit * FROM c WHERE c.patient_id = @pid ORDER BY c.timestamp DESC"
        parameters: list[dict[str, Any]] = [
            {"name": "@pid", "value": patient_id},
            {"name": "@limit", "value": limit},
        ]

        return list(
            self._alerts_container.query_items(
                query=query,
                parameters=parameters,
                partition_key=patient_id,
            )
        )

    def get_patient_recommendations(
        self,
        patient_id: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Retrieve recent recommendations for a patient (single-partition query)."""
        if not self._enabled or self._recommendations_container is None:
            return []

        query = "SELECT TOP @limit * FROM c WHERE c.patient_id = @pid ORDER BY c.timestamp DESC"
        parameters: list[dict[str, Any]] = [
            {"name": "@pid", "value": patient_id},
            {"name": "@limit", "value": limit},
        ]

        return list(
            self._recommendations_container.query_items(
                query=query,
                parameters=parameters,
                partition_key=patient_id,
            )
        )

    # --- Helpers --------------------------------------------------------------

    @staticmethod
    def _prepare_document(doc: dict[str, Any], doc_type: str) -> dict[str, Any]:
        """Add required Cosmos DB fields to a document."""
        doc = dict(doc)  # shallow copy

        # Ensure 'id' field exists (required by Cosmos DB)
        if "id" not in doc:
            patient_id = doc.get("patient_id", "unknown")
            ts = doc.get("timestamp", datetime.now(timezone.utc).isoformat())
            doc["id"] = f"{patient_id}_{doc_type}_{ts}"

        doc["doc_type"] = doc_type
        doc["_created_at"] = datetime.now(timezone.utc).isoformat()

        return doc
