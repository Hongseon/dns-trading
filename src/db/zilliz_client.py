"""Singleton Zilliz Cloud (Milvus) client wrapper.

Provides a lazily-initialized, module-level MilvusClient so that
every module in the project shares a single connection instance.
Also handles collection creation on first use.
"""

from __future__ import annotations

import logging

from pymilvus import MilvusClient, DataType

from src.config import settings

logger = logging.getLogger(__name__)

_client: MilvusClient | None = None


def get_client() -> MilvusClient:
    """Return the shared MilvusClient, creating it on first call."""
    global _client

    if _client is not None:
        return _client

    if not settings.zilliz_uri or not settings.zilliz_token:
        raise ValueError(
            "ZILLIZ_URI and ZILLIZ_TOKEN must be set. "
            "Check your .env file or environment variables."
        )

    logger.info("Initializing Zilliz client for %s", settings.zilliz_uri)
    _client = MilvusClient(uri=settings.zilliz_uri, token=settings.zilliz_token)

    return _client


# ------------------------------------------------------------------
# Collection initialization
# ------------------------------------------------------------------

_DOCUMENTS_COLLECTION = "documents"
_SYNC_STATE_COLLECTION = "sync_state"
_BRIEFINGS_COLLECTION = "briefings"


def init_collections() -> None:
    """Create all required collections if they do not already exist."""
    client = get_client()
    _init_documents_collection(client)
    _init_sync_state_collection(client)
    _init_briefings_collection(client)
    logger.info("All collections initialized")


def _init_documents_collection(client: MilvusClient) -> None:
    """Create the documents collection with vector + scalar fields."""
    if client.has_collection(_DOCUMENTS_COLLECTION):
        logger.info("Collection '%s' already exists", _DOCUMENTS_COLLECTION)
        return

    schema = client.create_schema(enable_dynamic_field=False)

    schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=settings.embedding_dim)

    # Core fields
    schema.add_field("source_type", DataType.VARCHAR, max_length=20)
    schema.add_field("source_id", DataType.VARCHAR, max_length=500)
    schema.add_field("content", DataType.VARCHAR, max_length=10000)
    schema.add_field("chunk_index", DataType.INT32)

    # Common metadata
    schema.add_field("created_date", DataType.VARCHAR, max_length=50, default_value="")
    schema.add_field("updated_date", DataType.VARCHAR, max_length=50, default_value="")

    # Dropbox metadata
    schema.add_field("filename", DataType.VARCHAR, max_length=500, default_value="")
    schema.add_field("folder_path", DataType.VARCHAR, max_length=1000, default_value="")
    schema.add_field("file_type", DataType.VARCHAR, max_length=20, default_value="")

    # Email metadata
    schema.add_field("email_from", DataType.VARCHAR, max_length=200, default_value="")
    schema.add_field("email_to", DataType.VARCHAR, max_length=500, default_value="")
    schema.add_field("email_subject", DataType.VARCHAR, max_length=500, default_value="")
    schema.add_field("email_date", DataType.VARCHAR, max_length=50, default_value="")

    # Create with COSINE index on embedding
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        index_type="AUTOINDEX",
        metric_type="COSINE",
    )

    client.create_collection(
        collection_name=_DOCUMENTS_COLLECTION,
        schema=schema,
        index_params=index_params,
    )
    logger.info("Created collection '%s'", _DOCUMENTS_COLLECTION)


def _init_sync_state_collection(client: MilvusClient) -> None:
    """Create the sync_state collection for cursor persistence."""
    if client.has_collection(_SYNC_STATE_COLLECTION):
        logger.info("Collection '%s' already exists", _SYNC_STATE_COLLECTION)
        return

    schema = client.create_schema(enable_dynamic_field=False)

    schema.add_field("sync_type", DataType.VARCHAR, max_length=20, is_primary=True)
    schema.add_field("last_cursor", DataType.VARCHAR, max_length=5000, default_value="")
    schema.add_field("last_sync_time", DataType.VARCHAR, max_length=50, default_value="")
    schema.add_field("updated_at", DataType.VARCHAR, max_length=50, default_value="")

    # sync_state has no vector field, but Milvus requires at least one
    # vector field. Use a dummy 1-dim vector.
    schema.add_field("_dummy_vec", DataType.FLOAT_VECTOR, dim=2)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="_dummy_vec",
        index_type="AUTOINDEX",
        metric_type="COSINE",
    )

    client.create_collection(
        collection_name=_SYNC_STATE_COLLECTION,
        schema=schema,
        index_params=index_params,
    )
    logger.info("Created collection '%s'", _SYNC_STATE_COLLECTION)


def _init_briefings_collection(client: MilvusClient) -> None:
    """Create the briefings collection for briefing history."""
    if client.has_collection(_BRIEFINGS_COLLECTION):
        logger.info("Collection '%s' already exists", _BRIEFINGS_COLLECTION)
        return

    schema = client.create_schema(enable_dynamic_field=False)

    schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field("briefing_type", DataType.VARCHAR, max_length=20)
    schema.add_field("content", DataType.VARCHAR, max_length=10000)
    schema.add_field("generated_at", DataType.VARCHAR, max_length=50, default_value="")
    schema.add_field("sent", DataType.BOOL, default_value=False)

    # Milvus requires at least one vector field.
    schema.add_field("_dummy_vec", DataType.FLOAT_VECTOR, dim=2)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="_dummy_vec",
        index_type="AUTOINDEX",
        metric_type="COSINE",
    )

    client.create_collection(
        collection_name=_BRIEFINGS_COLLECTION,
        schema=schema,
        index_params=index_params,
    )
    logger.info("Created collection '%s'", _BRIEFINGS_COLLECTION)
