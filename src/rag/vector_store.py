"""Multimodal Qdrant vector storage, inherits LangChain QdrantVectorStore.

Extended functions：
1. Multimodal input storage（_multimodal_input payload）
2. Deterministic UUIDv5 ID (idempotent write）
3. Image search
4. Hybrid Search mode（dense + sparse）
"""

import threading
import uuid
from typing import Any

try:
    import torch  # pyright: ignore[reportMissingImports]
except ImportError:
    torch = None
from langchain_qdrant import QdrantVectorStore, RetrievalMode  # pyright: ignore[reportMissingImports]
from qdrant_client import QdrantClient  # pyright: ignore[reportMissingImports]
from qdrant_client.http import models  # pyright: ignore[reportMissingImports]

from config.settings import config
from src.utils.logger import get_logger

logger = get_logger(__name__)

_NS = uuid.UUID("c0ffeeee-dead-beef-cafe-000000000000")


def _with_current_filter(
    filter_obj: models.Filter | None,
    current_only: bool,
) -> models.Filter | None:
    if not current_only:
        return filter_obj

    current_condition = models.FieldCondition(
        key="metadata.is_current",
        match=models.MatchValue(value=True),
    )

    if filter_obj is None:
        return models.Filter(must=[current_condition])

    must_conditions = list(filter_obj.must or [])
    must_conditions.append(current_condition)
    return models.Filter(
        must=must_conditions,
        should=filter_obj.should,
        must_not=filter_obj.must_not,
        min_should=filter_obj.min_should,
    )


def _content_uuid(*parts: str) -> str:
    """Deterministic UUIDv5 for idempotent writes.

    The same content generates the same ID, and repeated upsert will overwrite rather than create a duplicate.。
    """
    key = "\x00".join(parts)
    return str(uuid.uuid5(_NS, key))


class MultimodalQdrantStore(QdrantVectorStore):
    MULTIMODAL_KEY = "_multimodal_input"

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        embedding: Any,
        sparse_embedding: Any | None = None,
        retrieval_mode: RetrievalMode = RetrievalMode.DENSE,
        **kwargs,
    ):
        """Initialize multimodal vector storage。

        Args:
            client: Qdrant client
            collection_name: Collection name
            embedding: Multimodal embedding instance (must support dict input）
            sparse_embedding: Optional sparse embedding (required for HYBRID mode）
            retrieval_mode: Search mode（DENSE/SPARSE/HYBRID）
        """
        super().__init__(
            client=client,
            collection_name=collection_name,
            embedding=embedding,
            sparse_embedding=sparse_embedding,
            retrieval_mode=retrieval_mode,
            validate_collection_config=False,
            **kwargs,
        )
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Make sure the collection exists, create it if it does not exist。"""
        if self.client.collection_exists(self.collection_name):
            return

        vector_size = len(self._embeddings.embed_query("test"))

        if self.retrieval_mode == RetrievalMode.DENSE:
            vectors_config = models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            )
            sparse_vectors_config = None
        elif self.retrieval_mode == RetrievalMode.HYBRID:
            vectors_config = models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            )
            sparse_vectors_config = {
                self.sparse_vector_name: models.SparseVectorParams()
            }
        else:  # SPARSE
            vectors_config = None
            sparse_vectors_config = {
                self.sparse_vector_name: models.SparseVectorParams()
            }

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_vectors_config,
        )
        logger.info("Created Qdrant collection: %s", self.collection_name)

    # ========== multimodal storage ==========

    def add_multimodal(
        self,
        inputs: list[dict[str, Any]],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
        batch_size: int | None = None,
    ) -> list[str]:
        """Store multi-modal input and support image and text mixing embedding。

        Args:
            inputs: Multimodal input list, each element is dict:
                - {"text": "description text"}
                - {"image": "/path/to/image.jpg"}
                - {"text": "describe", "image": "/path/to/image.jpg"}
            metadatas: Metadata list, one-to-one correspondence with inputs
            ids: Optional custom ID, if not provided generates a deterministic UUIDv5
            batch_size: batch size

        Returns:
            List of stored IDs
        """
        metadatas = metadatas or [{} for _ in inputs]

        if len(inputs) != len(metadatas):
            raise ValueError("inputs must be the same length as metadatas")

        # generate certainty ID
        if ids is None:
            ids = [
                _content_uuid(
                    m.get("pdf_name", ""),
                    str(m.get("paper_version", "")),
                    str(m.get("page_idx", "")),
                    m.get("chunk_type", ""),
                    inp.get("text", "") if isinstance(inp, dict) else str(inp),
                    m.get("img_path", ""),
                )
                for inp, m in zip(inputs, metadatas)
            ]

        resolved_batch_size = self._resolve_embedding_batch_size(batch_size)

        all_ids: list[str] = []
        total_batches = -(-len(inputs) // resolved_batch_size)

        for i in range(0, len(inputs), resolved_batch_size):
            batch_inputs = inputs[i : i + resolved_batch_size]
            batch_metas = metadatas[i : i + resolved_batch_size]
            batch_ids = ids[i : i + resolved_batch_size]

            # Use embedding to generate vectors
            vectors = self._embeddings.embed_documents(batch_inputs)

            # build points
            points = []
            for inp, meta, vec, pid in zip(
                batch_inputs, batch_metas, vectors, batch_ids
            ):
                text = inp.get("text", "") if isinstance(inp, dict) else str(inp)
                payload = {
                    self.content_payload_key: text,
                    self.metadata_payload_key: meta,
                    self.MULTIMODAL_KEY: inp,
                }

                # Build vector structure based on search pattern
                if self.retrieval_mode == RetrievalMode.DENSE:
                    vector_struct = {self.vector_name: vec}
                elif self.retrieval_mode == RetrievalMode.HYBRID:
                    sparse_vec = self.sparse_embeddings.embed_documents([text])[0]
                    vector_struct = {
                        self.vector_name: vec,
                        self.sparse_vector_name: models.SparseVector(
                            indices=sparse_vec.indices,
                            values=sparse_vec.values,
                        ),
                    }
                else:  # SPARSE
                    sparse_vec = self.sparse_embeddings.embed_documents([text])[0]
                    vector_struct = {
                        self.sparse_vector_name: models.SparseVector(
                            indices=sparse_vec.indices,
                            values=sparse_vec.values,
                        ),
                    }

                points.append(
                    models.PointStruct(
                        id=pid,
                        vector=vector_struct,
                        payload=payload,
                    )
                )

            self.client.upsert(self.collection_name, points=points)
            all_ids.extend(batch_ids)
            logger.info(
                "Upserted batch %d/%d (%d items)",
                i // resolved_batch_size + 1,
                total_batches,
                len(batch_inputs),
            )
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()

        logger.info(
            "Stored %d multimodal items in collection %s",
            len(inputs),
            self.collection_name,
        )
        return all_ids

    def _resolve_embedding_batch_size(self, batch_size: int | None) -> int:
        requested = batch_size if batch_size is not None else config.EMBEDDING_BATCH_SIZE
        requested = max(1, int(requested))
        if torch is None or not torch.cuda.is_available():
            return requested

        try:
            free_bytes, _ = torch.cuda.mem_get_info()
        except Exception:
            return requested

        model_name = str(getattr(config, "EMBEDDING_MODEL", "")).lower()
        if "72b" in model_name or "32b" in model_name:
            model_scale = 8
        elif "14b" in model_name or "13b" in model_name:
            model_scale = 4
        elif "7b" in model_name:
            model_scale = 2
        else:
            model_scale = 1

        estimated_per_sample = 256 * 1024 * 1024 * model_scale
        safe_limit = int((free_bytes * 0.7) // estimated_per_sample)
        if safe_limit <= 0:
            return 1
        return min(requested, safe_limit)

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter: models.Filter | None = None,
        score_threshold: float = 0.0,
        candidate_k: int | None = None,
        current_only: bool = True,
    ) -> list[dict[str, Any]]:
        """By default, vector search using hybrid retrieval is used to return the original payload format.。

        Args:
            query: Query text
            k: Number of results returned
            filter: Qdrant Filter object
            score_threshold: Score threshold (results below this value are filtered）
            candidate_k: Number of candidates, automatically set to if not specified k

        Returns:
            [{"score": float, "payload": dict}, ...]
        """
        fetch_k = candidate_k
        if fetch_k is None:
            fetch_k = k

        qdrant_filter = _with_current_filter(filter, current_only=current_only)

        # Call parent class search
        results = self.similarity_search_with_score(
            query,
            k=fetch_k,
            filter=qdrant_filter,
            score_threshold=score_threshold if score_threshold > 0 else None,
        )

        docs = [doc for doc, _ in results]
        payloads = self._reconstruct_payloads(docs)

        raw = []
        for (doc, score), payload in zip(results, payloads):
            if score < score_threshold:
                continue
            raw.append({"score": score, "payload": payload})

        return raw[:k]

    def _reconstruct_payloads(self, docs: list[Any]) -> list[dict[str, Any]]:
        payload_by_id: dict[str, dict[str, Any]] = {}
        point_ids: list[Any] = []
        point_id_keys: list[str] = []

        for doc in docs:
            point_id = doc.metadata.get("_id")
            if point_id is None:
                continue
            key = str(point_id)
            if key in payload_by_id:
                continue
            point_id_keys.append(key)
            point_ids.append(point_id)

        if point_ids:
            try:
                points = self.client.retrieve(
                    self.collection_name,
                    point_ids,
                    with_payload=True,
                )
                for key, point in zip(point_id_keys, points):
                    if getattr(point, "payload", None):
                        payload_by_id[key] = dict(point.payload)
            except Exception:
                payload_by_id = {}

        return [self._reconstruct_payload(doc, payload_by_id) for doc in docs]

    def _reconstruct_payload(
        self,
        doc: Any,
        payload_by_id: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        meta = dict(doc.metadata)
        point_id = doc.metadata.get("_id")
        if point_id is not None and payload_by_id:
            payload = payload_by_id.get(str(point_id))
            if payload is not None:
                return payload

        meta.pop("_id", None)
        meta.pop("_collection_name", None)
        return {
            self.content_payload_key: doc.page_content,
            self.metadata_payload_key: meta,
        }

    # ========== Image search ==========

    def search_by_image(
        self,
        image_path: str,
        instruction: str | None = None,
        text: str | None = None,
        k: int = 5,
        filter: models.Filter | None = None,
    ) -> list[dict[str, Any]]:
        """Search pictures by pictures/arts。

        Args:
            image_path: Image path
            instruction: Embed instructions
            text: Optional additional text
            k: Return quantity
            filter: Qdrant Filter

        Returns:
            [{"score": float, "payload": dict}, ...]
        """
        input_dict: dict[str, Any] = {"image": image_path}
        if text:
            input_dict["text"] = text

        vector = self._embeddings.embed_query(
            input_dict,
            instruction=instruction
            or "Retrieve images or text relevant to the user's query.",
        )

        results = self.similarity_search_with_score_by_vector(
            vector,
            k=k,
            filter=filter,
        )

        docs = [doc for doc, _ in results]
        payloads = self._reconstruct_payloads(docs)
        return [
            {"score": score, "payload": payload}
            for (_, score), payload in zip(results, payloads)
        ]

    # ========== Metadata operations ==========

    def mark_paper_chunks_non_current(
        self,
        pdf_name: str,
        keep_version: int,
        batch_size: int = 256,
    ) -> int:
        selector = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.pdf_name",
                    match=models.MatchValue(value=pdf_name),
                )
            ]
        )

        updated_count = 0
        offset: Any | None = None
        while True:
            points, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=selector,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )

            if not points:
                break

            updated_points: list[models.PointStruct] = []
            for point in points:
                payload = dict(point.payload or {})
                metadata = dict(payload.get(self.metadata_payload_key, {}) or {})

                if metadata.get("paper_version") == keep_version:
                    continue
                if metadata.get("is_current") is False:
                    continue

                metadata["is_current"] = False
                payload[self.metadata_payload_key] = metadata
                updated_points.append(
                    models.PointStruct(
                        id=point.id,
                        vector=point.vector,
                        payload=payload,
                    )
                )

            if updated_points:
                self.client.upsert(self.collection_name, points=updated_points)
                updated_count += len(updated_points)

            if next_offset is None:
                break
            offset = next_offset

        return updated_count

    def fetch_by_metadata(
        self,
        filter: models.Filter,
        limit: int = 20,
        current_only: bool = True,
    ) -> list[dict[str, Any]]:
        """Get points by metadata (no vector search）。

        Args:
            filter: Qdrant Filter object
            limit: Maximum return quantity

        Returns:
            [{"payload": dict}, ...]
        """
        qdrant_filter = _with_current_filter(filter, current_only=current_only)
        points, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=qdrant_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return [{"payload": p.payload} for p in points]

    def scroll_chunks(
        self,
        filter: models.Filter | None = None,
        limit: int = 10000,
        offset: Any | None = None,
        current_only: bool = True,
    ) -> tuple[list[dict[str, Any]], Any | None]:
        """Page scrolling acquisition chunks。

        Args:
            filter: Qdrant Filter object
            limit: Quantity per page
            offset: Paging offset (returned by the previous call to）

        Returns:
            (results, next_offset)
        """
        qdrant_filter = _with_current_filter(filter, current_only=current_only)
        points, next_offset = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=qdrant_filter,
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        return [{"id": p.id, "payload": p.payload} for p in points], next_offset

    def delete_by_metadata(
        self,
        filter: models.Filter,
    ) -> bool:
        """Delete by metadata points。

        Args:
            filter: Qdrant Filter object

        Returns:
            Is it successful?
        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(filter=filter),
            )
            logger.info("Deleted points matching filter")
            return True
        except Exception as exc:
            logger.error("Failed to delete points: %s", exc)
            return False

    def delete_paper(self, pdf_name: str) -> bool:
        """Delete all the specified papers chunks。

        Args:
            pdf_name: Paper title (excluding .pdf suffix）

        Returns:
            Is it successful?
        """
        filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.pdf_name", match=models.MatchValue(value=pdf_name)
                )
            ]
        )
        return self.delete_by_metadata(filter)

    def get_all_papers(
        self,
        filter: models.Filter | None = None,
        current_only: bool = True,
    ) -> list[dict[str, Any]]:
        """Get the payload of all chunks (used to extract the paper list）。"""
        qdrant_filter = _with_current_filter(filter, current_only=current_only)
        points, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=qdrant_filter,
            limit=10000,
            with_payload=True,
            with_vectors=False,
        )
        return [{"payload": p.payload} for p in points]

    def count_chunks(
        self,
        filter: models.Filter | None = None,
        current_only: bool = True,
    ) -> int:
        """Count the number of chunks。

        Args:
            filter: Optional Qdrant Filter

        Returns:
            Number of matching chunks
        """
        qdrant_filter = _with_current_filter(filter, current_only=current_only)
        return self.client.count(
            collection_name=self.collection_name,
            count_filter=qdrant_filter,
            exact=True,
        ).count


# ========== Module-level singleton (lazy initialization） ==========

_vector_store: MultimodalQdrantStore | None = None
_vector_store_lock = threading.Lock()
_qdrant_client: QdrantClient | None = None
_qdrant_client_lock = threading.Lock()


def _get_qdrant_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        with _qdrant_client_lock:
            if _qdrant_client is None:
                client_kwargs: dict[str, Any] = {
                    "url": f"http://{config.QDRANT_HOST}:{config.QDRANT_PORT}",
                    "timeout": config.QDRANT_TIMEOUT_SECONDS,
                }
                try:
                    import httpx  # pyright: ignore[reportMissingImports]

                    client_kwargs["limits"] = httpx.Limits(
                        max_keepalive_connections=config.QDRANT_HTTP_KEEPALIVE_CONNECTIONS,
                        max_connections=config.QDRANT_HTTP_MAX_CONNECTIONS,
                    )
                    _qdrant_client = QdrantClient(**client_kwargs)
                except TypeError:
                    client_kwargs.pop("limits", None)
                    _qdrant_client = QdrantClient(**client_kwargs)
    return _qdrant_client


def _create_vector_store() -> MultimodalQdrantStore:
    """Create vector_store instance (lazy initialization, avoid CUDA/vLLM conflict）。"""
    import torch  # pyright: ignore[reportMissingImports]

    from src.rag.embedding import Qwen3VLEmbeddings

    client = _get_qdrant_client()

    embeddings = Qwen3VLEmbeddings(
        model_name_or_path=config.EMBEDDING_MODEL,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )

    # Sparse embedding（HYBRID model）
    sparse_embedding = None
    retrieval_mode = RetrievalMode.DENSE
    if getattr(config, "ENABLE_HYBRID", False):
        try:
            from langchain_qdrant import FastEmbedSparse  # pyright: ignore[reportMissingImports]

            sparse_embedding = FastEmbedSparse(model_name="Qdrant/bm25")
            retrieval_mode = RetrievalMode.HYBRID
            logger.info("Hybrid mode enabled with FastEmbed sparse")
        except Exception as exc:
            logger.warning(
                "Could not load sparse embedding, falling back to DENSE: %s", exc
            )

    return MultimodalQdrantStore(
        client=client,
        collection_name=config.QDRANT_COLLECTION_NAME,
        embedding=embeddings,
        sparse_embedding=sparse_embedding,
        retrieval_mode=retrieval_mode,
    )


def get_vector_store() -> MultimodalQdrantStore:
    """Get vector_store singleton (thread-safe）。"""
    global _vector_store
    if _vector_store is None:
        with _vector_store_lock:
            if _vector_store is None:
                _vector_store = _create_vector_store()
    return _vector_store


# Backward compatibility: keep old global variable names (but change to function calls）
# Note: Use directly `vector_store` will return None, need to use instead `get_vector_store()`
vector_store = None  # type: ignore[assignment]
