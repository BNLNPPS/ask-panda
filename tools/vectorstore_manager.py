# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
# Authors:
# - Paul Nilsson, paul.nilsson@cern.ch, 2025

"""VectorStoreManager class for managing vector stores."""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from pathlib import Path
from typing import List, Iterable, Set, Union

import chromadb
from chromadb.config import Settings

# LangChain community packages
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
    logger.addHandler(_h)


class VectorStoreManager:
    """
    Manages a persistent Chroma collection via LangChain's Chroma wrapper.

    Key features:
    - Uses stable IDs for chunks so repeated indexing doesn't append duplicates.
    - Uses collection.upsert(...) to replace-or-add chunks.
    - Optional pruning of vectors whose 'source' files are gone.
    """

    def __init__(
        self,
        resources_dir: Path,
        chroma_dir: Path,
        collection_name: str = "document_collection",
        chunk_size: int = 2000,
        chunk_overlap: int = 150,
        file_glob: str = "*.txt",
        index_on_start: bool = True,
        prune_missing_on_start: bool = False,
    ):
        self.resources_dir = Path(resources_dir)
        self.chroma_dir = Path(chroma_dir)
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.file_glob = file_glob
        self.lock = threading.Lock()

        # Define type for embeddings before assignment
        embeddings: Union[OpenAIEmbeddings, HuggingFaceEmbeddings]
        # if any change is done with the embeddings, make sure to delete the old vectorstore
        # before starting the server
        if False:  # This block is currently not executed, kept for potential future use
            # embeddings = OpenAIEmbeddings()
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small", request_timeout=60, chunk_size=50)
        else:
            model_name: str = "all-MiniLM-L6-v2"  # sentence-transformers/all-mpnet-base-v2
            embeddings = HuggingFaceEmbeddings(model_name=model_name)

        self.embeddings = embeddings

        # Create a persistent Chroma client with telemetry disabled
        self.client = chromadb.PersistentClient(
            path=str(self.chroma_dir),
            settings=Settings(anonymized_telemetry=False),
        )

        # Ensure the collection exists, then create the LC wrapper
        self.vectorstore = self._initialize_vectorstore()

        # Optionally perform an idempotent indexing pass on startup
        if index_on_start:
            self.index_incremental(prune_missing=prune_missing_on_start)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def _initialize_vectorstore(self) -> Chroma:
        with self.lock:
            existing_names = {c.name for c in self.client.list_collections()}
            if self.collection_name not in existing_names:
                self.client.get_or_create_collection(self.collection_name)
                logger.info(f"Created collection '{self.collection_name}'.")
            else:
                logger.info(f"Loaded existing collection '{self.collection_name}'.")

            return Chroma(
                client=self.client,
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
            )

    # ------------------------------------------------------------------
    # Loading and splitting
    # ------------------------------------------------------------------
    def _load_documents(self) -> List:
        docs = []
        paths = sorted(self.resources_dir.glob(self.file_glob))
        for p in paths:
            loader = TextLoader(str(p), encoding="utf-8")
            docs.extend(loader.load())
        logger.info(f"Loaded {len(docs)} documents from {self.resources_dir} ({self.file_glob}).")
        return docs

    def _split_documents(self, docs: List) -> List:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        chunks = splitter.split_documents(docs)
        logger.info(f"Split into {len(chunks)} chunks (size={self.chunk_size}, overlap={self.chunk_overlap}).")
        return chunks

    # ------------------------------------------------------------------
    # ID generation and indexing
    # ------------------------------------------------------------------
    @staticmethod
    def _content_digest(text: str, n: int = 16) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]

    def _build_stable_ids(self, chunks: Iterable, use_digest: bool = True) -> List[str]:
        ids: List[str] = []
        for i, ch in enumerate(chunks):
            src = str(ch.metadata.get("source", "unknown"))
            if use_digest:
                digest = self._content_digest(ch.page_content, n=16)
                uid = f"{src}::{i}::{digest}"
            else:
                uid = f"{src}::{i}"
            ids.append(uid)
        return ids

    # ------------------------------------------------------------------
    # Public indexing API
    # ------------------------------------------------------------------
    def index_incremental(self, prune_missing: bool = True) -> int:
        with self.lock:
            collection = self.client.get_or_create_collection(self.collection_name)

            docs = self._load_documents()
            chunks = self._split_documents(docs)

            if not chunks:
                logger.info("No chunks to index.")
                if prune_missing:
                    self._prune_missing(collection)
                return 0

            texts = [c.page_content for c in chunks]
            metadatas = [c.metadata for c in chunks]
            ids = self._build_stable_ids(chunks, use_digest=True)

            logger.info("Computing embeddings for upsert...")
            embeddings = self.embeddings.embed_documents(texts)
            collection.upsert(
                ids=ids,
                documents=texts,
                metadatas=metadatas,
                embeddings=embeddings,
            )
            logger.info(f"Upserted {len(texts)} chunks into '{self.collection_name}'.")

            if prune_missing:
                self._prune_missing(collection)

            # Refresh LC wrapper
            self.vectorstore = Chroma(
                client=self.client,
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
            )

            return len(texts)

    def _prune_missing(self, collection) -> None:
        current_sources = self._current_source_paths()
        if not current_sources:
            logger.info("Prune skipped: no current sources detected.")
            return

        collection.delete(where={"source": {"$nin": list(current_sources)}})
        logger.info("Pruned chunks whose source files are missing.")

    def _current_source_paths(self) -> Set[str]:
        return {str(p) for p in self.resources_dir.glob(self.file_glob)}

    # ------------------------------------------------------------------
    # Query and utility methods
    # ------------------------------------------------------------------
    def query(self, question: str, k: int = 5) -> List[str]:
        with self.lock:
            docs = self.vectorstore.similarity_search(question, k=k)
            return [d.page_content for d in docs]

    def collection_count(self) -> int:
        with self.lock:
            col = self.client.get_or_create_collection(self.collection_name)
            try:
                return col.count()
            except Exception:
                res = col.get(include=["documents"], limit=0)
                return len(res.get("ids", []))

    def start_periodic_updates(self, interval_seconds: int = 60, prune_missing: bool = True) -> None:
        def _snapshot() -> dict[Path, float]:
            return {p: p.stat().st_mtime for p in self.resources_dir.glob(self.file_glob)}

        def worker():
            logger.info("Background vectorstore updater thread started.")
            known = _snapshot()
            while True:
                time.sleep(interval_seconds)
                current = _snapshot()
                if current != known:
                    logger.info("Detected changes in resources; reindexing incrementally...")
                    self.index_incremental(prune_missing=prune_missing)
                    known = current

        t = threading.Thread(target=worker, daemon=True)
        t.start()

    def prune_now(self) -> None:
        with self.lock:
            col = self.client.get_or_create_collection(self.collection_name)
            self._prune_missing(col)


# ---------------------------------------------------------------------
# Example usage (adjust paths to your project layout)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    mgr = VectorStoreManager(
        resources_dir=Path("./resources"),
        chroma_dir=Path("./chroma_store"),
        collection_name="document_collection",
        chunk_size=2000,
        chunk_overlap=150,
        file_glob="*.txt",
        index_on_start=True,
        prune_missing_on_start=False,
    )

    logger.info(f"Collection count: {mgr.collection_count()}")
    answers = mgr.query("What topics are covered?", k=3)
    for i, a in enumerate(answers, 1):
        print(f"\n--- Result {i} ---\n{a[:500]}")
