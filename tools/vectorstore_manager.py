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

import logging
from pathlib import Path
import threading
import time
from typing import List, Optional

import chromadb
from chromadb.config import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from typing import Union

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s',
    handlers=[
        logging.FileHandler("vectorstore_manager.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Class to manage Chroma-based vector store."""

    def __init__(self, resources_dir: Path, chroma_dir: Path):
        """
        Initializes the VectorStoreManager with the specified directories.

        Args:
            resources_dir (Path): Directory containing text documents to be indexed.
            chroma_dir (Path): Directory where the Chroma vector store will be stored.
        """
        # Define type for embeddings before assignment
        embeddings: Union[OpenAIEmbeddings, HuggingFaceEmbeddings]
        # if any change is done with the embeddings, make sure to delete the old vectorstore
        # before starting the server
        if False:  # This block is currently not executed, kept for potential future use
            # embeddings = OpenAIEmbeddings()
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small", request_timeout=60, chunk_size=50)
        else:
            model_name: str = "all-MiniLM-L6-v2"
            embeddings = HuggingFaceEmbeddings(model_name=model_name)

        self.embeddings = embeddings
        self.resources_dir = resources_dir
        self.chroma_dir = chroma_dir
        self.lock = threading.Lock()

        self.client = chromadb.PersistentClient(
            path=str(self.chroma_dir),
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection_name = "document_collection"
        self.vectorstore = self._initialize_vectorstore()

    def _initialize_vectorstore(self) -> Chroma:
        """
        Initializes or loads the Chroma vector store.

        If the collection already exists, it loads the existing collection.
        If not, it builds a new vector store from the documents in the resources directory.

        Returns:
            Chroma: The initialized or loaded Chroma vector store.
        """
        with self.lock:
            if self.collection_name in self.client.list_collections():
                logger.info("Loading existing Chroma collection.")
                return Chroma(
                    client=self.client,
                    collection_name=self.collection_name,
                    embedding_function=self.embeddings
                )
            else:
                logger.info("Creating new Chroma collection.")
                return self._build_vectorstore()

    def _load_documents(self) -> List:
        """
        Load text documents from the resources directory.

        Returns:
            List: A list of documents loaded from text files in the resources directory.
        """
        documents = []
        for file_path in self.resources_dir.glob("*.txt"):
            loader = TextLoader(str(file_path))
            documents.extend(loader.load())
        logger.info(f"Loaded {len(documents)} documents from {self.resources_dir}.")

        return documents

    def _build_vectorstore(self) -> Chroma:
        """
        Build Chroma vectorstore from scratch.

        This method loads documents from the resources directory, splits them into chunks,
        and creates a new Chroma vector store.

        Returns:
            Chroma: The newly created Chroma vector store.
        """
        documents = self._load_documents()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
        chunks = splitter.split_documents(documents)

        vectorstore = Chroma.from_documents(
            chunks,
            self.embeddings,
            client=self.client,
            collection_name=self.collection_name
        )
        logger.info("Vectorstore built successfully.")

        return vectorstore

    def update_vectorstore(self):
        """Update the vectorstore with any new or modified documents."""
        with self.lock:
            logger.info("Updating the vectorstore with current documents.")
            self.client.delete_collection(self.collection_name)
            self.vectorstore = self._build_vectorstore()
            logger.info("Vectorstore updated successfully.")

    def query(self, question: str, k: int = 5) -> Optional[List[str]]:
        """
        Query the vector store to retrieve relevant documents.

        Args:
            question (str): The query text.
            k (int): Number of relevant documents to retrieve.

        Returns:
            Optional[List[str]]: List of relevant document contents.
        """
        with self.lock:
            results = self.vectorstore.similarity_search(question, k=k)
            return [doc.page_content for doc in results]

    def start_periodic_updates(self, interval_seconds: int = 60):
        """
        Start a background thread to periodically update the vectorstore.

        Args:
            interval_seconds (int): Interval in seconds to check for updates.
        """

        def periodic_update():
            logger.info("Background vectorstore updater thread started.")
            known_files = {
                file: file.stat().st_mtime for file in self.resources_dir.glob("*.txt")
            }
            while True:
                time.sleep(interval_seconds)
                current_files = {
                    file: file.stat().st_mtime for file in self.resources_dir.glob("*.txt")
                }
                if current_files != known_files:
                    logger.info("Detected changes in resources, updating vectorstore...")
                    self.update_vectorstore()
                    known_files = current_files.copy()

        update_thread = threading.Thread(target=periodic_update, daemon=True)
        update_thread.start()
