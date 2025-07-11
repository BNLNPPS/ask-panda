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

""" Inspect Chroma Vectorstore Contents"""

import argparse
from pathlib import Path
import chromadb
from chromadb.config import Settings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma  # <-- Updated import


def load_chroma_vectorstore(chroma_dir: Path):
    """
    Load Chroma vectorstore from specified directory.

    Args:
        chroma_dir (Path): Path to the directory containing the Chroma vectorstore.
    Returns:
        Chroma: Loaded Chroma vectorstore object.
    """
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(
        path=str(chroma_dir),
        settings=Settings(anonymized_telemetry=False)
    )

    vectorstore = Chroma(
        client=client,
        collection_name="document_collection",
        embedding_function=embeddings
    )
    return vectorstore


def display_vectorstore_contents(vectorstore: Chroma, dump_binary: bool):
    """
    Display contents of the Chroma vectorstore.

    Args:
        vectorstore (Chroma): The loaded vectorstore.
        dump_binary (bool): If True, also display raw vector embedding data.
    """
    docs = vectorstore.get(include=["documents", "embeddings"])
    ids = docs.get('ids', [])
    contents = docs.get('documents', [])
    embeddings = docs.get('embeddings', [])

    if not ids or not contents:
        print("No documents found in the vectorstore.")
        return

    for idx, doc_id in enumerate(ids):
        print(f"Document ID: {doc_id}")

        if dump_binary:
            embedding_str = ' '.join(f'{x:.4f}' for x in embeddings[idx])
            print(f"Vector Embedding:\n{embedding_str}\n")
        else:
            print(f"Content:\n{contents[idx]}\n")

        print("-" * 40)


def main():
    parser = argparse.ArgumentParser(description="Inspect Chroma vectorstore contents.")
    parser.add_argument(
        "--chroma-dir",
        type=Path,
        default=Path("chromadb"),
        help="Path to the Chroma vectorstore directory."
    )
    parser.add_argument(
        "--dump",
        action="store_true",
        help="Dump vector embedding data as well."
    )

    args = parser.parse_args()

    vectorstore = load_chroma_vectorstore(args.chroma_dir)
    display_vectorstore_contents(vectorstore, args.dump)


if __name__ == "__main__":
    main()
