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

"""This script creates a vector store from a text document."""

# Run it once to initialize the vector store:
# ```python create_vectorstore.py```


from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Load the document and create a vector store
loader = TextLoader("panda_documentation.txt")
docs = loader.load()

# Split the document into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# Use either OpenAI or HuggingFace embeddings
if False:
    # OpenAI embeddings
    embeddings = OpenAIEmbeddings()
else:
    model_name = "all-MiniLM-L6-v2"  # Efficient, recommended model
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

# Create the vector store
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("vectorstore")
