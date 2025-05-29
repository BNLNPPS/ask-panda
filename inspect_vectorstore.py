# inspect_vectorstore.py
import argparse
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def load_vectorstore(vectorstore_dir: Path):
    """Load FAISS vectorstore from specified directory.

    Args:
        vectorstore_dir (Path): Path to the directory containing the vectorstore.

    Returns:
        FAISS: Loaded vectorstore object.
    """
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(
        str(vectorstore_dir), embeddings, allow_dangerous_deserialization=True
    )
    return vectorstore

def display_vectorstore_contents(vectorstore, dump_binary: bool):
    """Display the contents of the vectorstore.

    Args:
        vectorstore (FAISS): The loaded vectorstore.
        dump_binary (bool): If True, also display raw vector data.
    """
    docstore_dict = vectorstore.docstore._dict
    for doc_id, doc in docstore_dict.items():
        print(f"Document ID: {doc_id}")

        if dump_binary:
            embedding = vectorstore.embeddings.embed_query(doc.page_content)
            binary_vector = ' '.join(f'{x:.4f}' for x in embedding)
            print(f"Vector Embedding:\n{binary_vector}\n")
        else:
            print(f"Content:\n{doc.page_content}\n")

        print("-" * 40)

def main():
    parser = argparse.ArgumentParser(description="Inspect FAISS vectorstore contents.")
    parser.add_argument(
        "--vectorstore-dir",
        type=Path,
        default=Path("vectorstore"),
        help="Path to the FAISS vectorstore directory."
    )
    parser.add_argument(
        "--dump",
        action="store_true",
        help="Dump vector embedding data as well."
    )

    args = parser.parse_args()

    vectorstore = load_vectorstore(args.vectorstore_dir)
    display_vectorstore_contents(vectorstore, args.dump)

if __name__ == "__main__":
    main()
