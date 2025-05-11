import os
import glob
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from typing import List, Dict, Any

def load_documents_from_folder(folder_path: str) -> List[Dict[Any, Any]]:
    txt_files = glob.glob(os.path.join(folder_path, "*.txt"))

    if not txt_files:
        raise ValueError(f"No .txt files found in {folder_path}")

    print(f"Found {len(txt_files)} .txt files in {folder_path}")

    documents = []
    for file_path in txt_files:
        try:
            loader = TextLoader(file_path, encoding="utf-8")
            documents.extend(loader.load())
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    return documents

def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )

    split_docs = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(split_docs)} chunks")

    return split_docs

def create_embeddings_and_store(documents, embeddings_model_name, index_name="./VectorStore/GV"):
    embeddings = HuggingFaceEmbeddings(
        model_name=embeddings_model_name,
        model_kwargs={"device": "cpu"}
    )

    vectorstore = FAISS.from_documents(documents, embeddings)

    vectorstore.save_local(index_name)
    print(f"Saved vector store to {index_name}")

    return vectorstore

def process_indian_laws(folder_path, embeddings_model_name="sentence-transformers/LaBSE",
                        chunk_size=1000, chunk_overlap=200,
                        index_name="./VectorStore/GV"):
    
    print(f"Loading documents from {folder_path}...")
    documents = load_documents_from_folder(folder_path)

    print("Splitting documents into chunks...")
    split_docs = split_documents(documents, chunk_size, chunk_overlap)

    print(f"Creating embeddings using {embeddings_model_name}...")
    vectorstore = create_embeddings_and_store(split_docs, embeddings_model_name, index_name)

    return vectorstore

if __name__ == "__main__":
    folder_path = "./Acts/GiAct"  
    embeddings_model_name = "sentence-transformers/LaBSE"
    chunk_size = 1000
    chunk_overlap = 200
    index_name = "./VectorStore/GV"
    vectorstore = process_indian_laws(
        folder_path,
        embeddings_model_name=embeddings_model_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        index_name=index_name
    )

    print("\nExample of how to query the vector store:")
    print("""
    # Load the saved vector store
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS

    # Initialize the embeddings model (must be the same as used for indexing)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/LaBSE",
        model_kwargs={"device": "cpu"}
    )

    # Load the vector store
    vectorstore = FAISS.load_local("indian_laws_index", embeddings)

    # Search the vector store
    query = "What are the penalties for corruption under Indian law?"
    docs = vectorstore.similarity_search(query, k=3)

    # Print the results
    for doc in docs:
        print(doc.page_content)
        print("-" * 50)
    """)
