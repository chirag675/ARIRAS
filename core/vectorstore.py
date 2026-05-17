import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from dotenv import load_dotenv

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vectorstore(file_path: str):
    loader      = PyPDFLoader(file_path)
    docs        = loader.load()
    splitter    = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks      = splitter.split_documents(docs)
    embeddings  = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    return vectorstore


def load_vectorstore():
    embeddings = get_embeddings()
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )