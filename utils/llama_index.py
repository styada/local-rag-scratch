import os
import torch
import streamlit as st
import utils.logs as logs

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
)

# === Force global Settings to use a local Hugging Face embedder (runs entirely locally) ===
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5",
    device="cpu" if not torch.backends.mps.is_available() else "mps"
)

###################################
#
# Setup Embedding Model (legacy)
#
###################################

@st.cache_resource(show_spinner=False)
def setup_embedding_model(
    model: str,
):
    """
    (Legacy) Placeholder; embedding model is controlled by global Settings.
    """
    logs.log.info("Using global HuggingFaceEmbedding for embeddings.")
    return

###################################
#
# Load Documents
#
###################################

def load_documents(data_dir: str):
    """
    Loads documents from a directory of files.

    Args:
        data_dir (str): Path to the directory containing documents.

    Returns:
        List of loaded documents.
    """
    try:
        reader = SimpleDirectoryReader(input_dir=data_dir, recursive=True)
        documents = reader.load_data(reader)
        logs.log.info(f"Loaded {len(documents):,} documents from files")
        return documents
    except Exception as err:
        logs.log.error(f"Error loading documents: {err}")
        raise
    finally:
        # Clean up any temporary files
        for file in os.scandir(data_dir):
            if file.is_file() and not file.name.startswith(".gitkeep"):
                os.remove(file.path)
        logs.log.info("Document loading complete; removed local files.")

###################################
#
# Create Document Index
#
###################################

@st.cache_resource(show_spinner=False)
def create_index(_documents):
    """
    Creates a VectorStoreIndex from provided documents using global Settings.embed_model.
    """
    try:
        index = VectorStoreIndex.from_documents(
            documents=_documents,
            show_progress=True,
        )
        logs.log.info("Index created successfully.")
        return index
    except Exception as err:
        logs.log.error(f"Index creation failed: {err}")
        raise

###################################
#
# Create Query Engine
#
###################################

def create_query_engine(_documents):
    """
    Builds and caches a QueryEngine from the index.
    """
    try:
        index = create_index(_documents)
        query_engine = index.as_query_engine(
            similarity_top_k=st.session_state.get("top_k", 5),
            response_mode=st.session_state.get("chat_mode", "compact"),
            streaming=True,
        )
        st.session_state["query_engine"] = query_engine
        logs.log.info("Query Engine created successfully.")
        return query_engine
    except Exception as e:
        logs.log.error(f"Error creating Query Engine: {e}")
        raise
