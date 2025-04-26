import streamlit as st

import utils.logs as logs

from utils.ollama import get_models


def set_initial_state():

    ###########
    # General #
    ###########

    if "sidebar_state" not in st.session_state:
        st.session_state["sidebar_state"] = "expanded"

    if "ollama_endpoint" not in st.session_state:
        st.session_state["ollama_endpoint"] = "http://localhost:11434"
        print("ollama_endpoint", st.session_state["ollama_endpoint"])
        print("state", st.session_state)

    if "embedding_model" not in st.session_state:
        st.session_state["embedding_model"] = "Default (bge-large-en-v1.5)"

    if "ollama_models" not in st.session_state:
        try:
            models = get_models()
            st.session_state["ollama_models"] = models
        except Exception:
            st.session_state["ollama_models"] = []
            pass

    if "selected_model" not in st.session_state:
        try:
            if "llama3:8b" in st.session_state["ollama_models"]:
                st.session_state["selected_model"] = (
                    "llama3:8b"  # Default to llama3:8b on initial load
                )
            elif "mistral:7b-instruct-v0.3-q8_0" in st.session_state["ollama_models"]:
                st.session_state["selected_model"] = (
                    "mistral:7b-instruct-v0.3-q8_0"  # Default to mistral:7b-instruct-v0.3-q8_0 on initial load
                )
            elif "deepseek-r1:7B" in st.session_state["ollama_models"]:
                st.session_state["selected_model"] = (
                    "deepseek-r1:7B"  # Default to deepseek-r1:7B on initial load
                )
            elif "phi4:latest" in st.session_state["ollama_models"]:
                st.session_state["selected_model"] = (
                    "phi4:latest"  # Default to phi4:latest on initial load
                )
            else:
                st.session_state["selected_model"] = st.session_state["ollama_models"][
                    0
                ]  # If llama2:7b is not present, select the first model available
        except Exception:
            st.session_state["selected_model"] = None
            pass

    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {
                "role": "assistant",
                "content": "Welcome to Local RAG! To begin, please either import some files or ingest a GitHub repo. Once you've completed those steps, we can continue the conversation and explore how I can assist you further.",
            }
        ]

    ################################
    #  Files, Documents & Websites #
    ################################

    if "file_list" not in st.session_state:
        st.session_state["file_list"] = []

    if "github_repo" not in st.session_state:
        st.session_state["github_repo"] = None

    if "websites" not in st.session_state:
        st.session_state["websites"] = []

    ###############
    # Llama-Index #
    ###############

    if "llm" not in st.session_state:
        st.session_state["llm"] = None

    if "documents" not in st.session_state:
        st.session_state["documents"] = None

    if "query_engine" not in st.session_state:
        st.session_state["query_engine"] = None

    if "chat_mode" not in st.session_state:
        st.session_state["chat_mode"] = "compact"

    #####################
    # Advanced Settings #
    #####################

    if "advanced" not in st.session_state:
        st.session_state["advanced"] = False

    if "system_prompt" not in st.session_state:
        st.session_state["system_prompt"] = (
            """
                You are a highly advanced virtual assistant specialized in assisting users with answering questions from a diverse array of documents.
                Your strength lies in addressing intricate or nuanced questions based on the content of the provided documents.
                
                Your capabilities include:

                - Understanding and contextualizing complex documents.
                - Extracting and synthesizing key information.
                - Providing detailed explanations and summaries.
                - Answering specific queries with precise and relevant information.
                - Offering insights and recommendations based on document analysis.
                - Selecting the best answers for multiple-choice questions
                
                **INSTRUCTIONS** - Your goal is to THOROUGHLY understand the documents and enable users to ask ANY nuanced or direct question from the documents information.
                ENSURE you respond BRIEFLY (in 3-4 sentences) and accurately while using information from the documents as a primary source ALWAYS. 
                REMEMBER you MUST WITHOUT FAIL cite your document source using the document name
                When asked to solve a multiple choice question with answer choices, you MUST identify AT LEAST one correct answer and AT MOST four correct answers. 
            
            """
        )

    if "top_k" not in st.session_state:
        st.session_state["top_k"] = 5

    if "embedding_model" not in st.session_state:
        st.session_state["embedding_model"] = None

    if "other_embedding_model" not in st.session_state:
        st.session_state["other_embedding_model"] = None

    if "chunk_size" not in st.session_state:
        st.session_state["chunk_size"] = 1024

    if "chunk_overlap" not in st.session_state:
        st.session_state["chunk_overlap"] = 200
