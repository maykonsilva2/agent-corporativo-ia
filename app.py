import os
import tempfile # Used to create temporary files (In this case, the temporary files for the Large Language Models (LLMs) to process.)
import streamlit as st
from dotenv import load_dotenv # Used to load environment variables from .env file (in this case, the API key for the LLMs)
from pydantic import SecretStr # Used to hide sensitive information (in this case, the API key for the LLMs)

from langchain_core.tools import tool
from langchain_core.caches import InMemoryCache # Used to cache the responses of the LLMs(in this case, the responses of the LLMs in the memory)
from langchain_core.globals import set_llm_cache # Used to set the cache for the LLMs(in this case, the cache for the responses of the LLMs)

from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter # Used to split the documents into smaller chunks (this is done because the LLMs have a limited context window)
from langchain_community.vectorstores import FAISS # Can use other vector stores like Chroma, Pinecone, LanceDB, Qdrant, etc.
from langchain.agents import create_agent


load_dotenv()

# ==========================================
# Streamlit PAGE CONFIGURATION (must be the first Streamlit command)
# ==========================================

st.set_page_config(
    page_title="Agente Corporativo",
    page_icon="🤖",
    layout="wide" # Set options: 'wide', 'centered'
)

# It is temporary; For production, use Redis or other distributed caches
# InMemoryCache stores data in computer's memory. If you stop your Streamlit app or terminal, the cache is completely wiped.
set_llm_cache(InMemoryCache()) # It is used to cache the responses of the LLMs(in this case, the responses of the LLMs in the memory)

# ==========================================
# LLM Configuration with Fallback
# LLM: Assembles a list of available models in order of priority.
# The first becomes the primary model; the rest become automatic fallbacks.

# Understand: `if key := os.getenv('OPENROUTER_API_KEY'):` 
# First, key receives the value. Second, Python checks if that value is valid (not empty). If it is valid, the code enters the if block and executes models.append().
# ==========================================

def build_available_llms() -> list:
    models = []

    if key := os.getenv('OPENROUTER_API_KEY'):
        models.append(
            ChatOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=SecretStr(key),
                model="openai/gpt-4o-mini",
                temperature=0
            )
        )
    
    if key := os.getenv('GROQ_API_KEY'):
        # llama-3.3-70b-versatile and llama-3.1-8b-instant have been discontinued
        # by Groq (shutdown ~Aug 16, 2026) — use the official replacements
        # URL: https://console.groq.com/docs/deprecations
        models.append(
            ChatGroq(
                api_key=SecretStr(key),
                model="openai/gpt-oss-120b",
                temperature=0
            )
        )
        
        models.append(
            ChatGroq(
                api_key=SecretStr(key),
                model="openai/gpt-oss-20b",
                temperature=0
            )
        )

    if key := os.getenv('GEMINI_API_KEY'):
        models.append(
            ChatGoogleGenerativeAI(
                api_key=SecretStr(key),
                model='gemini-3.6-flash',
                temperature=0
            )
        )

    return models

available_models = build_available_llms()

if not available_models:
    st.error("❌ Nenhuma chave de API de LLM encontrada. Defina pelo menos uma "
              "(OPENROUTER_API_KEY, GROQ_API_KEY ou GEMINI_API_KEY) no seu .env."
              "❌ No LLM API key found. Set at least one"
              "(OPENROUTER_API_KEY, GROQ_API_KEY, or GEMINI_API_KEY) in your .env."
              )
    st.stop()

llm = available_models[0]
if len(available_models) > 1:
    llm = llm.with_fallbacks(available_models[1:])


# Handle Embeddings (Note: OpenAI is required for embeddings in your current setup)
openai_key     = os.getenv("OPENAI_API_KEY")
if openai_key:
    embeddings = OpenAIEmbeddings(api_key=SecretStr(openai_key))
else:
    # If you want to be fully independent of OpenAI, you could use HuggingFaceEmbeddings here instead!
    st.error("❌ OPENAI_API_KEY is missing. The app needs it to process document embeddings.")
    st.stop()

# Optional:
# EMBEDDINGS: uses OpenAI if the key exists; otherwise, it falls back to
#    free local embeddings (HuggingFace) instead of crashing the entire app
#    or simply disabling RAG.
#    pip install langchain-huggingface sentence-transformers

# else:
#    from langchain_huggingface import HuggingFaceEmbeddings
#    st.sidebar.info("ℹ️ OPENAI_API_KEY ausente — usando embeddings locais "
#                     "(HuggingFace: all-MiniLM-L6-v2) para o RAG.")
#    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# ==========================================
# STREAMLIT INTERFACE
# ==========================================