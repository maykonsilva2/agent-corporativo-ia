# 🤖 Corporate AI Agent

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.60%2B-FF4B4B.svg)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/langchain-1.3%2B-1C3C3C.svg)](https://www.langchain.com/)

🌐 **Languages / Idiomas**: [ 🇺🇸 English ](README.md) | [ 🇧🇷 Português (PT-BR) ](README_pt.md)

---

## Project Description

**Corporate AI Agent** is an intelligent Retrieval-Augmented Generation (RAG) assistant designed for corporate environments. Built with Python, Streamlit, and LangChain, it enables team members and managers to upload internal company documents (`.pdf`, `.txt`, `.csv`, `.docx`) and query them interactively using natural language.

### Key Capabilities
- 📄 **Multi-Format Support**: Reads PDF, Text, CSV, and DOCX files — single or multiple at once.
- 🎯 **Strict Context Adherence**: Answers exclusively based on the provided documents without inventing information (zero hallucination design).
- 📍 **Source Attribution**: Automatically cites specific page or line numbers for every answer.
- ⚡ **Multi-LLM Fallback Architecture**: Seamlessly shifts between OpenRouter (GPT-4o mini), Groq (GPT-OSS models), and Google Gemini (Gemini 3.6 Flash) to guarantee high availability and low operational costs.
- 💬 **Persistent Chat History (SQLite)**: Conversations and messages are saved in a local SQLite database (`chat_history.db`). Reloading the page or reopening a conversation restores the full history.
- 🕘 **Sidebar Conversation History**: Lists previous conversations with load/delete buttons. Reopening a conversation that used test documents automatically reindexes them.
- 🧪 **Test Documents**: Sample files in the `docs/` folder can be selected from the sidebar — no upload required to try the app.
- 🔍 **CSV Delimiter Auto-Detection**: Uses `csv.Sniffer` to handle both `,` (international) and `;` (Brazilian Excel) delimiters automatically.
- ⚡ **LLM Response Caching**: `InMemoryCache` avoids redundant LLM calls for identical prompts.

---

## Architecture

The system utilizes a modular RAG (Retrieval-Augmented Generation) pipeline orchestrated by LangChain and Streamlit:

```
                  +-----------------------+
                  |  Document Upload      |
                  | (.pdf/.txt/.csv/.docx)|
                  +-----------+-----------+
                              |
                              v
                  +-----------------------+
                  |   Document Loader     |
                  | (PyPDF/Text/CSV/Docx) |
                  +-----------+-----------+
                              |
                              v
                  +-----------------------+
                  |  Text Splitter        |
                  | Chunk: 1000 | Overlap: 200|
                  +-----------+-----------+
                              |
                              v
                  +-----------------------+
                  | HuggingFace Embeddings|
                  | (all-MiniLM-L6-v2)    |
                  | (local, free, no key) |
                  +-----------+-----------+
                              |
                              v
                  +-----------------------+
                  | Vector Store (FAISS)  |
                  +-----------+-----------+
                              |
  +------------------+         |
  |   User Query     |-------->+
  +------------------+         |
                              v
                  +-----------------------+
                  | Custom RAG Tool       |
                  | (buscar_no_documento) |
                  +-----------+-----------+
                              |
                              v
                  +-----------------------+
                  | LangChain React Agent |
                  +-----------+-----------+
                              |
            +-----------------+-----------------+
            |                 |                 |
            v                 v                 v
     +--------------+  +--------------+  +--------------+
     | OpenRouter   |  | Groq         |  | Google Gemini|
     | (Primary)    |->| (Fallback 1) |->| (Fallback 2) |
     +--------------+  +--------------+  +--------------+
                              |
                              v
                  +-----------------------+
                  | Answer + Citation     |
                  +-----------+-----------+
                              |
                              v
                  +-----------------------+
                  | SQLite Persistence    |
                  | (chat_history.db)     |
                  +-----------------------+
```

### Workflow Steps
1. **Document Loading**: Uploaded documents are parsed into text using specialized loaders (`PyPDFLoader`, `TextLoader`, `CSVLoader`, `Docx2txtLoader`). Multiple files are processed in a loop.
2. **Text Chunking**: Document content is split into manageable chunks using `RecursiveCharacterTextSplitter` (chunk size: 1000 characters, overlap: 200 characters).
3. **Vector Indexing**: Text chunks are embedded via **HuggingFace Embeddings** (`sentence-transformers/all-MiniLM-L6-v2`) — a local, free model requiring no API key — and stored in an in-memory `FAISS` vector database. The embedding model (~90 MB) is downloaded once on first run, then cached.
4. **Agent Retrieval & Generation**: When a question is submitted, the LangChain agent invokes a custom tool (`buscar_no_documento`) that performs semantic similarity search against FAISS, retrieves the top 3 relevant chunks, and synthesizes a verified answer with exact citations.
5. **SQLite Persistence**: Every user question and agent answer is saved to `chat_history.db`. Conversations can be reopened from the sidebar; those that used test documents from `docs/` are automatically reindexed.

---

## Technologies Used

- **Language**: Python 3.10+
- **Frontend / Web UI**: [Streamlit](https://streamlit.io/)
- **Orchestration Framework**: [LangChain](https://www.langchain.com/) / [LangChain Core](https://python.langchain.com/)
- **Vector Database**: [FAISS (Facebook AI Similarity Search)](https://github.com/facebookresearch/faiss)
- **Embeddings Provider**: [HuggingFace Embeddings](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (`all-MiniLM-L6-v2`) — local, free, no API key required
- **Persistence**: `sqlite3` (Python built-in) for chat history — no external database needed
- **LLM Integrations** (at least one required; the rest are automatic fallbacks):
  - [OpenRouter](https://openrouter.ai/) (`openai/gpt-4o-mini`) — Primary Model
  - [Groq](https://groq.com/) (`openai/gpt-oss-120b`, `openai/gpt-oss-20b`) — High-speed Fallback
  - [Google Generative AI](https://ai.google.dev/) (`gemini-3.6-flash`) — Multi-tier Fallback
- **Document Parsers**: `pypdf`, `docx2txt`, `python-dotenv`

---

## Installation Instructions (Local)

### Prerequisites
- Python 3.10 or higher installed.
- Git installed.
- At least one LLM API key (OpenRouter, Groq, or Google Gemini). **No OpenAI key is needed** — embeddings run locally via HuggingFace.

### Step-by-Step Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/agente-corporativo-ia.git
   cd agent-corporativo-ia
   ```

2. **Create and activate a virtual environment:**
   - **Linux / macOS:**
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```
   - **Windows:**
     ```cmd
     python -m venv .venv
     .venv\Scripts\activate
     ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables:**
   Copy the example environment file:
   ```bash
   cp .env.example .env
   ```
   Open `.env` in your text editor and add your API keys (at least one):
   ```env
   # Primary LLM Provider
   OPENROUTER_API_KEY=your_openrouter_api_key_here

   # Fallback LLM Providers (Optional but recommended)
   GROQ_API_KEY=your_groq_api_key_here
   GEMINI_API_KEY=your_gemini_api_key_here
   ```
   > ℹ️ **No `OPENAI_API_KEY` needed.** Embeddings use HuggingFace's `all-MiniLM-L6-v2`, which runs locally and is free.

5. **Run the Streamlit Application:**
   ```bash
   streamlit run app.py
   ```
   The application will automatically open in your web browser at `http://localhost:8501`.

   On first run, the HuggingFace embedding model (~90 MB) downloads automatically and is cached for subsequent runs.

---

## Deploy to Streamlit Cloud

The app is designed to work on [Streamlit Community Cloud](https://streamlit.io/cloud) **without any code changes**. The `get_secret()` helper reads from `st.secrets` (Cloud) first, then falls back to `os.environ` (local `.env`).

### Step-by-Step Deployment

1. **Push your code** to a public GitHub repository.

2. **Go to** [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.

3. **Create a new app:**
   - **Repository**: `link to your public GitHub repository`
   - **Branch**: `branch name`
   - **Main file path**: `main application file name`

4. **Configure Secrets (API keys):**
   - Click **Settings ⚙️** → **Secrets**
   - Paste your keys in **TOML format** (no `.env` file on Cloud):
      ```toml
      OPENROUTER_API_KEY = "your_api_key"
      GROQ_API_KEY = "your_api_key"
      GEMINI_API_KEY = "your_api_key"
      ```
   - At least **one** key is required; the rest serve as automatic fallbacks.

5. **Click Deploy.** The first deployment downloads the HuggingFace embedding model (~90 MB), so it may take a few minutes to start.

### How Secrets Work on Streamlit Cloud

| Local development | Streamlit Cloud |
|---|---|
| `.env` file loaded by `python-dotenv` | `st.secrets` configured in the dashboard |
| `os.environ` | `st.secrets` (TOML format) |
| Both are read by `get_secret()` in `app.py:45` | Both are read by `get_secret()` in `app.py:45` |

The `get_secret()` function transparently handles both environments:
```python
def get_secret(key: str) -> str | None:
    try:
        return st.secrets[key]       # Streamlit Cloud
    except (KeyError, FileNotFoundError):
        return os.getenv(key)        # Local .env
```

> ⚠️ **Never commit your `.env` file.** It is listed in `.gitignore`. On Streamlit Cloud, secrets are encrypted and stored separately from your repository.

---

## Example Questions

Below are example questions based on the sample documents located in the [`docs/`](docs/) directory:

1. **FAQ / Account Setup** (`docs/faq_suporte.txt`):
   > *"What is the step-by-step process to create an account at Nexolve, and how long does the free trial last?"*

2. **Pricing & Plans** (`docs/planos_preços.csv`):
   > *"What are the limits and prices for the Business plan compared to the Pro plan?"*

3. **Out-of-Scope Question** (Testing zero-hallucination):
   > *"What is the formula for calculating employee annual performance bonuses?"*

---

## Example Answers

### Answer to Question 1:
**User:** *"What is the step-by-step process to create an account at Nexolve, and how long does the free trial last?"*
**Agent:**
> To create an account at Nexolve:
> 1. Access https://app.nexolve.com.br.
> 2. Click on "Criar conta" (Create Account).
> 3. Fill in the form with your name, corporate email, and password.
> 4. Enter the confirmation code sent to your email.
>
> Once confirmed, your account will be activated on the Free plan with a **14-day trial of the Pro plan** (no credit card required).
>
> **Source:** Line 16-24 of document `docs/faq_suporte.txt`.

---

### Answer to Question 2:
**User:** *"What are the limits and prices for the Business plan compared to the Pro plan?"*
**Agent:**
> Based on the pricing schedule:
> - **Pro Plan:** Costs **R$ 89/user/month**, includes **10 users**, **50 GB storage**, **10,000 daily API calls**, 15 integrations, 99.5% uptime SLA, and a 14-day trial.
> - **Business Plan:** Costs **R$ 249/user/month**, includes **25 users**, **250 GB storage**, **100,000 daily API calls**, 30 integrations, 99.9% uptime SLA, and a 14-day trial.
>
> **Source:** Row 3 and Row 4 of document `docs/planos_preços.csv`.

---

### Answer to Question 3 (Out-of-Scope):
**User:** *"What is the formula for calculating employee annual performance bonuses?"*
**Agent:**
> A informação solicitada não consta no documento fornecido.
>
> *(The requested information is not present in the provided document.)*
>
> **Source:** Document search completed via `buscar_no_documento`.
