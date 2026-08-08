# 🤖 Corporate AI Agent

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.42%2B-FF4B4B.svg)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/langchain-1.3%2B-1C3C3C.svg)](https://www.langchain.com/)

🌐 **Languages / Idiomas**: [ 🇺🇸 English ](README.md) | [ 🇧🇷 Português (PT-BR) ](README_pt.md)

---

## Project Description

**Corporate AI Agent** is an intelligent Retrieval-Augmented Generation (RAG) assistant designed for corporate environments. Built with Python, Streamlit, and LangChain, it enables team members and managers to upload internal company documents (`.pdf`, `.txt`, `.csv`, `.docx`) and query them interactively using natural language.

### Key Capabilities
- 📄 **Multi-Format Support**: Reads PDF, Text, CSV, and DOCX files.
- 🎯 **Strict Context Adherence**: Answers exclusively based on the provided document without inventing information (zero hallucination design).
- 📍 **Source Attribution**: Automatically cites specific page or line numbers for every answer.
- ⚡ **Multi-LLM Fallback Architecture**: Seamlessly shifts between OpenRouter (GPT-4o mini), Groq (GPT-OSS models), and Google Gemini (Gemini 3.6 Flash) to guarantee high availability and low operational costs.
- 💬 **Interactive Chat Interface**: Maintains context within active sessions and automatically clears history when new documents are uploaded.

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
                  | OpenAI Embeddings     |
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
                  +-----------------------+
```

### Workflow Steps
1. **Document Loading**: The uploaded document is parsed into text using specialized loaders (`PyPDFLoader`, `TextLoader`, `CSVLoader`, `Docx2txtLoader`).
2. **Text Chunking**: Document content is split into manageable chunks using `RecursiveCharacterTextSplitter` (chunk size: 1000 characters, overlap: 200 characters).
3. **Vector Indexing**: Text chunks are embedded via `OpenAIEmbeddings` and stored in a high-performance in-memory `FAISS` vector database.
4. **Agent Retrieval & Generation**: When a question is submitted, the LangChain agent invokes a custom tool (`buscar_no_documento`) that performs semantic similarity search against FAISS, retrieves the top 3 relevant chunks, and synthesizes a verified answer with exact citations.

---

## Technologies Used

- **Language**: Python 3.10+
- **Frontend / Web UI**: [Streamlit](https://streamlit.io/)
- **Orchestration Framework**: [LangChain](https://www.langchain.com/) / [LangChain Core](https://python.langchain.com/)
- **Vector Database**: [FAISS (Facebook AI Similarity Search)](https://github.com/facebookresearch/faiss)
- **Embeddings Provider**: [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)
- **LLM Integrations**:
  - [OpenRouter](https://openrouter.ai/) (`openai/gpt-4o-mini`) - Primary Model
  - [Groq](https://groq.com/) (`openai/gpt-oss-120b`, `openai/gpt-oss-20b`) - High-speed Fallback
  - [Google Generative AI](https://ai.google.dev/) (`gemini-3.6-flash`) - Multi-tier Fallback
- **Document Parsers**: `pypdf`, `docx2txt`, `python-dotenv`

---

## Installation Instructions

### Prerequisites
- Python 3.10 or higher installed.
- Git installed.
- API Keys for OpenAI (for embeddings) and at least one LLM provider (OpenRouter, Groq, or Google Gemini).

### Step-by-Step Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/agente-corporativo-ia.git
   cd agente-corporativo-ia
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
   Open `.env` in your text editor and add your API keys:
   ```env
   # Required for embeddings
   OPENAI_API_KEY=your_openai_api_key_here

   # Primary LLM Provider
   OPENROUTER_API_KEY=your_openrouter_api_key_here

   # Fallback LLM Providers (Optional but recommended)
   GROQ_API_KEY=your_groq_api_key_here
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

5. **Run the Streamlit Application:**
   ```bash
   streamlit run app.py
   ```
   The application will automatically open in your web browser at `http://localhost:8501`.

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
