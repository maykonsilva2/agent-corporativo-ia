import os
import tempfile # Used to create temporary files (In this case, the temporary files for the Large Language Models (LLMs) to process.)
from pydantic.v1 import tools
import streamlit as st
from dotenv import load_dotenv # Used to load environment variables from .env file (in this case, the API key for the LLMs)
from pydantic import SecretStr # Used to hide sensitive information (in this case, the API key for the LLMs)

from langchain_core.tools import retriever, tool
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
# 0. Streamlit PAGE CONFIGURATION (must be the first Streamlit command)
# This command configures how the browser tab displays your application and sets global layout rules.
# It does not render content inside the web page body itself.
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
# 1. LLM Configuration with Fallback
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
# 2. STREAMLIT INTERFACE
# This block renders actual visual elements and interactive widgets inside the body of the web page that users see and interact with.
# ==========================================

st.title("🤖 Agente Corporativo IA")

# st.write("Faça o upload de um documento (.pdf, .txt, .csv ou .docx) e converse com ele.")
# Replace st.write with st.info.
st.info("""
**ℹ️ Como usar esta ferramenta:**

1. **Upload:** Carregue um documento da empresa nos formatos `.pdf`, `.txt`, `.csv` ou `.docx` no botão abaixo.
2. **Processamento:** Aguarde alguns segundos enquanto a IA lê e indexa o conteúdo do arquivo.
3. **Conversa:** Faça perguntas sobre o documento no campo de chat. 

**Regras do Agente:**
- As respostas serão baseadas **exclusivamente** no documento enviado.
- Sempre citará a fonte da informação (página ou linha).
- Se a informação não existir no documento, a IA informará que não foi encontrada (não há invenção de respostas).
""")

# This widget is used here in code because first we need a place where the user can upload a document. If there is no such place, the user will not be able to upload a document.
uploaded_file = st.file_uploader("Escolha um arquivo", type=['pdf', 'txt', 'csv', 'docx'])

# Initialize the Vector Database memory
# "If this is the very first time the app is loading, create an empty slot called vector_db and set it to None."
# This is necessary why: The Streamlit Rule: Every time a user interacts with a widget (clicks a button, types a chat message, uploads a file)
# Streamlit re-runs your entire Python script from top to bottom.
# When the user uploads a file, the app processes it and store the result here. When the user types a chat message 5 seconds later, Streamlit reruns the scrips. Because of this `if` statement, it doensn't overwite `vector_db` back to `None`. It keeps the processed PDf in memory.

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None


# Initialize the Chat History memory
# "If this is the very first time the app is loading, create an empty slot called vector_db and set it to None."
# Why it's crucial: This stores the conversation history [{role: "user", content: "hi"}, {role: "assistant", content: "hello!"}]. Without this, every time the user sends a new message, the previous messages would disappear from the screen, and the LLM would forget what was just said.

if "messages" not in st.session_state:
    st.session_state.messages = []


# What it means: "If this is the first time the app is loading, create a slot called last_file_name and set it to None."
# Why it's crucial: Look at block in your code:

# if uploaded_file.name != st.session_state.last_file_name:
#    st.session_state.vector_db = None
#    st.session_state.messages = []
#    st.session_state.last_file_name = uploaded_file.name

#This checks if the user uploaded a new file. If they did, it resets the vectors and clears the chat history, because the old chat no longer makes sense with the new document. Without storing last_file_name, the app wouldn't know whether it's dealing with the same file or a new one.

if "last_file_name" not in st.session_state:
    st.session_state.last_file_name = None

# ==========================================
# 3. DOCUMENT PROCESSING (RAG)
# Question about the difference:
# Use `!=` to compare Values(Strings, Numbers, Lists, Dicts).
# Use `is not` to compare Objects in Memory(None, True, False)
# Best Practice in Python (PEP 8 standard), you should always use `is` or `is not` when checking against `None`. It is faster and avoids unexpected behavior.
# ==========================================

# When the page loads, the st.file_uploader is empty (value is None). This line checks if the user has actually selected a file. If they haven't, the code skips this entire block. If they have, it enters the block.
if uploaded_file is not None:
    # Reset when loading a new  file
    if uploaded_file.name != st.session_state.last_file_name:
        st.session_state.vector_db = None # Clears old document memory
        st.session_state.messages = [] # Clears old chat history
        st.session_state.last_file_name = uploaded_file.name # Rememberes the new file

    # If the database is None, it means the file hasn't been processed yet. If the database already exists (because the user just sent a chat message, causing a rerun), the code skips the block completely, saving time and API costs.
    
    if st.session_state.vector_db is None:
        # Extract the File Extension
        # What it does: Splits the file name by . and extracts the last part (e.g., "report.pdf" -> "pdf"), converting it to lowercase.

        # Question dubt: In Python, [-1] is used to select the last element of a list.When a filename is split by a period using .split('.'), it creates a list of all the parts of the name. Since the file extension always comes after the final period, accessing the index [-1] guarantees you get the actual extension, even if the file name contains multiple periods (e.g., document.backup.v2.pdf).

        file_extension = uploaded_file.name.split('.')[-1].lower()

        # Create a Temporary Disk File
        # Streamlit keeps uploaded files in RAM (`BytesIO` buffer). However, standard file loaders (like `PyPDFLoader`) require an actual physical file path on your hard drive.
        # `delete=False`: Stops Python from automatically deleting the temporary file the moment the with block closes, allowing LangChain to open and read it.
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp:
            tmp.write(uploaded_file.getvalue()) # This line writes the actual contents of the file into the temporary file.
            tmp_path = tmp.name # This line assigns the file path of the temporary file to the variable 'tmp_pth'.

        
        with st.spinner("Processando o documento..."):
            # Load and Parse the File
            # Fallback:  Any other extension defaults to TextLoader
            # Easy to Extend: Adding support for a new file type (like JSON or Markdown) just requires adding one key-value pair to the loaders dictionary.
            try:
                loaders = {'pdf': PyPDFLoader, 'csv': CSVLoader, 'docx': Docx2txtLoader}

                # Is actually doing two steps in one
                # Finding the Class (The Lookup)

                # `loaders.get(extensao, TextLoader)` -> It gets the loader class: If extensao is 'pdf', it evaluates to the class `PyPDFLoader`.
                #  It falls back to a default: .get(key, default) returns TextLoader as a safe fallback if extensao isn't in the dictionary (e.g., for .txt, .py, or .md files).

                # Instantiating the Class (The (tmp_path) Part) -> Once the dictionary resolves to a class (e.g., PyPDFLoader), Python replaces that part of the code and executes it like this: 
                # If extensao == 'pdf':
                # loader = PyPDFLoader(tmp_path)

                # If extensao is unknown (e.g., 'txt'):
                # loader = TextLoader(tmp_path)
                loader = loaders.get(file_extension, TextLoader)(tmp_path)
                docs = loader.load()

                chunks = RecursiveCharacterTextSplitter( chunk_size=1000, chunk_overlap=200).split_documents(docs)

                # It converts those chunks into mathematical vectors (using the embeddings model you configured earlier) and stores them in FAISS (a fast vector database).
                # It saves the FAISS database into st.session_state so it survives future Streamlit reruns.
                st.session_state.vector_db = FAISS.from_documents(chunks, embeddings)
                st.success(f"✅ Documento '{uploaded_file.name}' processado! {len(chunks)} trechos indexados. ")
            except Exception as e:
                st.error(f"Erro ao processar: {e}")
            finally:
                # Run to delete the temporary file from your disk after processing finishes, preventing temporary file memory leaks on your machine.
                os.remove(tmp_path)

# ==========================================
# 4. AGENT TOOL (RAG Search)
# ==========================================

@tool
def buscar_no_documento(pergunta:str) -> str:
    """Searches the uploaded document to answer the user's question. Always cite the source."""
    if st.session_state.vector_db is None:
        return "Nenhum documento foi carregado ainda."

    # `st.session_state.vector_db`
    # This is your FAISS Vector Store stored in Streamlit's session memory. 
    # It holds all the text chunks from the file you uploaded, converted into mathematical numbers (embeddings) so Python can understand their meaning.

    # `.as_retriever(...)`
    # A Vector Store is just a database that holds information. The .as_retriever() method converts that raw database into a searchable tool that LangChain can plug directly into your AI workflow.

    # search_kwargs={"k": 3}
    # search_kwargs stands for "search keyword arguments"
    # If k=1: It returns only the top 1 most similar text chunk. (Fast, but might miss context).
    #If k=3: It returns the top 3 most similar text chunks. (This is the sweet spot—it gives the LLM enough context to answer accurately without being too expensive).
    #If k=10: It returns 10 chunks (might overwhelm the AI with too much irrelevant text).

    retriever = st.session_state.vector_db.as_retriever(search_kwargs={"k": 3})

    docs_encontrados = retriever.invoke(pergunta)

    if not docs_encontrados:
        return "Nenhuma informação encontrada no documento para esta pergunta."
    
    resultado = ""

    # Cite the source
    # REMEMBER: `enumerate()` allows you to loop over a list(or any iterable) and get both the item itself AND its position(index) at the same time
    # The start parameter -> `enumerate(..., start=1)` tells Python to start counting from 1 instead of the default 0.

    # What is doc.metadata -> In LangChain, every chunk of text loaded from a PDF, CSV, or TXT has two main parts:
    #  1. doc.page_content: The actual text
    #  2. doc.metadata: A Python dictionary containing info about where the text came from.
    # doc.metadata: A Python dictionary containing info about where the text came from.

    # Why the second parameter? (The .get() method)
    # In Python, if you try to get a key from a dictionary that doesn't exist, the code crashes immediately with a KeyError.
    # To prevent this, we use the dictionary .get('key', 'default_value'). If it doesn't find the key, it returns the 'default_value' you provided instead of crashing.
    
    for i, doc in enumerate(docs_encontrados):
        fonte = doc.metadata.get('source', 'documento desconhecido') # Try to get 'source', if fails, return 'documento'
        pagina = doc.metadata.get('page', doc.metadata.get('row','N/A')) # Try to get 'page', if fails, try 'row', if fails, return 'N/A'
        
        resultado += f"\n[FONTE {i+1}: {fonte} - Página/Linha {pagina}\n{doc.page_content}\n"

    return resultado

# =========================================================
# 5. LEAD AGENT (LangGraph)
# =========================================================

system_prompt = """Você é um assistente corporativo especializado em análise de documentos internos.
Responda SEMPRE em Português do Brasil (pt-BR).

REGRAS OBRIGATÓRIAS:
1. Use a ferramenta `buscar_no_documento` para TODA pergunta do usuário.
2. Baseie sua resposta APENAS nos trechos retornados pela ferramenta.
3. Se não encontrar a informação, responda: "A informação solicitada não consta no documento fornecido."
4. Cite SEMPRE a fonte ao final da resposta. Exemplo: "Fonte: Página 2 do documento."
5. Jamais invente informações.
"""

# Deprecated
#  agente = create_react_agent(model=llm, tools=[buscar_no_documento], prompt=system_prompt)
tools=[buscar_no_documento]
agente = create_agent(model=llm, tools=tools, system_prompt=system_prompt)

# Doubt How Work the citation FONTE:
# ┌─────────────────────────────────────────────────────────────────┐
# │ 1. User asks a question                                         │
# └─────────────────────────────────────────────────────────────────┘
#                               ↓
# ┌─────────────────────────────────────────────────────────────────┐
# │ 2. LLM reads the system_prompt → sees rule "use the tool"       │
# │    (system_prompt in action)                                    │
# └─────────────────────────────────────────────────────────────────┘
#                              ↓
# ┌─────────────────────────────────────────────────────────────────┐
# │ 3. LLM decides to CALL the `buscar_no_documento` tool           │
# └─────────────────────────────────────────────────────────────────┘
#                              ↓
# ┌─────────────────────────────────────────────────────────────────┐
# │ 4. The TOOL (Python code) RUNS on the server:                   │
# │    - Retrieves documents from FAISS                             │
# │    - Reads doc.metadata.get('source')   ← FILE NAME             │
# │    - Reads doc.metadata.get('page')     ← PAGE NUMBER           │
# │    - Assembles the string: [SOURCE 1: file.pdf — Page 2]        │
# │    - Returns all this to the LLM                                │
# └─────────────────────────────────────────────────────────────────┘
#                              ↓
# ┌─────────────────────────────────────────────────────────────────┐
# │ 5. LLM receives the returned text (with the [SOURCE N] tags)    │
# └─────────────────────────────────────────────────────────────────┘
#                              ↓
# ┌─────────────────────────────────────────────────────────────────┐
# │ 6. LLM reads the system_prompt again → sees rule "4 .cite the source"│
# │    (system_prompt in action again)                              │
# └─────────────────────────────────────────────────────────────────┘
#                              ↓
# ┌─────────────────────────────────────────────────────────────────┐
# │ 7. LLM writes the final answer INCLUDING the source citation    │
# └─────────────────────────────────────────────────────────────────┘

# ==========================================
# 6. CHAT INTERFACE WITH HISTORY
# ==========================================

if st.session_state.vector_db is not None:
    st.divider() # The st.divider() function draws a horizontal line across the page
    st.subheader("💬 Converse com o documento") # The st.subheader() function displays a sub-header

    # Show the entire conversation history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): # The st.chat_message() function displays a chat message
            st.markdown(msg["content"]) # The st.markdown() function displays markdown formatted text

    # The `:=` operator assigns the value returned by st.chat_input() to the variable and also checks if it has a value (i.e., if it's not empty). If it does, the code inside the if block is executed.
    if pergunta := st.chat_input("Faça uma pergunta sobre o documento..."):
        st.session_state.messages.append({"role": "user", "content": pergunta}) # `append()` adds the user's message to the session state

        # Display the user's message immediately, without waiting for the agent to respond
        with st.chat_message("user"):
            st.markdown(pergunta)

        with st.chat_message("assistant"):
            with st.spinner("Analisando..."):
                try:
                    resposta = agente.invoke({"messages": st.session_state.messages})
                    
                    resposta_final = resposta['messages'][-1].content # The `[-1]` operator accesses the last element of the list

                    st.markdown(resposta_final)

                    st.session_state.messages.append({"role": "assistant", "content": resposta_final}) # `append()` adds the assistant's message to the session state

                except Exception as e:
                    st.error(f"Erro ao analisar documento: {e}")
else:
    st.info("⬆️ Faça o upload de um documento para começar a conversar.")