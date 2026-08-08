# 🤖 Agente Corporativo IA

[![Versão Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.42%2B-FF4B4B.svg)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/langchain-1.3%2B-1C3C3C.svg)](https://www.langchain.com/)

🌐 **Idiomas / Languages**: [ 🇧🇷 Português (PT-BR) ](README_pt.md) | [ 🇺🇸 English ](README.md)

---

## Descrição do Projeto

O **Agente Corporativo IA** é um assistente inteligente baseado em RAG (Geração Aumentada por Recuperação) desenvolvido para ambientes corporativos. Construído com Python, Streamlit e LangChain, ele permite que colaboradores e gestores façam upload de documentos internos (`.pdf`, `.txt`, `.csv`, `.docx`) e realizem consultas interativas em linguagem natural.

### Principais Funcionalidades
- 📄 **Suporte Multiformato**: Leitura nativa de arquivos PDF, Texto simples, CSV e DOCX.
- 🎯 **Aderência Estrita ao Contexto**: Responde exclusivamente com base no documento fornecido, eliminando alucinações.
- 📍 **Citação de Fonte**: Cita automaticamente a página ou linha exata de onde a informação foi extraída.
- ⚡ **Arquitetura Multi-LLM com Fallback**: Alterna automaticamente entre OpenRouter (GPT-4o mini), Groq (modelos GPT-OSS) e Google Gemini (Gemini 3.6 Flash) para garantir alta disponibilidade e menor custo.
- 💬 **Interface de Chat Interativa**: Mantém histórico de conversas durante a sessão e limpa automaticamente a memória ao carregar um novo documento.

---

## Arquitetura

O sistema utiliza um pipeline de RAG orquestrado pelo LangChain e Streamlit:

```
                  +-----------------------+
                  | Upload de Documento   |
                  | (.pdf/.txt/.csv/.docx)|
                  +-----------+-----------+
                              |
                              v
                  +-----------------------+
                  |  Carregador de Docs   |
                  | (PyPDF/Text/CSV/Docx) |
                  +-----------+-----------+
                              |
                              v
                  +-----------------------+
                  |  Divisor de Texto     |
                  | Chunk: 1000 | Overlap: 200|
                  +-----------+-----------+
                              |
                              v
                  +-----------------------+
                  | Embeddings OpenAI     |
                  +-----------+-----------+
                              |
                              v
                  +-----------------------+
                  | Banco Vetorial(FAISS) |
                  +-----------+-----------+
                              |
 +------------------+         |
 | Pergunta Usuário |-------->+
 +------------------+         |
                              v
                  +-----------------------+
                  | Ferramenta RAG        |
                  | (buscar_no_documento) |
                  +-----------+-----------+
                              |
                              v
                  +-----------------------+
                  | Agente React LangChain|
                  +-----------+-----------+
                              |
            +-----------------+-----------------+
            |                 |                 |
            v                 v                 v
     +--------------+  +--------------+  +--------------+
     | OpenRouter   |  | Groq         |  | Google Gemini|
     | (Principal)  |->| (Fallback 1) |->| (Fallback 2) |
     +--------------+  +--------------+  +--------------+
                              |
                              v
                  +-----------------------+
                  | Resposta + Citação    |
                  +-----------------------+
```

### Etapas do Fluxo
1. **Carregamento**: O arquivo enviado é processado pelo carregador correspondente (`PyPDFLoader`, `TextLoader`, `CSVLoader`, `Docx2txtLoader`).
2. **Divisão (Chunking)**: O conteúdo é particionado em blocos de texto usando `RecursiveCharacterTextSplitter` (tamanho: 1000 caracteres, sobreposição: 200 caracteres).
3. **Indexação Vetorial**: Os trechos são convertidos em vetores via `OpenAIEmbeddings` e armazenados no banco de dados FAISS em memória.
4. **Recuperação e Resposta**: Ao receber uma dúvida, o Agente executa a ferramenta `buscar_no_documento`, consulta o FAISS recuperando os 3 trechos mais relevantes e gera uma resposta contextualizada contendo a citação da fonte.

---

## Tecnologias Utilizadas

- **Linguagem**: Python 3.10+
- **Interface Web**: [Streamlit](https://streamlit.io/)
- **Orquestração de IA**: [LangChain](https://www.langchain.com/) / [LangChain Core](https://python.langchain.com/)
- **Banco Vetorial**: [FAISS (Facebook AI Similarity Search)](https://github.com/facebookresearch/faiss)
- **Embeddings**: [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)
- **Integração com LLMs**:
  - [OpenRouter](https://openrouter.ai/) (`openai/gpt-4o-mini`) - Modelo Principal
  - [Groq](https://groq.com/) (`openai/gpt-oss-120b`, `openai/gpt-oss-20b`) - Fallback de Alta Velocidade
  - [Google Generative AI](https://ai.google.dev/) (`gemini-3.6-flash`) - Fallback de Segurança
- **Manipulação de Arquivos**: `pypdf`, `docx2txt`, `python-dotenv`

---

## Instruções de Instalação

### Pré-requisitos
- Python 3.10 ou superior instalado.
- Git instalado.
- Chave de API da OpenAI (para embeddings) e pelo menos uma chave de provedor LLM (OpenRouter, Groq ou Google Gemini).

### Passo a Passo de Configuração

1. **Clonar o repositório:**
   ```bash
   git clone https://github.com/SEU_USUARIO/agente-corporativo-ia.git
   cd agente-corporativo-ia
   ```

2. **Criar e ativar um ambiente virtual:**
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

3. **Instalar as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configurar as Variáveis de Ambiente:**
   Copie o modelo de arquivo `.env`:
   ```bash
   cp .env.example .env
   ```
   Abra o arquivo `.env` no seu editor de texto e insira suas chaves de API:
   ```env
   # Obrigatório para os embeddings
   OPENAI_API_KEY=sua_chave_openai_aqui

   # Provedor LLM Principal
   OPENROUTER_API_KEY=sua_chave_openrouter_aqui

   # Provedores LLM de Fallback (Opcionais, mas recomendados)
   GROQ_API_KEY=sua_chave_groq_aqui
   GEMINI_API_KEY=sua_chave_gemini_aqui
   ```

5. **Executar a aplicação Streamlit:**
   ```bash
   streamlit run app.py
   ```
   A aplicação será aberta no navegador em `http://localhost:8501`.

---

## Exemplos de Perguntas

Com base nos documentos demonstrativos da pasta [`docs/`](docs/):

1. **FAQ / Cadastro de Conta** (`docs/faq_suporte.txt`):
   > *"Como faço para criar uma conta na Nexolve e quanto tempo dura o período de testes?"*

2. **Planos e Preços** (`docs/planos_preços.csv`):
   > *"Quais são os limites e o valor do plano Business em comparação ao plano Pro?"*

3. **Pergunta Fora do Escopo** (Teste de prevenção a alucinações):
   > *"Qual é a fórmula de cálculo para o bônus anual de desempenho dos funcionários?"*

---

## Exemplos de Respostas

### Resposta à Pergunta 1:
**Usuário:** *"Como faço para criar uma conta na Nexolve e quanto tempo dura o período de testes?"*  
**Agente:**
> Para criar uma conta na Nexolve:
> 1. Acesse https://app.nexolve.com.br.
> 2. Clique em "Criar conta".
> 3. Preencha o formulário com nome, e-mail corporativo e senha.
> 4. Insira o código de confirmação recebido por e-mail.
> 
> Após a confirmação, sua conta estará ativa no plano Free com **14 dias de trial do plano Pro** (sem necessidade de cartão de crédito).
> 
> **Fonte:** Linhas 16 a 24 do documento `docs/faq_suporte.txt`.

---

### Resposta à Pergunta 2:
**Usuário:** *"Quais são os limites e o valor do plano Business em comparação ao plano Pro?"*  
**Agente:**
> Com base na tabela de preços:
> - **Plano Pro:** Custa **R$ 89/usuário/mês**, inclui **10 usuários**, **50 GB de armazenamento**, **10.000 requisições de API por dia**, 15 integrações e SLA de 99,5% de uptime.
> - **Plano Business:** Custa **R$ 249/usuário/mês**, inclui **25 usuários**, **250 GB de armazenamento**, **100.000 requisições de API por dia**, 30 integrações e SLA de 99,9% de uptime.
> 
> **Fonte:** Linha 3 (Pro) e Linha 4 (Business) do documento `docs/planos_preços.csv`.

---

### Resposta à Pergunta 3 (Fora do Escopo):
**Usuário:** *"Qual é a fórmula de cálculo para o bônus anual de desempenho dos funcionários?"*  
**Agente:**
> A informação solicitada não consta no documento fornecido.
> 
> **Fonte:** Busca realizada no documento via `buscar_no_documento`.
