# Guia de Instalação e Execução

Este guia descreve como configurar e rodar o pipeline RAG do Assistente Virtual do SEI localmente.

---

## 📋 Pré-requisitos

1. **Python 3.9** (ou superior) instalado.
2. **Ollama** instalado e rodando em segundo plano (se optar pelo backend padrão).
3. **PowerShell** ou **Prompt de Comando (CMD)** com permissões adequadas.

---

## 🛠️ Passo a Passo de Instalação

### 1. Criar e Ativar o Ambiente Virtual (venv)
Na pasta raiz do projeto (`e:\RAG_LLM`), execute:

```powershell
# Criar o ambiente virtual (caso não exista)
python -m venv venv

# Ativar no Windows (PowerShell)
.\venv\Scripts\activate

# Ativar no Windows (CMD)
.\venv\Scripts\activate.bat
```

### 2. Instalar Dependências
Com o ambiente virtual ativado, instale todas as bibliotecas necessárias:

```powershell
pip install -r requirements.txt
```

---

## ⚙️ Configuração das Variáveis de Ambiente (`.env.qwen`)

O arquivo `.env.qwen` centraliza as configurações do modelo Qwen. Verifique ou ajuste os seguintes parâmetros nele:

```ini
# Diretórios
DOCS_DIRECTORY="./docs"
PERSIST_DIRECTORY_QWEN="./chroma_db_qwen"

# Divisão de texto
CHUNK_SIZE=500
CHUNK_OVERLAP=50
SEARCH_K=5

# Configurações de Embedding
USE_HUGGINGFACE_EMBEDDINGS=true
EMBEDDING_MODEL_QWEN="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# Caso USE_HUGGINGFACE_EMBEDDINGS=false:
EMBEDDING_MODEL_OLLAMA="nomic-embed-text"
OLLAMA_BASE_URL="http://localhost:11434"

# Escolha do Backend da LLM
# Modo 1: Ollama
USE_LLAMA_SERVER=false
OLLAMA_LLM_MODEL="qwen3:8b"

# Modo 2: Llama-server (Vulkan / RX 580)
# USE_LLAMA_SERVER=true
# LLAMA_SERVER_URL="http://localhost:8080/v1"
# LLAMA_SERVER_MODEL="qwen3-8b"

LLM_TEMPERATURE=0.1
```

---

## 🚀 Execução do Sistema

### Passo 1: Preparar os Modelos no Ollama (Se usando Modo 1)
Certifique-se de que o Ollama está ativo e baixe os modelos:
```bash
ollama pull qwen3:8b
ollama pull nomic-embed-text
```

---

### Passo 2: Ingestão de Documentos (Indexação)
Coloque os manuais oficiais em PDF ou texto limpo dentro da pasta `./docs`. Depois, execute a indexação.

> [!IMPORTANT]
> **Antes de rodar a ingestão:** Se a interface Streamlit estiver rodando em segundo plano, **feche-a** (pressione `Ctrl + C` no terminal dela). Caso contrário, o banco de dados estará bloqueado e ocorrerá um `PermissionError`.

Com o Streamlit fechado, execute:
```powershell
python rag_qwen.py --ingest
```
Este comando carregará os documentos, dividirá em pedaços (chunks), gerará os embeddings e salvará em `./chroma_db_qwen`.

---

### Passo 3: Iniciar a Interface Streamlit (Frontend Web)
Para rodar o assistente com a interface web moderna e personalizada:
```powershell
streamlit run app.py
```
A página abrirá automaticamente no navegador em:
- **Local:** [http://localhost:8501](http://localhost:8501)

---

### Passo 4: Executar via Linha de Comando (CLI Alternativo)
Se preferir interagir diretamente pelo terminal de texto clássico:
```powershell
python rag_qwen.py
```
Digite sua pergunta no terminal e pressione `Enter`. Digite `sair` para encerrar.
