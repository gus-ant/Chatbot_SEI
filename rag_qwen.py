"""
rag_qwen.py - Pipeline RAG com Qwen3 8B + nomic-embed-text + ChromaDB
======================================================================
Paralelo ao rag.py (Command R / OCC-RAG).
Usa ChromaDB separado (chroma_db_qwen) para não conflitar.

Dois modos de LLM (configurados no .env.qwen):
  USE_LLAMA_SERVER=false → Ollama com qwen3:8b (padrão Windows)
  USE_LLAMA_SERVER=true  → llama-server local na porta 8080 (Vulkan/RX 580)

Uso:
  python rag_qwen.py --ingest   # indexar documentos (rodar uma vez)
  python rag_qwen.py            # chatbot interativo
"""

import os
import sys
import argparse

# Garantir saida UTF-8 no Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
from dotenv import dotenv_values

# ──────────────────────────────────────────────────────────────────────────────
# 1. Carregar configurações do .env.qwen
# ──────────────────────────────────────────────────────────────────────────────
config = dotenv_values(".env.qwen")

DOCS_DIR         = config.get("DOCS_DIRECTORY",       "./docs")
PERSIST_DIR      = config.get("PERSIST_DIRECTORY_QWEN", "./chroma_db_qwen")
CHUNK_SIZE       = int(config.get("CHUNK_SIZE",       500))
CHUNK_OVERLAP    = int(config.get("CHUNK_OVERLAP",    50))
SEARCH_K         = int(config.get("SEARCH_K",         5))
USE_HF_EMBEDDINGS = config.get("USE_HUGGINGFACE_EMBEDDINGS", "true").lower() == "true"
EMBEDDING_MODEL_HF = config.get("EMBEDDING_MODEL_QWEN", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
EMBEDDING_MODEL_OLLAMA = config.get("EMBEDDING_MODEL_OLLAMA", "nomic-embed-text")
OLLAMA_BASE_URL  = config.get("OLLAMA_BASE_URL",      "http://localhost:11434")
USE_LLAMA_SERVER = config.get("USE_LLAMA_SERVER",     "false").lower() == "true"
LLAMA_SERVER_URL = config.get("LLAMA_SERVER_URL",     "http://localhost:8080/v1")
LLAMA_SERVER_MODEL = config.get("LLAMA_SERVER_MODEL", "qwen3-8b")
OLLAMA_LLM_MODEL = config.get("OLLAMA_LLM_MODEL",    "qwen3:8b")
LLM_TEMPERATURE  = float(config.get("LLM_TEMPERATURE", 0.1))


# ──────────────────────────────────────────────────────────────────────────────
# 2. Imports LangChain
# ──────────────────────────────────────────────────────────────────────────────
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Embeddings
from langchain_ollama import OllamaEmbeddings
if USE_HF_EMBEDDINGS:
    from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Vector store (novo pacote - sem deprecation warning)
from langchain_chroma import Chroma

# Retrievers para busca híbrida
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document

# Loaders de documento
try:
    from langchain_community.document_loaders import PyMuPDFLoader as PDFLoader
    _LOADER = "PyMuPDF"
except ImportError:
    from langchain_community.document_loaders import PyPDFLoader as PDFLoader
    _LOADER = "PyPDF"

from langchain_community.document_loaders import TextLoader


# ──────────────────────────────────────────────────────────────────────────────
# 3. Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_documents(docs_dir: str):
    """Carrega todos os PDFs e TXTs de uma pasta."""
    documents = []
    if not os.path.isdir(docs_dir):
        print(f"[ERRO] Pasta '{docs_dir}' não encontrada.")
        return documents

    files = os.listdir(docs_dir)
    if not files:
        print(f"[AVISO] Pasta '{docs_dir}' está vazia.")
        return documents

    for file in files:
        file_path = os.path.join(docs_dir, file)
        if file.lower().endswith(".pdf"):
            print(f"  Carregando PDF ({_LOADER}): {file}")
            loader = PDFLoader(file_path)
            documents.extend(loader.load())
        elif file.lower().endswith(".txt"):
            print(f"  Carregando TXT: {file}")
            loader = TextLoader(file_path, encoding="utf-8")
            documents.extend(loader.load())

    return documents


def build_embeddings():
    """Retorna o objeto de embeddings configurado (HuggingFace MiniLM ou Ollama nomic)."""
    if USE_HF_EMBEDDINGS:
        print(f"  Modelo de embeddings : HuggingFace ({EMBEDDING_MODEL_HF})")
        return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_HF)
    else:
        print(f"  Modelo de embeddings : Ollama ({EMBEDDING_MODEL_OLLAMA})")
        print(f"  Ollama URL           : {OLLAMA_BASE_URL}")
        return OllamaEmbeddings(
            model=EMBEDDING_MODEL_OLLAMA,
            base_url=OLLAMA_BASE_URL,
        )


def build_llm():
    """
    Retorna o LLM configurado de acordo com USE_LLAMA_SERVER:
      - True  → ChatOpenAI apontando ao llama-server (OpenAI-compat API)
      - False → OllamaLLM com qwen3:8b
    """
    if USE_LLAMA_SERVER:
        from langchain_openai import ChatOpenAI
        print(f"  Backend LLM : llama-server ({LLAMA_SERVER_URL})")
        print(f"  Modelo      : {LLAMA_SERVER_MODEL}")
        return ChatOpenAI(
            base_url=LLAMA_SERVER_URL,
            api_key="no-key",          # llama-server não exige chave
            model=LLAMA_SERVER_MODEL,
            temperature=LLM_TEMPERATURE,
            streaming=False,
        )
    else:
        from langchain_ollama import OllamaLLM
        print(f"  Backend LLM : Ollama")
        print(f"  Modelo      : {OLLAMA_LLM_MODEL}")
        return OllamaLLM(
            model=OLLAMA_LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            base_url=OLLAMA_BASE_URL,
        )


# ──────────────────────────────────────────────────────────────────────────────
# 4. Prompt em português
# ──────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────
# 4. Prompt em português (Otimizado para ser mais "esperto")
# ──────────────────────────────────────────────────────────────────────────────
RAG_PROMPT_TEMPLATE = """Você é um assistente virtual gentil e prestativo especializado no manual do SEI.
Responda às dúvidas com clareza e objetividade, sempre em português.

Regras:
1. Sempre priorize as informações contidas no Contexto abaixo para responder de forma direta.
2. Se a informação exata solicitada NÃO estiver explicitamente no Contexto:
   - Diga educadamente que não encontrou essa informação exata no documento de forma direta.
   - Em seguida, examine atenciosamente todos os fragmentos de texto do Contexto abaixo.
   - Liste de 2 a 3 assuntos, ferramentas, ícones ou processos descritos no Contexto que possam ter alguma proximidade ou relação indireta com a dúvida do usuário (por exemplo, se o usuário perguntou sobre 'atestados' e o contexto fala sobre 'incluir documento externo' ou 'adicionar arquivo', sugira essas ações correlatas).
   - Sugira que o usuário reformule a pergunta usando termos como esses.
   - NÃO invente ou alucine fatos que não constam no documento.

Contexto:
{context}

Pergunta: {question}

Resposta:"""


def preprocess_query(query: str) -> str:
    """Aplica regras de redundância e normalização de queries comuns."""
    q = query.strip()
    q_lower = q.lower()
    
    # 1. Expandir palavra isolada 'sei'
    if q_lower in ["sei", "o que e sei", "o que e o sei", "sei?"]:
        return "O que é o SEI (Sistema Eletrônico de Informações)?"
        
    # 2. Variações para atestados / anexos / documentos
    if "atestado" in q_lower or "atestados" in q_lower:
        # Se for apenas a palavra isolada, expandir
        if len(q_lower.split()) <= 2:
            return "Como incluir documento externo ou atestado no processo?"
            
    return q


# ──────────────────────────────────────────────────────────────────────────────
# 5. Ingestão (--ingest)
# ──────────────────────────────────────────────────────────────────────────────
def ingest():
    """Indexa documentos da pasta docs/ e salva no ChromaDB (chroma_db_qwen)."""
    print("\n=== Iniciando ingestão de documentos ===")
    print(f"  Pasta de documentos  : {DOCS_DIR}")
    print(f"  ChromaDB destino     : {PERSIST_DIR}")
    print(f"  Chunk size / overlap : {CHUNK_SIZE} / {CHUNK_OVERLAP}")

    # Carregar docs
    print("\n[1/4] Carregando documentos...")
    raw_docs = load_documents(DOCS_DIR)
    if not raw_docs:
        print("[ERRO] Nenhum documento encontrado. Coloque PDFs ou TXTs em ./docs")
        sys.exit(1)
    print(f"  >> {len(raw_docs)} paginas/secoes carregadas.")

    # Dividir em chunks
    print("\n[2/4] Dividindo em chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "],
        length_function=len,
    )
    chunks = splitter.split_documents(raw_docs)
    print(f"  >> {len(chunks)} chunks gerados.")

    # Embeddings
    print("\n[3/4] Inicializando embeddings...")
    embeddings = build_embeddings()

    # Salvar no ChromaDB
    print("\n[4/4] Criando banco de vetores...")
    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        print(f"  [AVISO] '{PERSIST_DIR}' já existe. Tentando limpar...")
        try:
            import shutil
            shutil.rmtree(PERSIST_DIR)
        except PermissionError:
            print(f"\n[ERRO] Não foi possível excluir a pasta '{PERSIST_DIR}' porque os arquivos do banco de dados estão abertos por outro processo (por exemplo, a interface Streamlit rodando em segundo plano).")
            print("Para corrigir:")
            print("  1. Feche ou cancele o processo do Streamlit (com Ctrl+C no terminal onde ele está executando).")
            print("  2. Execute a ingestão novamente: python rag_qwen.py --ingest")
            print("  3. Depois que a ingestão terminar, reinicie o Streamlit.")
            sys.exit(1)

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
    )

    print(f"\n[OK] Indexacao concluida! {len(chunks)} chunks salvos em '{PERSIST_DIR}'.")


# ──────────────────────────────────────────────────────────────────────────────
# 6. Chatbot
# ──────────────────────────────────────────────────────────────────────────────
def setup_rag_pipeline():
    """Inicializa e retorna o pipeline RAG (QA Chain)."""
    # Verificar se o ChromaDB existe
    if not os.path.exists(PERSIST_DIR) or not os.listdir(PERSIST_DIR):
        raise FileNotFoundError(f"Banco de vetores não encontrado em '{PERSIST_DIR}'. Execute a ingestão primeiro.")

    # Embeddings
    embeddings = build_embeddings()

    # Vector store
    vectordb = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
    )

    # LLM
    llm = build_llm()

    # Pipeline RAG com Busca Híbrida (Chroma + BM25)
    data = vectordb.get()
    documents = []
    for doc_text, metadata in zip(data['documents'], data['metadatas']):
        documents.append(Document(page_content=doc_text, metadata=metadata))
        
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = SEARCH_K
    
    chroma_retriever = vectordb.as_retriever(search_kwargs={"k": SEARCH_K})
    
    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, chroma_retriever],
        weights=[0.5, 0.5]
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=RAG_PROMPT_TEMPLATE,
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=hybrid_retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    
    return qa_chain


def chat():
    """Loop interativo de perguntas e respostas com RAG."""
    print("\n=== Inicializando Chatbot RAG — Qwen3 + nomic-embed-text ===")

    try:
        qa_chain = setup_rag_pipeline()
    except Exception as e:
        print(f"\n[ERRO] {e}")
        sys.exit(1)

    backend = "llama-server" if USE_LLAMA_SERVER else f"Ollama ({OLLAMA_LLM_MODEL})"
    print(f"\n[OK] Chatbot pronto! Backend: {backend}")
    print("Digite sua pergunta ou 'sair' para encerrar.\n")
    print("-" * 60)

    while True:
        try:
            question = input("\nPergunta: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nEncerrando. Até mais!")
            break

        if not question:
            continue
        if question.lower() in ["sair", "exit", "quit"]:
            print("Encerrando. Até mais!")
            break

        # Pré-processar a query para tratar case sensitivity e abreviações
        processed_query = preprocess_query(question)
        if processed_query != question:
            print(f"  (Buscando por: '{processed_query}')")

        print("\nProcessando...", end="", flush=True)
        try:
            result = qa_chain.invoke({"query": processed_query})
            print("\r" + " " * 20 + "\r", end="")  # limpar "Processando..."
            print(f"\nResposta:\n{result['result']}")

            # Mostrar fontes
            sources = result.get("source_documents", [])
            if sources:
                print("\n--- Fontes consultadas ---")
                for i, doc in enumerate(sources[:3], 1):
                    page = doc.metadata.get("page", "?")
                    source = os.path.basename(doc.metadata.get("source", "Desconhecida"))
                    print(f"  [{i}] {source} - Pagina {page}")
                    print(f"       {doc.page_content[:150].strip()}...")
            print("-" * 60)

        except Exception as e:
            print(f"\n[ERRO] {e}")
            if USE_LLAMA_SERVER:
                print("Dica: Verifique se o llama-server está rodando em localhost:8080")
            else:
                print("Dica: Verifique se o Ollama está rodando (ollama serve) e se o modelo qwen3:8b está disponível.")


# ──────────────────────────────────────────────────────────────────────────────
# 7. Entry point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RAG Pipeline — Qwen3 8B + nomic-embed-text + ChromaDB"
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Indexar documentos da pasta docs/ (rodar uma vez antes do chat)",
    )
    args = parser.parse_args()

    if args.ingest:
        ingest()
    else:
        chat()
