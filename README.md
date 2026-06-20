<div align="center">

# 📖 RAG LLM Assistente Virtual do SEI

### Pipeline de Retrieval-Augmented Generation especializado no manual do usuário do<br>Sistema Eletrônico de Informações

![Status](https://img.shields.io/badge/status-ativo-00c2a8?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.11+-1e6fff?style=flat-square&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-orquestração-1a7a6e?style=flat-square)
![ChromaDB](https://img.shields.io/badge/ChromaDB-vector--store-534ab7?style=flat-square)
![Streamlit](https://img.shields.io/badge/Interface-Streamlit-00c2a8?style=flat-square&logo=streamlit&logoColor=white)
![Offline](https://img.shields.io/badge/execução-100%25%20offline-0a3d8f?style=flat-square&logo=lock&logoColor=white)
![GPU](https://img.shields.io/badge/GPU-RX%20580%20·%20Vulkan-ef9f27?style=flat-square)

</div>

---

## Sobre o projeto

Este repositório contém a implementação de um pipeline de **RAG (Retrieval-Augmented Generation)** projetado para atuar como um assistente inteligente especializado no manual do usuário do SEI. O objetivo principal é facilitar a consulta rápida a procedimentos e conceitos administrativos descritos no manual oficial do sistema, sem necessidade de conexão com a internet.

---

## 🎯 Objetivos

| | Objetivo | Descrição |
|---|---|---|
| 💬 | **Acesso direto à informação** | Substitui consultas manuais longas no PDF por respostas precisas em linguagem natural |
| 🔒 | **Privacidade e execução local** | 100% offline — LLM e embeddings rodam na máquina do usuário, nenhum dado sai do computador |
| 🔍 | **Busca robusta e híbrida** | Responde corretamente mesmo a perguntas curtas usando busca semântica + correspondência exata |

---

## ⚡ Funcionalidades

### 🔀 Busca híbrida com `EnsembleRetriever`
Une a precisão de palavras-chave exatas do algoritmo **BM25** com a compreensão semântica vetorial do **ChromaDB**, garantindo recuperação de contexto mais robusta para perguntas curtas como `"sei"` ou `"atestados"`.

### 🧹 Pré-processamento inteligente
Tratamento automático de termos comuns, expansão de abreviações e tolerância a variações de maiúsculas/minúsculas antes da indexação do manual.

### 💡 Respostas com sugestões correlatas
Quando a informação exata não consta no manual, o assistente detecta a ausência e sugere **2 a 3 assuntos correlacionados** encontrados no contexto para que o usuário refine sua dúvida.

### 🖥️ Interface gráfica moderna
Chat web com **tema escuro personalizado** (paleta azul/cyan-mint), histórico de conversações e exibição expandida das fontes e páginas consultadas do manual — construído em Streamlit.

---

## 🛠️ Stack de tecnologias

| Camada | Tecnologia | Função |
|---|---|---|
| Orquestração | [LangChain](https://github.com/langchain-ai/langchain) | Pipeline RAG e chains |
| LLM | Qwen-8B · Command R | Geração de respostas (local via Ollama ou llama.cpp) |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` · `nomic-embed-text` | Vetorização dos chunks |
| Vector store | [ChromaDB](https://github.com/chroma-core/chroma) | Busca por similaridade semântica |
| Busca lexical | BM25 | Correspondência exata de palavras-chave |
| PDF parsing | PyMuPDF (`fitz`) · PyPDF | Extração e chunking do manual |
| Interface | [Streamlit](https://streamlit.io/) | Frontend do chat |
| Inferência GPU | llama.cpp + Vulkan | Aceleração RX 580 sem ROCm |

---

## 📂 Documentação

| # | Guia | Conteúdo |
|---|---|---|
| 1 | [**Instalação e Execução**](instructions.md) | Preparo do ambiente, variáveis `.env`, pipeline de ingestão e servidores locais |
| 2 | [**Arquitetura e Funcionamento**](development.md) | RAG híbrido, processamento de texto, lógica do prompt e estrutura de pastas |

---

##  Início rápido

```bash
# 1. clonar o repositório
git clone https://github.com/seu-usuario/rag-sei.git
cd rag-sei

# 2. instalar dependências
pip install -r requirements.txt

# 3. baixar modelo de embedding
ollama pull nomic-embed-text

# 4. subir o LLM na GPU (RX 580 via Vulkan)
./llama-server --device Vulkan0 \
  -hf unsloth/Qwen3-8B-GGUF:Q4_K_M \
  --ctx-size 8192 --port 8080

# 5. indexar o manual
python smart_ingest.py

# 6. iniciar o assistente
streamlit run app.py
```

---

<div align="center">

![Desenvolvido com por **Gustavo**](https://img.shields.io/badge/Desenvolvido%20por%20Gustavo-solução%20local-00c2a8?style=flat-square)

</div>
