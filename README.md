# RAG LLM — Assistente Virtual do SEI (Sistema Eletrônico de Informações)

Este repositório contém a implementação de um pipeline de **RAG (Retrieval-Augmented Generation)** projetado para atuar como um assistente inteligente especializado no manual do usuário do SEI. O objetivo principal do projeto é facilitar a consulta rápida a procedimentos e conceitos administrativos descritos no manual oficial do sistema.

---

## Objetivos do Projeto

- **Facilitar o acesso à informação**: Substituir consultas manuais longas no arquivo PDF por respostas diretas e precisas em linguagem natural.
- **Privacidade e Funcionamento Local**: Permitir a execução 100% offline, rodando modelos de inferência (como Qwen 3.8B) e embeddings (como MiniLM ou Nomic) localmente no computador do usuário (com suporte a aceleração por GPU RX 580 via Vulkan/llama.cpp ou por CPU via Ollama).
- **Busca Robusta e Híbrida**: Responder corretamente mesmo a perguntas curtas (como "sei" ou "atestados") usando busca híbrida baseada em proximidade semântica e correspondência de palavras-chave exatas.

---

## ⚡ Principais Funcionalidades

- **Busca Híbrida (`EnsembleRetriever`)**: Une a precisão de palavras-chave exatas do algoritmo **BM25** com a compreensão semântica vetorial do **ChromaDB**.
- **Pré-processamento Inteligente**: Tratamento automático de termos comuns, expansão de abreviações e tolerância a variações de maiúsculas/minúsculas.
- **Respostas com Sugestões Correlatas**: Se a informação exata solicitada não constar no manual, o assistente detecta e sugere de 2 a 3 assuntos correlacionados encontrados no contexto para que o usuário possa refinar sua dúvida.
- **Interface Gráfica Moderna (Streamlit)**: Um chat web com tema escuro personalizado (paleta azul/cyan-mint), histórico de conversações e exibição expandida das fontes e páginas consultadas do manual.

---

## 🛠️ Tecnologias Utilizadas

- **Orquestração**: [LangChain](https://github.com/langchain-ai/langchain)
- **Modelos de Linguagem (LLM)**: Qwen-8B ou Command R (local via Ollama ou llama.cpp)
- **Embeddings**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (via HuggingFace) ou `nomic-embed-text` (via Ollama)
- **Banco de Dados Vetorial**: [ChromaDB](https://github.com/chroma-core/chroma)
- **Busca Lexográfica**: BM25 (via `langchain_community.retrievers`)
- **Processamento de PDFs**: PyMuPDF (`fitz`) / PyPDF
- **Frontend / Interface**: [Streamlit](https://streamlit.io/)

---

## 📂 Estrutura de Documentação do Projeto

Para informações detalhadas de como usar e de como a arquitetura do projeto foi estruturada, consulte os guias a seguir:

1. **[Guia de Instalação e Execução (instructions.md)](file:///e:/RAG_LLM/instructions.md)**: Passo a passo de como preparar o ambiente, configurar variáveis de ambiente `.env`, rodar o pipeline de ingestão e levantar os servidores localmente.
2. **[Arquitetura e Funcionamento do Sistema (development.md)](file:///e:/RAG_LLM/development.md)**: Explicação técnica detalhada de como o RAG híbrido está estruturado, processamento de texto, lógica do prompt redundante e estrutura das pastas.
