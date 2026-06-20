import streamlit as st
import os

from rag_qwen import setup_rag_pipeline, preprocess_query

# 1. Configuração da página
st.set_page_config(
    page_title="Chatbot SEI",
    layout="centered",
    initial_sidebar_state="expanded"
)

# 2. Estilos customizados (CSS)
custom_css = """
<style>
/* Variáveis de cores do usuário */
:root {
    --bg-primary:    #0d1117;   /* fundo geral */
    --bg-card:       #131b24;   /* cards e sidebar */
    --bg-bubble-bot: #0e1e2b;   /* bolha do assistente */
    --bg-bubble-user:#0a3d8f;   /* bolha do usuário */
    --accent:        #00c2a8;   /* cyan mint — borda bot, CTA */
    --accent-muted:  #1a7a6e;   /* teal escuro — hover */
    --blue:          #1e6fff;   /* electric blue — borda user */
    --text-primary:  #e8f0fe;   /* títulos e conteúdo */
    --text-muted:    #6b8aa8;   /* labels, hints */
    --border:        #1e2e3d;   /* divisores */
}

/* Fundo geral e texto principal */
.stApp {
    background-color: var(--bg-primary);
    color: var(--text-primary);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: var(--bg-card);
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] * {
    color: var(--text-primary);
}

/* Títulos globais */
h1, h2, h3, h4, h5, h6 {
    color: var(--text-primary) !important;
}

/* Inputs de texto no chat */
.stChatInputContainer textarea {
    background-color: var(--bg-card) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
}
.stChatInputContainer textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 1px var(--accent) !important;
}

/* Customização de botões (opcional) */
.stButton > button {
    background-color: var(--bg-card);
    color: var(--accent);
    border: 1px solid var(--accent);
    transition: all 0.2s ease-in-out;
}
.stButton > button:hover {
    background-color: var(--accent-muted);
    color: var(--text-primary);
    border-color: var(--accent);
}

/* Customização do chat_message (Streamlit native classes override) */

/* Mensagens do Bot */
[data-testid="chatAvatarIcon-assistant"] {
    background-color: var(--accent);
}

div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background-color: var(--bg-bubble-bot);
    border: 1px solid var(--accent);
    border-radius: 10px;
    padding: 10px;
    margin-bottom: 10px;
}

/* Mensagens do User */
[data-testid="chatAvatarIcon-user"] {
    background-color: var(--blue);
}

div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background-color: var(--bg-bubble-user);
    border: 1px solid var(--blue);
    border-radius: 10px;
    padding: 10px;
    margin-bottom: 10px;
}

/* Expanders (Fontes consultadas) */
.streamlit-expanderHeader {
    background-color: var(--bg-card) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
}
.streamlit-expanderContent {
    background-color: var(--bg-primary) !important;
    color: var(--text-muted) !important;
    border: 1px solid var(--border) !important;
    border-top: none !important;
}

</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Carregando modelos e banco de dados...")
def get_qa_chain():
    try:
        return setup_rag_pipeline()
    except Exception as e:
        import traceback
        st.error("Erro ao carregar o pipeline:")
        st.exception(e)
        st.write(traceback.format_exc())
        return None

# Inicializar RAG
qa_chain = get_qa_chain()

# 4. Interface da Sidebar
with st.sidebar:
    st.title("Chatbot SEI")
    st.write("Assistente virtual treinado com o manual do SEI.")
    
    st.markdown("---")
    st.subheader("Informações")
    if qa_chain:
        st.success("✅ Sistema Online")
        st.caption("Backend Híbrido: ChromaDB + BM25")
    else:
        st.error("❌ Erro de inicialização")
    
    st.markdown("---")
    if st.button("Limpar Histórico"):
        st.session_state.messages = []
        st.rerun()

# 5. Interface Principal (Chat)
st.title("Assistente SEI")
st.write("Tire suas dúvidas sobre o Sistema Eletrônico de Informações.")

# Inicializar histórico na sessão
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Olá! Sou o assistente do SEI. Como posso te ajudar hoje?"}
    ]

# Exibir histórico de mensagens
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Exibir fontes se for uma resposta do assistente e contiver fontes
        if message["role"] == "assistant" and "sources" in message and message["sources"]:
            with st.expander("Ver fontes consultadas"):
                for i, doc in enumerate(message["sources"], 1):
                    page = doc.metadata.get("page", "?")
                    source = os.path.basename(doc.metadata.get("source", "Desconhecida"))
                    st.markdown(f"**[{i}] {source} - Página {page}**")
                    st.caption(f"{doc.page_content[:300].strip()}...")

# 6. Input do Usuário
if prompt := st.chat_input("Pergunte sobre atestados, processos, tramitação..."):
    # Adicionar pergunta ao histórico
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Processar a resposta do assistente
    with st.chat_message("assistant"):
        if not qa_chain:
            st.error("O sistema não foi inicializado corretamente.")
        else:
            with st.spinner("Pesquisando no manual..."):
                try:
                    # Pré-processar a query
                    processed_query = preprocess_query(prompt)
                    
                    # Invocar RAG
                    result = qa_chain.invoke({"query": processed_query})
                    answer = result.get("result", "Desculpe, não consegui obter uma resposta.")
                    sources = result.get("source_documents", [])
                    
                    st.markdown(answer)
                    if sources:
                        with st.expander("Ver fontes consultadas"):
                            for i, doc in enumerate(sources[:3], 1):
                                page = doc.metadata.get("page", "?")
                                source = os.path.basename(doc.metadata.get("source", "Desconhecida"))
                                st.markdown(f"**[{i}] {source} - Página {page}**")
                                st.caption(f"{doc.page_content[:300].strip()}...")
                    
                    # Salvar resposta no histórico
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "sources": sources[:3] if sources else None
                    })
                    
                except Exception as e:
                    st.error(f"Ocorreu um erro ao processar sua pergunta: {e}")
