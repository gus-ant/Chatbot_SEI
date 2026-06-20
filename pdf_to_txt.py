"""
pdf_to_txt.py - Conversor de PDF para TXT Limpo para RAG
======================================================
Este script extrai o texto do PDF usando PyMuPDF (fitz),
remove cabeçalhos/rodapés repetitivos e aplica correções
para termos que possam ter sumido ou ficado corrompidos
(como a palavra 'SEI' representada por imagens no PDF).

Uso:
    python pdf_to_txt.py docs/manual_do_usuario_sei.pdf docs/manual_do_usuario_sei.txt
"""

import sys
import os
import re

try:
    import fitz  # PyMuPDF
except ImportError:
    print("[ERRO] PyMuPDF (pymupdf) não instalado. Instale usando: pip install pymupdf")
    sys.exit(1)

def clean_and_convert(pdf_path, txt_path):
    if not os.path.exists(pdf_path):
        print(f"[ERRO] Arquivo PDF não encontrado em: {pdf_path}")
        return

    print(f"Abrindo PDF: {pdf_path}")
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    print(f"Total de páginas: {total_pages}")

    lines_out = []

    # Cabeçalhos e rodapés comuns no manual do SEI para remover
    header_pattern = re.compile(r"Sistema Eletr[ôo]nico de Informa[çc][õo]es\s*–?\s*Manual do Usu[áa]rio", re.IGNORECASE)

    for page_num in range(total_pages):
        page = doc[page_num]
        text = page.get_text("text")
        
        # Separar linhas
        lines = text.split("\n")
        cleaned_lines = []
        
        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue
            
            # Remover número de página isolado ou cabeçalhos repetitivos
            if line_str.isdigit():
                continue
            if header_pattern.search(line_str):
                continue
            
            # Correções heurísticas para caracteres órfãos/corrompidos
            # No manual do SEI, o logotipo do SEI às vezes é extraído como vazio ou caracteres especiais
            # como '\x8e', '\x90', etc. ou palavras cortadas.
            # Vamos corrigir strings onde a palavra 'SEI' sumiu:
            
            # "O [vazio] é um" -> "O SEI é um"
            line_str = re.sub(r"\bO\s+[\x80-\xff]?\s*é\b", "O SEI é", line_str)
            # "facilidades do [vazio]:" -> "facilidades do SEI:"
            line_str = re.sub(r"\bfacilidades\s+do\s+[\x80-\xff]?\b", "facilidades do SEI", line_str)
            # "pesquisa do [vazio] pode" -> "pesquisa do SEI pode"
            line_str = re.sub(r"\bpesquisa\s+do\s+[\x80-\xff]?\b", "pesquisa do SEI", line_str)
            # "usuários do [vazio]" -> "usuários do SEI"
            line_str = re.sub(r"\busuários\s+do\s+[\x80-\xff]?\b", "usuários do SEI", line_str)
            # "uso do [vazio]" -> "uso do SEI"
            line_str = re.sub(r"\buso\s+do\s+[\x80-\xff]?\b", "uso do SEI", line_str)
            
            cleaned_lines.append(line_str)
            
        if cleaned_lines:
            # Separador visual de páginas opcional (ajuda o RAG a contextualizar a origem)
            lines_out.append(f"\n--- PÁGINA {page_num + 1} ---")
            lines_out.extend(cleaned_lines)

    # Escrever arquivo texto final em UTF-8
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines_out))
        
    print(f"\n[OK] Conversão concluída com sucesso!")
    print(f"Arquivo texto gerado: {txt_path}")
    print("Dica: Você pode abrir este arquivo .txt no Bloco de Notas para verificar/revisar antes de rodar o ingest.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        # Padrão para facilitar o uso do usuário
        pdf = "docs/manual_do_usuario_sei.pdf"
        txt = "docs/manual_do_usuario_sei.txt"
        print(f"Uso padrão: python pdf_to_txt.py")
        print(f"Processando {pdf} -> {txt}...")
        clean_and_convert(pdf, txt)
    else:
        clean_and_convert(sys.argv[1], sys.argv[2])
