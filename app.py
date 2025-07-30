# gerador_dialogo.py
#
# DESCRIÇÃO:
# Este script busca uma notícia recente sobre ciência, tecnologia ou economia,
# utiliza um modelo de linguagem (Gemma via GGUF) para gerar um diálogo filosófico
# sobre a notícia e, por fim, salva o resultado como uma página HTML formatada.
#
# PRÉ-REQUISITOS:
# 1. Python 3.8+
# 2. Instale as dependências necessárias:
#    pip install -r requirements.txt
# 3. Baixe um modelo de linguagem no formato GGUF.
#    Este script foi testado com um modelo Gemma.
# 4. Atualize a variável 'MODEL_DIRECTORY_PATH' na seção de configuração abaixo.

import os
import requests
import logging
import re
from datetime import datetime
from zoneinfo import ZoneInfo
from bs4 import BeautifulSoup
import random
import glob
from ctransformers import AutoModelForCausalLM

# --- 1. CONFIGURAÇÃO ---

# !!! ATENÇÃO: MODIFIQUE ESTA LINHA !!!
# Coloque o caminho para o DIRETÓRIO que contém seu arquivo de modelo .gguf
MODEL_DIRECTORY_PATH = "/caminho/para/seu/diretorio/gemma-gguf" 

# Nome do arquivo HTML que será gerado
OUTPUT_FILENAME = "dialogo_filosofico.html"

# Configurações de logging
logging.getLogger().setLevel(logging.ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# --- 2. FUNÇÃO PARA OBTER A NOTÍCIA ---
# (Esta função permanece inalterada)
def common_prefix_len_ignore_case(s1, s2):
    s1_lower = s1.lower()
    s2_lower = s2.lower()
    i = 0
    while i < len(s1_lower) and i < len(s2_lower) and s1_lower[i] == s2_lower[i]:
        i += 1
    return i

def obter_noticia():
    base_url = "https://news.google.com/rss/search?q=(ciência+OR+tecnologia+OR+economia)&hl=pt-BR&gl=BR&ceid=BR:pt-419"
    url = f"{base_url}&cache_bust={datetime.now().timestamp()}"
    
    try:
        resposta = requests.get(url)
        # Usando 'lxml-xml' para maior robustez com RSS/XML
        soup = BeautifulSoup(resposta.content, features="xml")
        items = soup.find_all("item")
        if not items:
            return {"titulo": "Nenhuma notícia encontrada", "conteudo": "Não foi possível localizar notícias neste momento.", "link": ""}
        item_selecionado = random.choice(items)

        noticia_recente = {
            "titulo": item_selecionado.title.text,
            "link": item_selecionado.link.text,
            "snippet": item_selecionado.description.text
        }

        titulo_bruto = noticia_recente["titulo"]
        titulo_final = re.sub(r' - .*$', '', titulo_bruto).strip()

        conteudo_bruto = noticia_recente.get("snippet", "Conteúdo não disponível.")
        conteudo_limpo = BeautifulSoup(conteudo_bruto, 'html.parser').get_text(separator=' ').strip()

        title_str = titulo_final
        content_str = conteudo_limpo

        common_len = common_prefix_len_ignore_case(title_str, content_str)
        if len(title_str) > 0 and (common_len / len(title_str)) > 0.9:
            conteudo_limpo = content_str[common_len:].lstrip(" -–—:").strip()
        else:
            conteudo_limpo = content_str

        if len(conteudo_limpo.split()) < 4:
            conteudo_limpo = ""

        if titulo_final.endswith("..."):
            titulo_final += " (título completo no link)"
        if conteudo_limpo.endswith("..."):
            conteudo_limpo += " (continuação disponível na matéria original)"

        return {
            "titulo": str(titulo_final),
            "conteudo": str(conteudo_limpo),
            "link": str(noticia_recente["link"])
        }

    except requests.exceptions.RequestException as e:
        print(f"Erro ao buscar notícia: {e}")
        return {"titulo": "Erro de Conexão", "conteudo": "Não foi possível conectar à API de notícias.", "link": ""}


# --- 3. FUNÇÃO PARA ENCONTRAR O MODELO ---
def encontrar_modelo_disponivel():
    """Tenta encontrar um modelo GGUF no diretório configurado."""
    
    print(f"Testando caminho configurado: {MODEL_DIRECTORY_PATH}")
    if os.path.exists(MODEL_DIRECTORY_PATH):
        # Procura por qualquer arquivo .gguf no diretório
        arquivos_gguf = glob.glob(f"{MODEL_DIRECTORY_PATH}/*.gguf")
        if arquivos_gguf:
            caminho_modelo = arquivos_gguf[0] # Pega o primeiro arquivo encontrado
            print(f"✅ Modelo GGUF encontrado em: {caminho_modelo}")
            return caminho_modelo
        else:
            print(f"❌ Caminho '{MODEL_DIRECTORY_PATH}' existe, mas nenhum arquivo .gguf foi encontrado dentro dele.")
    else:
        print(f"❌ Caminho '{MODEL_DIRECTORY_PATH}' não existe. Verifique a variável 'MODEL_DIRECTORY_PATH' no script.")
        
    return None


# --- 4. FUNÇÃO PARA GERAR O DIÁLOGO ---
def gerar_dialogo(noticia):
    print("🔍 Procurando modelo disponível...")
    model_path = encontrar_modelo_disponivel()
    
    if model_path is None:
        print("❌ Nenhum modelo GGUF encontrado!")
        return gerar_dialogo_fallback(noticia)
    
    print(f"Gerando diálogo com IA (modelo {os.path.basename(model_path)})...")

    # URLs dos ícones (mantidas para a geração do HTML)
    sagredo_icon_url = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='%232c5aa0' stroke-width='2'%3E%3Ccircle cx='12' cy='12' r='10'/%3E%3Cpath d='M8 14s1.5 2 4 2 4-2 4-2'/%3E%3Cline x1='9' y1='9' x2='9.01' y2='9'/%3E%3Cline x1='15' y1='9' x2='15.01' y2='9'/%3E%3C/svg%3E"
    salvati_icon_url = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='%23c53030' stroke-width='2'%3E%3Ccircle cx='12' cy='12' r='10'/%3E%3Cpath d='M16 16s-1.5-2-4-2-4 2-4 2'/%3E%3Cline x1='9' y1='9' x2='9.01' y2='9'/%3E%3Cline x1='15' y1='9' x2='15.01' y2='9'/%3E%3C/svg%3E"
    simplicio_icon_url = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='%2338a169' stroke-width='2'%3E%3Ccircle cx='12' cy='12' r='10'/%3E%3Cline x1='8' y1='15' x2='16' y2='15'/%3E%3Cline x1='9' y1='9' x2='9.01' y2='9'/%3E%3Cline x1='15' y1='9' x2='15.01' y2='9'/%3E%3C/svg%3E"

    try:
        print("Carregando modelo GGUF com ctransformers...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            model_type="gemma",
            gpu_layers=50,  # Descarrega 50 camadas na GPU. Ajuste para 0 se não tiver GPU compatível.
            context_length=4096 
        )
        print("✅ Modelo carregado com sucesso!")

        resumo_para_ia = noticia['conteudo']
        if not resumo_para_ia.strip():
            resumo_para_ia = "(O resumo da notícia não foi fornecido. Baseie o diálogo apenas no título.)"

        prompt_template = f"""<start_of_turn>user
Crie um diálogo filosófico em português do Brasil sobre a seguinte notícia.

Notícia:
- Título: "{noticia['titulo']}"
- Resumo: "{resumo_para_ia}"

O diálogo deve seguir estritamente a seguinte sequência de seis turnos:
1. Sagredo (personalidade: irônico, inquieto)
2. Salvati (personalidade: crítico, pós-moderno)
3. Simplicio (personalidade: conservador, confiante)
4. Sagredo
5. Simplicio
6. Salvati

A resposta deve ser apenas o código HTML, contendo EXATAMENTE seis falas, uma para cada turno, com cada fala dentro de uma tag <p> e o nome do personagem em <strong>. Não inclua nenhum outro texto, aspas ou marcador de código.
<end_of_turn>
<start_of_turn>model
"""
        print("Gerando texto (isso pode levar alguns minutos)...")
        dialogo_html = model(
            prompt_template,
            max_new_tokens=1024,
            temperature=0.7,
            top_p=0.9,
            stop=['<end_of_turn>']
        )

        print("Processando e formatando a saída...")
        
        # Lógica de pós-processamento para garantir o formato HTML correto
        icon_style = 'width="24" height="24" style="vertical-align: middle; margin-right: 8px;"'
        
        if 'img src=' not in dialogo_html:
            dialogo_html = dialogo_html.replace(
                "<strong>Sagredo:</strong>", f'<img src="{sagredo_icon_url}" {icon_style} alt="Sagredo"><strong>Sagredo:</strong>')
            dialogo_html = dialogo_html.replace(
                "<strong>Salvati:</strong>", f'<img src="{salvati_icon_url}" {icon_style} alt="Salvati"><strong>Salvati:</strong>')
            dialogo_html = dialogo_html.replace(
                "<strong>Simplicio:</strong>", f'<img src="{simplicio_icon_url}" {icon_style} alt="Simplicio"><strong>Simplicio:</strong>')

        return str(dialogo_html.strip())

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ Erro ao gerar diálogo com IA: {str(e)}")
        return gerar_dialogo_fallback(noticia)


# --- 5. FUNÇÃO FALLBACK E FUNÇÃO DE MONTAGEM DA PÁGINA ---
# (Estas funções permanecem inalteradas)
def gerar_dialogo_fallback(noticia):
    """Gera um diálogo básico quando o modelo IA não está disponível"""
    print("🔄 Gerando diálogo fallback...")
    
    sagredo_icon_url = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='%232c5aa0' stroke-width='2'%3E%3Ccircle cx='12' cy='12' r='10'/%3E%3Cpath d='M8 14s1.5 2 4 2 4-2 4-2'/%3E%3Cline x1='9' y1='9' x2='9.01' y2='9'/%3E%3Cline x1='15' y1='9' x2='15.01' y2='9'/%3E%3C/svg%3E"
    salvati_icon_url = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='%23c53030' stroke-width='2'%3E%3Ccircle cx='12' cy='12' r='10'/%3E%3Cpath d='M16 16s-1.5-2-4-2-4 2-4 2'/%3E%3Cline x1='9' y1='9' x2='9.01' y2='9'/%3E%3Cline x1='15' y1='9' x2='15.01' y2='9'/%3E%3C/svg%3E"
    simplicio_icon_url = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='%2338a169' stroke-width='2'%3E%3Ccircle cx='12' cy='12' r='10'/%3E%3Cline x1='8' y1='15' x2='16' y2='15'/%3E%3Cline x1='9' y1='9' x2='9.01' y2='9'/%3E%3Cline x1='15' y1='9' x2='15.01' y2='9'/%3E%3C/svg%3E"
    icon_style = 'width="24" height="24" style="vertical-align: middle; margin-right: 8px;"'
    
    titulo = noticia.get('titulo', 'Notícia não disponível')
    
    dialogo = f"""<p><img src="{sagredo_icon_url}" {icon_style} alt="Sagredo"><strong>Sagredo:</strong> Ora, que interessante esta notícia sobre "{titulo}". Mas será que devemos confiar plenamente no que lemos nos jornais modernos?</p>
<p><img src="{salvati_icon_url}" {icon_style} alt="Salvati"><strong>Salvati:</strong> Sagredo levanta uma questão pertinente. Vivemos numa era de informação fragmentada, onde cada notícia é apenas um recorte da realidade, moldado por interesses específicos.</p>
<p><img src="{simplicio_icon_url}" {icon_style} alt="Simplicio"><strong>Simplicio:</strong> Amigos, creio que vocês complicam demasiadamente as coisas. Se está nos jornais, especialmente em fontes respeitáveis, devemos considerar que há fundamento na informação.</p>
<p><img src="{sagredo_icon_url}" {icon_style} alt="Sagredo"><strong>Sagredo:</strong> Ah, Simplicio, sua fé na autoridade das fontes me impressiona! Mas não seria prudente questionar também as próprias bases dessas "fontes respeitáveis"?</p>
<p><img src="{simplicio_icon_url}" {icon_style} alt="Simplicio"><strong>Simplicio:</strong> O ceticismo excessivo nos levaria à paralisia total, Sagredo. É necessário confiar em alguma estrutura de conhecimento para que possamos avançar em nossa compreensão do mundo.</p>
<p><img src="{salvati_icon_url}" {icon_style} alt="Salvati"><strong>Salvati:</strong> Talvez a verdade esteja no meio-termo: nem a credulidade cega de Simplicio, nem o ceticismo absoluto de Sagredo, mas uma postura crítica que avalie cada informação em seu contexto específico.</p>"""
    return dialogo

def gerar_pagina_html(noticia, dialogo_html):
    agora_em_sao_paulo = datetime.now(ZoneInfo("America/Sao_Paulo"))
    dias_semana = {
        'Monday': 'Segunda-feira', 'Tuesday': 'Terça-feira', 'Wednesday': 'Quarta-feira',
        'Thursday': 'Quinta-feira', 'Friday': 'Sexta-feira', 'Saturday': 'Sábado', 'Sunday': 'Domingo'
    }
    dia = dias_semana.get(agora_em_sao_paulo.strftime('%A'), 'Dia')
    data_hora = agora_em_sao_paulo.strftime(f"{dia}, %d/%m/%Y, %H:%M")

    titulo = str(noticia.get("titulo", ""))
    conteudo = str(noticia.get("conteudo", ""))
    link = str(noticia.get("link", ""))
    dialogo_html = str(dialogo_html)

    titulo_com_link = f'<a href="{link}" target="_blank" style="text-decoration:none; color:inherit;">{titulo}</a>'

    return f"""
<!DOCTYPE html><html lang="pt"><head><meta charset="UTF-8" /><meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Eu li os jornais hoje, ô, cara</title>
<style>
    body{{font-family:Georgia,serif;margin:0;padding:0;background:#fdfdfd;color:#222;font-size:17px;}}
    header{{background-color:#e3e3e3;padding:1.5em;}}
    footer{{background-color:#e3e3e3;padding:1em;text-align:center;}}
    main{{max-width:800px;margin:auto;padding:2em;}}
    h1{{font-size:2em;text-align:center;}}
    h2 a:hover{{text-decoration:underline;}}
    .titulo-secao{{display:flex;justify-content:space-between;align-items:flex-start;border-bottom:1px solid #ccc;padding-bottom:10px;margin-bottom:1em;}}
    .titulo-secao h2{{flex-grow:1;margin:0 1em 0 0;font-size:1.6em;line-height:1.4;}}
    .data-hora{{font-style:italic;color:#666;font-size:1em;white-space:nowrap;padding-top:5px;}}
    .dialogo p{{margin-bottom:1.2em;line-height:1.6;}}
    .dialogo strong{{color:#2c5aa0;}}
    #dialogo{{margin-top:2em;}}
</style></head><body>
<header><h1>Eu li os jornais hoje, ô, cara</h1><p style="text-align:center;">"I read the news today, oh, boy" (Lennon/McCartney, A day in the life, 1967)</p></header>
<main>
    <section>
        <div class="titulo-secao">
            <h2>{titulo_com_link}</h2>
            <span class="data-hora">{data_hora}</span>
        </div>
        <p>{conteudo}</p>
    </section>
    <section id="dialogo" class="dialogo">
        <h2>Sagredo, Salvati e Simplício no século 21</h2>
        {dialogo_html}
    </section>
</main>
<footer><p>Script executado em {data_hora}. Para gerar uma nova página, execute o script novamente.</p></footer>
</body></html>
"""


# --- 6. EXECUÇÃO PRINCIPAL ---
def main():
    """Função principal que orquestra a execução do script."""
    print("🚀 Iniciando o programa...")
    
    noticia_atual = obter_noticia()
    if noticia_atual and noticia_atual["titulo"] not in ["Nenhuma notícia encontrada", "Erro de Conexão"]:
        print("📰 Notícia obtida com sucesso!")
        dialogo_gerado = gerar_dialogo(noticia_atual)
    else:
        print("❌ Falha ao obter notícia. Usando diálogo fallback.")
        dialogo_gerado = gerar_dialogo_fallback(noticia_atual)

    print("🎭 Montando página HTML...")
    pagina_completa = gerar_pagina_html(noticia_atual, dialogo_gerado)
    
    try:
        # Salva o conteúdo em um arquivo HTML, com codificação UTF-8
        with open(OUTPUT_FILENAME, "w", encoding="utf-8") as f:
            f.write(pagina_completa)
        print(f"✅ Página gerada com sucesso! Verifique o arquivo: {OUTPUT_FILENAME}")
    except IOError as e:
        print(f"❌ Erro ao salvar o arquivo HTML: {e}")
        
    print("✅ Processo concluído!")


if __name__ == "__main__":
    main()
