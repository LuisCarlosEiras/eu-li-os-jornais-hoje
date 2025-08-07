# pip install requests kaggle ipython torch transformers bitsandbytes accelerate beautifulsoup4

import os
import requests
from kaggle_secrets import UserSecretsClient
from IPython.display import display, HTML
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import json
import logging
import re
from datetime import datetime
from zoneinfo import ZoneInfo
from bs4 import BeautifulSoup
import random

logging.getLogger().setLevel(logging.ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# --- 1. CONFIGURAÇÃO E ACESSO AOS SECRETS ---
user_secrets = UserSecretsClient()
GOOGLE_API_KEY = user_secrets.get_secret("GOOGLE_API_KEY")
CUSTOM_SEARCH_CX = user_secrets.get_secret("CUSTOM_SEARCH_CX")

# --- 2. FUNÇÃO PARA OBTER A NOTÍCIA ---

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

# --- 3. FUNÇÃO PARA VERIFICAR E ENCONTRAR O MODELO ---
def encontrar_modelo_disponivel():
    """Tenta encontrar um modelo Gemma disponível no ambiente Kaggle"""
    
    # Lista de possíveis caminhos onde o modelo pode estar
    possiveis_caminhos = [
        "/kaggle/input/gemma/transformers/2b-it/1"      
      ]
    
    for caminho in possiveis_caminhos:
        try:
            print(f"Testando caminho: {caminho}")
            
            # Verifica se é um caminho local
            if caminho.startswith("/kaggle/"):
                if os.path.exists(caminho):
                    # Verifica se tem os arquivos necessários
                    arquivos_necessarios = ['config.json', 'tokenizer.json']
                    arquivos_encontrados = []
                    
                    for root, dirs, files in os.walk(caminho):
                        arquivos_encontrados.extend(files)
                    
                    if any(arquivo in arquivos_encontrados for arquivo in arquivos_necessarios):
                        print(f"✅ Modelo encontrado em: {caminho}")
                        return caminho
                    else:
                        print(f"❌ Caminho existe mas arquivos necessários não encontrados: {arquivos_encontrados}")
                else:
                    print(f"❌ Caminho não existe: {caminho}")
            else:
                # Tenta carregar do Hugging Face Hub
                try:
                    tokenizer = AutoTokenizer.from_pretrained(caminho)
                    print(f"✅ Modelo do HF Hub encontrado: {caminho}")
                    return caminho
                except Exception as e:
                    print(f"❌ Erro ao carregar do HF Hub {caminho}: {str(e)}")
                    continue
                    
        except Exception as e:
            print(f"❌ Erro ao testar {caminho}: {str(e)}")
            continue
    
    return None

# --- 4. FUNÇÃO PARA GERAR O DIÁLOGO ---
def gerar_dialogo(noticia):
    print("🔍 Procurando modelo disponível...")
    model_name = encontrar_modelo_disponivel()
    
    if model_name is None:
        print("❌ Nenhum modelo Gemma encontrado!")
        return gerar_dialogo_fallback(noticia)
    
    print(f"Gerando diálogo com IA (modelo {model_name})...")

    sagredo_icon_url = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='%232c5aa0' stroke-width='2'%3E%3Ccircle cx='12' cy='12' r='10'/%3E%3Cpath d='M8 14s1.5 2 4 2 4-2 4-2'/%3E%3Cline x1='9' y1='9' x2='9.01' y2='9'/%3E%3Cline x1='15' y1='9' x2='15.01' y2='9'/%3E%3C/svg%3E"
    Salviati_icon_url = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='%23c53030' stroke-width='2'%3E%3Ccircle cx='12' cy='12' r='10'/%3E%3Cpath d='M16 16s-1.5-2-4-2-4 2-4 2'/%3E%3Cline x1='9' y1='9' x2='9.01' y2='9'/%3E%3Cline x1='15' y1='9' x2='15.01' y2='9'/%3E%3C/svg%3E"
    simplicio_icon_url = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='%2338a169' stroke-width='2'%3E%3Ccircle cx='12' cy='12' r='10'/%3E%3Cline x1='8' y1='15' x2='16' y2='15'/%3E%3Cline x1='9' y1='9' x2='9.01' y2='9'/%3E%3Cline x1='15' y1='9' x2='15.01' y2='9'/%3E%3C/svg%3E"

    try:
        print("Carregando tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        print("Configurando quantização...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        print("Carregando modelo...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto"
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
2. Salviati (personalidade: crítico, pós-moderno)
3. Simplicio (personalidade: conservador, confiante)
4. Sagredo
5. Simplicio
6. Salviati

A resposta deve ser apenas o código HTML, contendo EXATAMENTE seis falas, uma para cada turno, com cada fala dentro de uma tag <p> e o nome do personagem em <strong>. Não inclua nenhum outro texto, aspas ou marcador de código.
<end_of_turn>
<start_of_turn>model
"""

        print("Tokenizando e gerando texto...")
        inputs = tokenizer(prompt_template, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=1024, do_sample=True,
                temperature=0.7, top_p=0.9
            )

        dialogo_html = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)

        print("Processando e formatando a saída...")
        
        # Primeiro, vamos limpar aspas desnecessárias
        dialogo_html = re.sub(r'(<strong>[^<]+</strong>\s*)"([^"]*)"', r'\1\2', dialogo_html)
        dialogo_html = re.sub(r"(<strong>[^<]+</strong>\s*)'([^']*)'", r'\1\2', dialogo_html)
        
        # Se não há tags <p>, vamos criar uma estrutura de diálogo formatada
        if '<p>' not in dialogo_html or dialogo_html.count('<p>') < 3:
            print("Reestruturando diálogo sem formatação adequada...")
            
            # Extrair falas dos personagens do texto bruto
            texto_limpo = re.sub(r'<[^>]+>', '', dialogo_html)  # Remove tags HTML
            
            # Padrões para identificar falas
            falas = []
            
            # Buscar padrões de fala
            padrao_sagredo = r'\*\*Sagredo\*\*:([^*]+?)(?=\*\*|$)'
            padrao_Salviati = r'\*\*Salviati\*\*:([^*]+?)(?=\*\*|$)'
            padrao_simplicio = r'\*\*Simplicio\*\*:([^*]+?)(?=\*\*|$)'
            
            for match in re.finditer(padrao_sagredo, texto_limpo):
                falas.append(('Sagredo', match.group(1).strip()))
            
            for match in re.finditer(padrao_Salviati, texto_limpo):
                falas.append(('Salviati', match.group(1).strip()))
                
            for match in re.finditer(padrao_simplicio, texto_limpo):
                falas.append(('Simplicio', match.group(1).strip()))
            
            # Se não encontrou falas estruturadas, usar fallback
            if len(falas) < 3:
                print("Texto muito desorganizado, usando diálogo fallback...")
                return gerar_dialogo_fallback(noticia)
            
            # Reorganizar falas na ordem encontrada no texto
            falas_ordenadas = []
            texto_para_busca = texto_limpo.lower()
            
            posicoes = []
            for personagem, fala in falas:
                pos = texto_para_busca.find(f'**{personagem.lower()}**:')
                if pos == -1:
                    pos = texto_para_busca.find(f'{personagem.lower()}:')
                posicoes.append((pos, personagem, fala))
            
            # Ordenar por posição no texto
            posicoes.sort(key=lambda x: x[0] if x[0] != -1 else 999999)
            
            # Construir HTML formatado
            dialogo_formatado = ""
            for _, personagem, fala in posicoes[:6]:  # Máximo 6 falas
                fala_limpa = fala.replace('"', '').replace("'", '').strip()
                if fala_limpa:
                    dialogo_formatado += f'<p><strong>{personagem}:</strong> {fala_limpa}</p>\n'
            
            dialogo_html = dialogo_formatado
        
        # Limpar formatação final
        dialogo_html = re.sub(r'</p>\s*</p>', '</p>', dialogo_html)
        dialogo_html = re.sub(r'<p>\s*<p>', '<p>', dialogo_html)
        dialogo_html = dialogo_html.strip()

        # Adicionar ícones apenas se não existirem
        icon_style = 'width="24" height="24" style="vertical-align: middle; margin-right: 8px;"'
        
        if 'img src=' not in dialogo_html:
            dialogo_html = dialogo_html.replace(
                "<strong>Sagredo:</strong>", f'<img src="{sagredo_icon_url}" {icon_style} alt="Sagredo"><strong>Sagredo:</strong>')
            dialogo_html = dialogo_html.replace(
                "<strong>Salviati:</strong>", f'<img src="{Salviati_icon_url}" {icon_style} alt="Salviati"><strong>Salviati:</strong>')
            dialogo_html = dialogo_html.replace(
                "<strong>Simplicio:</strong>", f'<img src="{simplicio_icon_url}" {icon_style} alt="Simplicio"><strong>Simplicio:</strong>')

        return str(dialogo_html.strip())

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ Erro ao gerar diálogo com IA: {str(e)}")
        return gerar_dialogo_fallback(noticia)

# --- 5. FUNÇÃO FALLBACK PARA GERAR DIÁLOGO ---
def gerar_dialogo_fallback(noticia):
    """Gera um diálogo básico quando o modelo IA não está disponível"""
    print("🔄 Gerando diálogo fallback...")
    
    sagredo_icon_url = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='%232c5aa0' stroke-width='2'%3E%3Ccircle cx='12' cy='12' r='10'/%3E%3Cpath d='M8 14s1.5 2 4 2 4-2 4-2'/%3E%3Cline x1='9' y1='9' x2='9.01' y2='9'/%3E%3Cline x1='15' y1='9' x2='15.01' y2='9'/%3E%3C/svg%3E"
    Salviati_icon_url = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='%23c53030' stroke-width='2'%3E%3Ccircle cx='12' cy='12' r='10'/%3E%3Cpath d='M16 16s-1.5-2-4-2-4 2-4 2'/%3E%3Cline x1='9' y1='9' x2='9.01' y2='9'/%3E%3Cline x1='15' y1='9' x2='15.01' y2='9'/%3E%3C/svg%3E"
    simplicio_icon_url = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='%2338a169' stroke-width='2'%3E%3Ccircle cx='12' cy='12' r='10'/%3E%3Cline x1='8' y1='15' x2='16' y2='15'/%3E%3Cline x1='9' y1='9' x2='9.01' y2='9'/%3E%3Cline x1='15' y1='9' x2='15.01' y2='9'/%3E%3C/svg%3E"
    icon_style = 'width="24" height="24" style="vertical-align: middle; margin-right: 8px;"'
    
    titulo = noticia.get('titulo', 'Notícia não disponível')
    
    dialogo = f"""<p><img src="{sagredo_icon_url}" {icon_style} alt="Sagredo"><strong>Sagredo:</strong> Ora, que interessante esta notícia sobre "{titulo}". Mas será que devemos confiar plenamente no que lemos nos jornais modernos?</p>

<p><img src="{Salviati_icon_url}" {icon_style} alt="Salviati"><strong>Salviati:</strong> Sagredo levanta uma questão pertinente. Vivemos numa era de informação fragmentada, onde cada notícia é apenas um recorte da realidade, moldado por interesses específicos.</p>

<p><img src="{simplicio_icon_url}" {icon_style} alt="Simplicio"><strong>Simplicio:</strong> Amigos, creio que vocês complicam demasiadamente as coisas. Se está nos jornais, especialmente em fontes respeitáveis, devemos considerar que há fundamento na informação.</p>

<p><img src="{sagredo_icon_url}" {icon_style} alt="Sagredo"><strong>Sagredo:</strong> Ah, Simplicio, sua fé na autoridade das fontes me impressiona! Mas não seria prudente questionar também as próprias bases dessas "fontes respeitáveis"?</p>

<p><img src="{simplicio_icon_url}" {icon_style} alt="Simplicio"><strong>Simplicio:</strong> O ceticismo excessivo nos levaria à paralisia total, Sagredo. É necessário confiar em alguma estrutura de conhecimento para que possamos avançar em nossa compreensão do mundo.</p>

<p><img src="{Salviati_icon_url}" {icon_style} alt="Salviati"><strong>Salviati:</strong> Talvez a verdade esteja no meio-termo: nem a credulidade cega de Simplicio, nem o ceticismo absoluto de Sagredo, mas uma postura crítica que avalie cada informação em seu contexto específico.</p>"""
    
    return dialogo

# --- 6. FUNÇÃO PARA MONTAR A PÁGINA HTML ---
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
            <h2>Sagredo, Salviati e Simplício no século 21</h2>
            {dialogo_html}
        </section>
    </main>
    <footer><p>Para gerar uma nova notícia e diálogo, clique em "Run All".</p></footer>
    </body></html>
    """

# --- 7. EXECUÇÃO PRINCIPAL ---
def main():
    print("🚀 Iniciando o programa...")
    
    noticia_atual = obter_noticia()
    if noticia_atual and noticia_atual["titulo"] not in ["Nenhuma notícia encontrada", "Erro de Conexão"]:
        print("📰 Notícia obtida com sucesso!")
        try:
            dialogo_gerado = str(gerar_dialogo(noticia_atual))
        except Exception as e:
            print(f"❌ Erro ao gerar diálogo: {str(e)}")
            dialogo_gerado = gerar_dialogo_fallback(noticia_atual)
    else:
        print("❌ Falha ao obter notícia")
        dialogo_gerado = "<p><em>Erro ao obter notícia.</em></p>"

    print("🎭 Tipo do diálogo:", type(dialogo_gerado))
    print("🎭 Montando página HTML...")
    pagina_completa = gerar_pagina_html(noticia_atual, dialogo_gerado)
    display(HTML(pagina_completa))
    print("✅ Processo concluído!")

# --- 8. EXECUÇÃO AUTOMÁTICA ---
# Executar o programa automaticamente
main()
