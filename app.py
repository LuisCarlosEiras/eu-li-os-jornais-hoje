import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoConfig
from accelerate import infer_auto_device_map
import os

# --- Configuração da Página do Streamlit ---
st.set_page_config(
    page_title="Gemma 3n Chat",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Chat com Gemma 3n")
st.caption("Um aplicativo para interagir com o modelo google/gemma-3n-E2B-it.")

# --- Função de Carregamento do Modelo com Cache ---
# @st.cache_resource garante que o modelo seja carregado apenas uma vez.
@st.cache_resource
def load_model():
    """
    Carrega o modelo e o tokenizador com offloading manual.
    Esta função será executada apenas uma vez graças ao cache do Streamlit.
    """
    # Tenta obter o token do Hugging Face dos segredos do Streamlit
    huggingface_token = st.secrets.get("HUGGINGFACE_TOKEN")
    if not huggingface_token:
        st.error("Token do Hugging Face não encontrado! Por favor, adicione-o aos segredos do seu Space.")
        st.stop()
        
    model_id = "google/gemma-3n-E2B-it"

    # Configuração de Quantização
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Carrega a configuração do modelo
    config = AutoConfig.from_pretrained(model_id, token=huggingface_token)

    # Estratégia de Offloading Manual
    with torch.device("meta"):
        meta_model = AutoModelForCausalLM.from_config(config)

    device_map = infer_auto_device_map(
        meta_model,
        max_memory={0: "14GiB", "cpu": "10GiB"},
        dtype='bfloat16',
        token=huggingface_token
    )
    
    # Carrega o modelo de verdade usando o mapa customizado
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=huggingface_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=huggingface_token,
        quantization_config=quantization_config,
        device_map=device_map,
    )
    
    return model, tokenizer

# --- Lógica Principal do Aplicativo ---

# Exibe um spinner enquanto o modelo é carregado pela primeira vez
with st.spinner("Carregando o modelo... Isso pode levar alguns minutos na primeira vez."):
    model, tokenizer = load_model()

st.success("Modelo carregado com sucesso!")
st.markdown("---")

# Inicializa o histórico do chat na sessão do Streamlit
if "messages" not in st.session_state:
    st.session_state.messages = []

# Exibe as mensagens do histórico
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Captura a entrada do usuário
if prompt := st.chat_input("Qual a sua pergunta?"):
    # Adiciona a mensagem do usuário ao histórico e exibe
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Gera a resposta do modelo
    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            chat = [{"role": "user", "content": prompt}]
            prompt_formatado = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer.encode(prompt_formatado, add_special_tokens=False, return_tensors="pt").to("cuda")

            outputs = model.generate(input_ids=inputs, max_new_tokens=1024)
            resposta = tokenizer.decode(outputs[0], skip_special_tokens=True)
            resposta_limpa = resposta.split("model\n")[-1].strip()
            
            st.markdown(resposta_limpa)
    
    # Adiciona a resposta do modelo ao histórico
    st.session_state.messages.append({"role": "assistant", "content": resposta_limpa})
