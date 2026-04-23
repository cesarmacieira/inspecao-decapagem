import streamlit as st
from PIL import Image
from ultralytics import YOLO
from pathlib import Path
import tempfile
import os
import numpy as np

st.set_page_config(
    page_title="SmartWeld — Inspeção de Solda",
    layout="wide"
)

# ─────────────────────────────────────────────────────────────────────────────
# Estilo
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 1.5rem; padding-bottom: 2.5rem; max-width: 1340px; }

.cabecalho-app {
    background: #1c2b3a; border-radius: 16px; padding: 2rem 2.4rem;
    margin-bottom: 1.8rem; display: flex; align-items: center;
    gap: 1.5rem; border-left: 5px solid #3d85c8;
}
.cabecalho-texto h1 { margin:0; font-size:1.85rem; font-weight:800; color:#f0f4f8; }
.cabecalho-texto p  { margin:0.3rem 0 0 0; font-size:0.95rem; color:#8daabf; }

.card-info  { background:#f7f9fb; border:1.5px solid #d5dde6; border-radius:14px; padding:1.2rem 1.3rem; }
.card-soft  { background:#ffffff; border:1.5px solid #dce5ef; border-radius:14px; padding:1rem 1.15rem; transition:box-shadow 0.2s; }
.card-soft:hover { box-shadow:0 2px 12px rgba(60,100,150,0.08); }

.bar-bg { background:#dde5ee; border-radius:999px; height:8px; overflow:hidden; margin-top:8px; }

.section-title {
    font-size:0.78rem; font-weight:700; color:#6b8399;
    letter-spacing:0.08em; text-transform:uppercase; margin-bottom:0.7rem;
}
.section-divider { border:none; border-top:1.5px solid #e4ecf4; margin:1.4rem 0; }
.small-muted { color:#7a90a4; font-size:0.88rem; }
.result-caption { text-align:center; color:#7a90a4; font-size:0.82rem; margin-top:0.3rem; font-style:italic; }

.stButton > button {
    background:#1c5d99; color:white; border:none; border-radius:10px;
    padding:0.6rem 1.2rem; font-weight:600; font-size:0.9rem;
    width:100%; transition:background 0.2s;
}
.stButton > button:hover { background:#164f85 !important; color:white !important; }

.stTabs [data-baseweb="tab-list"] { gap:4px; border-bottom:2px solid #e4ecf4; }
.stTabs [data-baseweb="tab"] {
    font-weight:600; font-size:0.9rem; padding:0.5rem 1.2rem;
    border-radius:8px 8px 0 0; color:#6b8399;
}
.stTabs [aria-selected="true"] { color:#1c5d99 !important; border-bottom:2px solid #1c5d99 !important; }

code { background:#edf2f7 !important; color:#2a7a4f !important;
    border-radius:5px; padding:1px 6px !important; font-size:0.83em !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("""<div class="cabecalho-app">
    <div class="cabecalho-texto">
        <h1>🔧 SmartWeld — Inspeção de Solda</h1>
        <p>Análise visual de defeitos em solda com apoio de modelo YOLO.</p>
    </div>
</div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent

MAPA_CLASSES_PT = {
    "adj": "Defeito adjacente",
    "int": "Defeito de integridade",
    "geo": "Defeito geométrico",
    "pro": "Defeito de pós-processamento",
    "non": "Defeito de não conformidade",
}

DESCRICOES_CLASSES = {
    "adj": "Indica defeitos relacionados a regiões adjacentes da solda.",
    "int": "Indica defeitos que comprometem a integridade da solda.",
    "geo": "Indica defeitos geométricos no cordão ou na forma da solda.",
    "pro": "Indica defeitos associados ao pós-processamento.",
    "non": "Indica defeitos ligados à não conformidade do processo ou do resultado.",
}

# ─────────────────────────────────────────────────────────────────────────────
# Funções auxiliares
# ─────────────────────────────────────────────────────────────────────────────
def texto_qtd_deteccoes(qtd):
    if qtd == 1:
        return "1 detecção encontrada"
    return f"{qtd} detecções encontradas"

def nome_classe_pt(nome_modelo):
    return MAPA_CLASSES_PT.get(nome_modelo, nome_modelo)

def descricao_classe(nome_modelo):
    return DESCRICOES_CLASSES.get(nome_modelo, "")

# ─────────────────────────────────────────────────────────────────────────────
# Modelo
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def carregar_modelo():
    caminho_modelo = BASE_DIR / "models" / "best.pt"
    if not caminho_modelo.exists():
        return None
    return YOLO(str(caminho_modelo))

modelo = carregar_modelo()

# ─────────────────────────────────────────────────────────────────────────────
# Estado da sessão
# ─────────────────────────────────────────────────────────────────────────────
for key, default in [('analisado', False), ('dados_resultado', None),
                     ('upload_key', 0), ('nome_arquivo', None)]:
    if key not in st.session_state:
        st.session_state[key] = default

# ─────────────────────────────────────────────────────────────────────────────
# Abas
# ─────────────────────────────────────────────────────────────────────────────
aba_analise, aba_info, aba_exemplos = st.tabs(['Análise', 'Sobre o Modelo', 'Exemplos'])

# ── Aba Análise ───────────────────────────────────────────────────────────────
with aba_analise:
    if modelo is None:
        st.error("Modelo não encontrado. Verifique se existe um arquivo best.pt dentro de models.")
        st.stop()

    st.markdown('<div class="section-title">Parâmetros de confiança</div>', unsafe_allow_html=True)
    col_cfg1, col_cfg2 = st.columns(2)
    with col_cfg1:
        confianca = st.number_input("Confiança mínima", min_value=0.0, max_value=1.0, value=0.25, step=0.05,
                                    help="Define o nível mínimo de probabilidade para considerar uma detecção válida.")
    with col_cfg2:
        iou = st.number_input("Sobreposição entre detecções", min_value=0.0, max_value=1.0, value=0.50, step=0.05,
                              help="Define o nível de sobreposição entre caixas para considerar detecções duplicadas.")

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Envio da imagem</div>', unsafe_allow_html=True)

    col_upload, col_preview = st.columns([1, 1], gap='large')
    with col_upload:
        arquivo = st.file_uploader("Selecione uma imagem", type=["jpg", "jpeg", "png"],
            key=f"uploader_{st.session_state.upload_key}")
    with col_preview:
        imagem = None
        if arquivo is not None:
            imagem = Image.open(arquivo).convert("RGB")
            st.markdown('<div class="section-title">Pré-visualização</div>', unsafe_allow_html=True)
            st.image(imagem, use_container_width=False, width=350)

    # Reset ao trocar arquivo
    if arquivo is None:
        st.session_state.analisado = False
        st.session_state.dados_resultado = None
        st.session_state.nome_arquivo = None
    else:
        if st.session_state.nome_arquivo != arquivo.name:
            st.session_state.analisado = False
            st.session_state.dados_resultado = None
            st.session_state.nome_arquivo = arquivo.name
    if arquivo is not None and imagem is not None:
        b1, b2, _ = st.columns([1, 1, 5])
        with b1:
            clicar_analisar = st.button("Analisar imagem", use_container_width=True, type="primary")
        with b2:
            clicar_limpar = st.button("Limpar", use_container_width=True)
        if clicar_limpar:
            st.session_state.upload_key += 1
            st.session_state.analisado = False
            st.session_state.dados_resultado = None
            st.session_state.nome_arquivo = None
            st.rerun()
        if clicar_analisar:
            with st.spinner("Processando imagem..."):
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                    imagem.save(tmp.name)
                    caminho_temporario = tmp.name
                try:
                    resultados = modelo.predict(caminho_temporario, conf=confianca, iou=iou, verbose=False)
                finally:
                    if os.path.exists(caminho_temporario):
                        os.unlink(caminho_temporario)
                boxes = resultados[0].boxes
                imagem_marcada = resultados[0].plot()
                deteccoes = []
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        score = float(box.conf[0])
                        classe_id = int(box.cls[0])
                        if isinstance(modelo.names, dict):
                            classe_modelo = modelo.names.get(classe_id, str(classe_id))
                        else:
                            classe_modelo = str(classe_id)
                        classe_pt = nome_classe_pt(classe_modelo)
                        descricao = descricao_classe(classe_modelo)
                        deteccoes.append({"classe_id": classe_id, "classe_modelo": classe_modelo,
                            "classe_pt": classe_pt, "descricao": descricao, "confianca": score})
                st.session_state.dados_resultado = {"imagem_marcada": imagem_marcada, "deteccoes": deteccoes}
                st.session_state.analisado = True
                st.rerun()

    if st.session_state.analisado and st.session_state.dados_resultado is not None:
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Análise</div>', unsafe_allow_html=True)
        col_imagem, col_resultado = st.columns([1.05, 1], gap='large')
        with col_imagem:
            st.markdown('<div class="section-title">Imagem com marcações</div>', unsafe_allow_html=True)
            st.image(st.session_state.dados_resultado["imagem_marcada"], use_container_width=False, width=350)
        with col_resultado:
            deteccoes = st.session_state.dados_resultado["deteccoes"]
            st.markdown('<div class="section-title">Resultado</div>', unsafe_allow_html=True)
            if not deteccoes:
                st.markdown('<div class="card-info">'
                    '<div style="font-size:1.9rem;font-weight:800;color:#2a7a4f;margin-top:0.2rem;">✅ Nenhum defeito detectado</div>'
                    '<div style="margin-top:0.4rem;color:#7a90a4;font-size:0.9rem;">'
                    'A imagem foi analisada e não houve identificação de defeitos dentro dos parâmetros definidos.</div>'
                    '</div>', unsafe_allow_html=True)
            else:
                quantidade = len(deteccoes)
                texto_quantidade = texto_qtd_deteccoes(quantidade)
                tipos = list(dict.fromkeys(d["classe_pt"] for d in deteccoes))
                confianca_media = sum(d["confianca"] for d in deteccoes) / quantidade
                st.markdown(f'<div class="card-info">'
                    f'<div style="font-size:1.9rem;font-weight:800;color:#c0392b;margin-top:0.2rem;">⚠️ {texto_quantidade}</div>'
                    f'<div style="margin-top:0.5rem;color:#5e6d7c;font-size:0.92rem;">'
                    f'<strong>Tipos identificados:</strong> {", ".join(tipos)}<br>'
                    f'<strong>Confiança média:</strong> {confianca_media:.4f}'
                    f'</div></div>', unsafe_allow_html=True)
        col_novo1, col_novo2, _ = st.columns([1.4, 1.4, 4])
        with col_novo1:
            if st.button("Analisar outra imagem", use_container_width=True):
                st.session_state.upload_key += 1
                st.session_state.analisado = False
                st.session_state.dados_resultado = None
                st.session_state.nome_arquivo = None
                st.rerun()
        with col_novo2:
            if st.button("Limpar resultado", use_container_width=True):
                st.session_state.analisado = False
                st.session_state.dados_resultado = None
                st.rerun()

# ── Aba Sobre o Modelo ────────────────────────────────────────────────────────
with aba_info:
    st.markdown("## SmartWeld — Inspeção de Defeitos em Solda")
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.metric("Modelo", "YOLOv8", help="Arquitetura utilizada para detecção")
    with col_m2:
        st.metric("Classes de defeito", str(len(MAPA_CLASSES_PT)), help="Número de tipos de defeito detectáveis")
    with col_m3:
        st.metric("Confiança padrão", "25%", help="Limiar mínimo padrão de confiança")

    st.write("""
    O SmartWeld utiliza um modelo YOLOv8 treinado para detectar e localizar defeitos
    em imagens de solda industrial. O modelo identifica regiões com anomalias e as
    classifica em categorias funcionais de defeito.

    **Pipeline deste app:**
    - **Etapa 1** — Recepção da imagem e configuração dos parâmetros de inferência
    - **Etapa 2** — Inferência com YOLOv8 (`best.pt`) sobre a imagem enviada
    - **Etapa 3** — Exibição das regiões detectadas com classe e confiança
    """)

    st.markdown('<div class="section-title">Classes de defeitos</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    for i, (cls, nome_pt) in enumerate(MAPA_CLASSES_PT.items()):
        with (c1 if i % 2 == 0 else c2):
            st.markdown(
                f'<div class="card-soft" style="margin-bottom:0.85rem;">'
                f'<div style="font-weight:700;font-size:1.05rem;">'
                f'{nome_pt} <code>{cls}</code></div>'
                f'<div style="margin-top:0.45rem;color:#5e6d7c;">{DESCRICOES_CLASSES.get(cls, "")}</div>'
                f'</div>', unsafe_allow_html=True
            )

# ── Aba Exemplos ──────────────────────────────────────────────────────────────
with aba_exemplos:
    st.caption("Faça download e envie na aba **Análise** para testar a detecção.")
    st.markdown("### Imagens de exemplo")
    exemplos = [("Exemplo 1", BASE_DIR / "imagem1_exemplo.jpg"), ("Exemplo 2", BASE_DIR / "imagem2_exemplo.jpg"),
        ("Exemplo 3", BASE_DIR / "imagem3_exemplo.jpg")]
    cols = st.columns(3)
    for i, (label, path) in enumerate(exemplos):
        with cols[i]:
            st.markdown(f"**{label}**")
            if path.exists():
                st.image(Image.open(path), use_container_width=True)
                with open(path, 'rb') as f:
                    st.download_button("Baixar", f, file_name=path.name, mime="image/jpeg", key=f"dl_img_{i}")
            else:
                st.warning(f"{path.name} não encontrada.")