import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss, f1_score

# Configurações iniciais
st.set_page_config(page_title="Dashboard Kobe Bryant", layout="wide")
st.title("🏀 Dashboard - Predições do Modelo de Arremessos do Kobe")

# Caminho padrão
default_path = "data/processed/predicoes_producao.parquet"

# Upload de arquivo
st.sidebar.header("📂 Upload de arquivo de predição")
uploaded_file = st.sidebar.file_uploader("Escolha um arquivo .parquet", type=["parquet"])

# Carregamento do arquivo
if uploaded_file:
    df = pd.read_parquet(uploaded_file)
else:
    if os.path.exists(default_path):
        df = pd.read_parquet(default_path)
        st.sidebar.info("Usando arquivo padrão de produção.")
    else:
        st.error("Nenhum arquivo encontrado.")
        st.stop()

# Mostrar preview
st.subheader("🔎 Visualização geral dos dados")
st.dataframe(df.head())

# Se tiver coluna de verdade (shot_made_flag), mostra métricas
if "shot_made_flag" in df.columns:
    y_true = df["shot_made_flag"]
    y_pred = df["prediction"]
    y_score = df["score"]

    try:
        logloss = log_loss(y_true, y_score)
        f1 = f1_score(y_true, y_pred)
        st.metric("Log Loss", f"{logloss:.4f}")
        st.metric("F1 Score", f"{f1:.4f}")
    except:
        st.warning("Não foi possível calcular métricas (valores inválidos?)")

# Gráfico: Acertos vs Erros por distância
st.subheader("📊 Acertos x Erros por Distância")
fig1, ax1 = plt.subplots()
df["prediction"].value_counts().plot(kind="bar", ax=ax1)
ax1.set_title("Distribuição de Previsões")
ax1.set_xticklabels(["Erro (0)", "Cesta (1)"], rotation=0)
st.pyplot(fig1)

# Gráfico: Distribuição por shot_distance
st.subheader("🏹 Distância dos Arremessos")
fig2, ax2 = plt.subplots()
df.hist("shot_distance", by="prediction", bins=30, ax=ax2, figsize=(10, 4), sharex=True)
st.pyplot(fig2)

# Gráfico: Dispersão geográfica (lat x lon)
st.subheader("📍 Mapa de Arremessos (Lat/Lon)")
fig3, ax3 = plt.subplots()
colors = df["prediction"].map({0: "red", 1: "green"})
ax3.scatter(df["lon"], df["lat"], c=colors, alpha=0.5)
ax3.set_xlabel("Longitude")
ax3.set_ylabel("Latitude")
ax3.set_title("Localização dos Arremessos")
st.pyplot(fig3)

# Extras
st.markdown("---")
st.caption("Desenvolvido com ❤️ usando Streamlit")
