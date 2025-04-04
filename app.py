
import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

st.set_page_config(page_title="Classificador de Tickets", layout="wide")
st.title("🎯 Classificação de Tickets com IA")

colunas_texto = ['ASSUNTO', 'DESCRICAO', 'DESCRICAOTICKET', 'MENSAGEMERRO', 'CATEGORIA', 'MODULO', 'TIPO']
alvos = ['TIPO 2', 'MOTIVO', 'ROTINA', 'FREQUÊNCIA', 'IMPACTO']
modelos = {}

def normalizar_coluna(col):
    return col.fillna('').str.strip().str.lower().str.capitalize()

def balancear_dataframe(df, coluna_alvo):
    classes = df[coluna_alvo].value_counts()
    maior_classe = classes.idxmax()
    n_samples = classes.max()
    dfs_balanceados = []
    for valor in classes.index:
        df_classe = df[df[coluna_alvo] == valor]
        if len(df_classe) < 3:
            continue
        df_upsampled = resample(df_classe, replace=True, n_samples=n_samples, random_state=42)
        dfs_balanceados.append(df_upsampled)
    return pd.concat(dfs_balanceados)

def treinar_modelos(df):
    modelos = {}
    st.header("📊 Acurácia dos Modelos")
    df['TEXTO'] = df[colunas_texto].fillna('').agg(' '.join, axis=1)
    for alvo in alvos:
        df[alvo] = normalizar_coluna(df[alvo])
        df_filtrado = df.dropna(subset=[alvo])
        if df_filtrado[alvo].nunique() < 2:
            st.warning(f"⚠️ Coluna '{alvo}' possui menos de 2 classes após filtragem. Ignorada.")
            continue
        df_bal = balancear_dataframe(df_filtrado, alvo)
        X = df_bal['TEXTO']
        y = df_bal[alvo]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        pipe = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000)),
            ('clf', LogisticRegression(max_iter=1000))
        ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.markdown(f"### ✅ {alvo.upper()} — Acurácia: {acc:.2%}")
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        df_report = pd.DataFrame(report).transpose()
        st.dataframe(df_report.style.format("{:.2f}"))
        modelos[alvo] = pipe
        joblib.dump(pipe, f"modelo_{alvo}.pkl")
    return modelos

def aplicar_modelos(df):
    df['TEXTO'] = df[colunas_texto].fillna('').agg(' '.join, axis=1)
    for alvo in alvos:
        if alvo not in df.columns:
            continue
        try:
            modelo = joblib.load(f"modelo_{alvo}.pkl")
            cond_total = df[alvo].isna() | (df[alvo].astype(str).str.strip() == '')
            df.loc[cond_total, alvo] = modelo.predict(df.loc[cond_total, 'TEXTO'])
        except Exception as e:
            st.error(f"Erro ao aplicar modelo '{alvo}': {e}")
    return df

def gerar_graficos(df):
    st.header("📈 Análises dos Tickets Classificados")
    for col in ['TIPO 2', 'MOTIVO', 'ROTINA', 'FREQUÊNCIA', 'IMPACTO', 'MODULO', 'CLIENTE']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.capitalize()

    if 'ABERTURA' in df.columns:
        df['ABERTURA'] = pd.to_datetime(df['ABERTURA'], errors='coerce')
        df['MES_ABERTURA'] = df['ABERTURA'].dt.to_period('M').astype(str)

    if 'TIPO 2' in df.columns:
        st.subheader("✔️ Quantidade de Chamados por Tipo 2")
        fig1, ax1 = plt.subplots()
        df['TIPO 2'].value_counts().plot(kind='barh', ax=ax1)
        st.pyplot(fig1)

    if 'MOTIVO' in df.columns and 'ROTINA' in df.columns:
        st.subheader("✔️ Mapa de Calor: MOTIVO x ROTINA")
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        heat = pd.crosstab(df['MOTIVO'], df['ROTINA'])
        sns.heatmap(heat, cmap="YlGnBu", ax=ax2)
        st.pyplot(fig2)

    if 'FREQUÊNCIA' in df.columns and 'MES_ABERTURA' in df.columns:
        st.subheader("✔️ Frequência ao Longo do Tempo")
        fig3, ax3 = plt.subplots()
        freq_tempo = df.groupby(['MES_ABERTURA', 'FREQUÊNCIA']).size().unstack().fillna(0)
        freq_tempo.plot(marker='o', ax=ax3)
        st.pyplot(fig3)

    if 'IMPACTO' in df.columns and 'MODULO' in df.columns:
        st.subheader("✔️ Impacto por Módulo")
        fig4, ax4 = plt.subplots()
        imp = pd.crosstab(df['MODULO'], df['IMPACTO'])
        imp.plot(kind='barh', stacked=True, ax=ax4)
        st.pyplot(fig4)

    if 'FREQUÊNCIA' in df.columns and 'IMPACTO' in df.columns:
        st.subheader("✔️ Chamados Recorrentes com Impacto Alto")
        recorrente = df[
            df['FREQUÊNCIA'].str.lower().str.contains('recorrente') &
            df['IMPACTO'].str.lower().str.contains('alto')
        ]
        st.dataframe(recorrente)

    return df

# ETAPAS
st.sidebar.title("Etapas do Processo")

# 1️⃣ Upload da planilha de treinamento
arquivo_treinamento = st.sidebar.file_uploader("📁 Upload: Planilha de Treinamento", type=["xlsx"], key="treinamento")
if arquivo_treinamento:
    df_train = pd.read_excel(arquivo_treinamento)
    modelos = treinar_modelos(df_train)

# 2️⃣ Upload da planilha a classificar
arquivo_classificar = st.sidebar.file_uploader("📁 Upload: Planilha a Classificar", type=["xlsx"], key="classificacao")
if arquivo_classificar and modelos:
    df_novos = pd.read_excel(arquivo_classificar)
    st.subheader("🔎 Amostra dos Dados a Serem Classificados")
    st.dataframe(df_novos.head())
    df_classificado = aplicar_modelos(df_novos)
    gerar_graficos(df_classificado)

    # 3️⃣ Download ao final
    st.subheader("📥 Download da Planilha Classificada")
    buffer = BytesIO()
    df_classificado.to_excel(buffer, index=False, engine='openpyxl')
    st.download_button(
        label="⬇️ Baixar Planilha Classificada",
        data=buffer.getvalue(),
        file_name="Tickets_Classificados.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
