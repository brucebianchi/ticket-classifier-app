import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from sklearn.metrics import classification_report, accuracy_score
import base64
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Classificador de Tickets", layout="wide")

colunas_texto = ['ASSUNTO', 'DESCRICAO', 'DESCRICAOTICKET', 'MENSAGEMERRO', 'CATEGORIA', 'MODULO', 'TIPO']
alvos = ['TIPO 2', 'MOTIVO', 'ROTINA', 'FREQUÊNCIA', 'IMPACTO']

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
    df['TEXTO'] = df[colunas_texto].fillna('').agg(' '.join, axis=1).str.strip()

    for alvo in alvos:
        df[alvo] = normalizar_coluna(df[alvo])
        df_filtrado = df.dropna(subset=[alvo])
        df_filtrado = df_filtrado[df_filtrado[alvo].str.strip() != '']

        if df_filtrado.empty or df_filtrado[alvo].nunique() < 2:
            st.warning(f"⚠️ Coluna '{alvo}' vazia ou com apenas uma classe.")
            continue

        df_balanceado = balancear_dataframe(df_filtrado, alvo)
        X = df_balanceado['TEXTO']
        y = df_balanceado[alvo]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000)),
            ('clf', LogisticRegression(max_iter=1000))
        ])
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.subheader(f"✅ {alvo} — Acurácia: {acc:.2%}")
        st.text(classification_report(y_test, y_pred, zero_division=0))
        modelos[alvo] = pipeline

    return modelos

def aplicar_modelos(df, modelos):
    df['TEXTO'] = df[colunas_texto].fillna('').agg(' '.join, axis=1).str.strip()

    for alvo in alvos:
        if alvo not in modelos:
            continue
        df[alvo] = df[alvo].astype(object)
        cond_total = df['TEXTO'].notna() & (df['TEXTO'].str.strip() != '')
        df.loc[cond_total, alvo] = modelos[alvo].predict(df.loc[cond_total, 'TEXTO'])

    return df.drop(columns=['TEXTO'])

def gerar_botao_download(df, nome_arquivo):
    csv = df.to_csv(index=False).encode()
    b64 = base64.b64encode(csv).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{nome_arquivo}">📥 Clique aqui para baixar a planilha classificada</a>'
    return href

def gerar_graficos(df):
    st.subheader("📊 Análises Gráficas")

    if 'TIPO 2' in df.columns:
        st.markdown("**Quantidade de Chamados por Tipo 2:**")
        fig1, ax1 = plt.subplots()
        sns.countplot(y='TIPO 2', data=df, order=df['TIPO 2'].value_counts().index, ax=ax1)
        st.pyplot(fig1)

    if 'MOTIVO' in df.columns and 'ROTINA' in df.columns:
        st.markdown("**Mapa de Calor: Motivo x Rotina:**")
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        heatmap_data = pd.crosstab(df['MOTIVO'], df['ROTINA'])
        sns.heatmap(heatmap_data, cmap='YlGnBu', linewidths=.5, ax=ax2)
        st.pyplot(fig2)

    if 'FREQUÊNCIA' in df.columns and 'ABERTURA' in df.columns:
        st.markdown("**Frequência ao longo do tempo:**")
        df['ABERTURA'] = pd.to_datetime(df['ABERTURA'], errors='coerce')
        df['MES_ABERTURA'] = df['ABERTURA'].dt.to_period('M').astype(str)
        freq_time = df.groupby(['MES_ABERTURA', 'FREQUÊNCIA']).size().unstack(fill_value=0)
        fig3, ax3 = plt.subplots()
        freq_time.plot(marker='o', ax=ax3)
        st.pyplot(fig3)

    if 'IMPACTO' in df.columns and 'MODULO' in df.columns:
        st.markdown("**Impacto por Módulo:**")
        impacto_modulo = pd.crosstab(df['MODULO'], df['IMPACTO'])
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        impacto_modulo.plot(kind='barh', stacked=True, ax=ax4)
        st.pyplot(fig4)

    if 'FREQUÊNCIA' in df.columns and 'IMPACTO' in df.columns:
        st.markdown("**Chamados recorrentes com impacto alto:**")
        filtro = df['FREQUÊNCIA'].str.contains('recorrente', case=False, na=False) & \
                 df['IMPACTO'].str.contains('alto', case=False, na=False)
        df_filtrado = df[filtro]
        if not df_filtrado.empty:
            st.dataframe(df_filtrado[['TICKET', 'ROTINA', 'MOTIVO', 'IMPACTO', 'FREQUÊNCIA', 'CLIENTE']])
        else:
            st.info("Nenhum chamado recorrente com impacto alto encontrado.")

# --- APP INICIA AQUI ---
st.title("🤖 Classificador Inteligente de Tickets")
st.markdown("Treine um modelo com tickets classificados e use para categorizar novos registros.")

with st.expander("1️⃣ Envie a planilha de **treinamento** (.xlsx)"):
    file_treinamento = st.file_uploader("Upload da planilha de treinamento", type="xlsx", key="treino")

if file_treinamento:
    df_treino = pd.read_excel(file_treinamento)
    st.success("✅ Planilha de treinamento carregada.")
    modelos = treinar_modelos(df_treino)

    with st.expander("2️⃣ Envie a planilha de **novos tickets** para classificar"):
        file_classificacao = st.file_uploader("Upload da planilha para classificação", type="xlsx", key="classifica")

    if file_classificacao:
        df_novos = pd.read_excel(file_classificacao)
        df_resultado = aplicar_modelos(df_novos.copy(), modelos)

        st.success("✅ Classificação concluída.")
        gerar_graficos(df_resultado)

        st.markdown("### 📥 Download da planilha classificada:")
        st.markdown(gerar_botao_download(df_resultado, "tickets_classificados.csv"), unsafe_allow_html=True)
