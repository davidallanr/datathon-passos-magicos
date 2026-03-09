import streamlit as st
import pandas as pd
import pickle
import os

BASE_DIR = os.path.dirname(__file__)
modelo_path = os.path.join(BASE_DIR, "modelo_risco.pkl")

modelo = pickle.load(open(modelo_path, "rb"))

st.title("Previsão de Risco Educacional")

st.write("""
Aplicação desenvolvida para identificar alunos em risco educacional
com base nos indicadores analisados no projeto Passos Mágicos.
""")

st.header("Insira os indicadores do aluno")

ida = st.slider("IDA - Desempenho Acadêmico", 0.0, 10.0, 5.0)
ieg = st.slider("IEG - Engajamento", 0.0, 10.0, 5.0)
ips = st.slider("IPS - Aspectos Psicossociais", 0.0, 10.0, 5.0)
ipp = st.slider("IPP - Indicador Psicopedagógico", 0.0, 10.0, 5.0)
ipv = st.slider("IPV - Ponto de Virada", 0.0, 10.0, 5.0)
inde = st.slider("INDE - Índice Educacional", 0.0, 10.0, 5.0)

dados = pd.DataFrame(
    [[ida, ieg, ips, ipp, ipv, inde]],
    columns=["IDA_2022","IEG_2022","IPS_2022","IPP_2022","IPV_2022","INDE_2022"]
)

if st.button("Prever risco educacional"):

    resultado = modelo.predict(dados)
    probabilidade = modelo.predict_proba(dados)

    risco = probabilidade[0][1] * 100

    st.subheader("Resultado da previsão")

    st.write(f"Probabilidade de risco educacional: **{risco:.2f}%**")
    st.progress(int(risco))

    if resultado[0] == 1:
        st.error("⚠️ Aluno em risco educacional")
    else:
        st.success("✅ Aluno com desenvolvimento educacional adequado")
        
        st.subheader("Indicadores que mais influenciam o risco")

importancias = modelo.feature_importances_

features = [
    "IDA - Desempenho Acadêmico",
    "IEG - Engajamento",
    "IPS - Aspectos Psicossociais",
    "IPP - Indicador Psicopedagógico",
    "IPV - Ponto de Virada",
    "INDE - Índice Educacional"
]

import pandas as pd

df_importancia = pd.DataFrame({
    "Indicador": features,
    "Importância": importancias
})

df_importancia = df_importancia.sort_values(by="Importância", ascending=True)

st.bar_chart(df_importancia.set_index("Indicador"))
