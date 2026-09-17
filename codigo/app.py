

import ast

import numpy as np
import pandas as pd
import streamlit as st

from motor import MESES, Motor, Preferencias

st.set_page_config(page_title="Motor TUI — Desafío 2", page_icon="🧭", layout="wide")


@st.cache_resource
def cargar_motor():
    return Motor("data/processed/experiencias.csv", "data/processed/presion.csv")


motor = cargar_motor()
exp = motor.exp

st.title("🧭 Motor de recomendación con redistribución de demanda")
st.caption("TUI Challenge — Desafío 2 · personalización + sostenibilidad + explicabilidad")

tab_rec, tab_sim, tab_datos = st.tabs(["Recomendador", "Simulador de impacto", "Datos"])

# ============================ RECOMENDADOR ============================
with tab_rec:
    col_form, col_res = st.columns([1, 2])

    with col_form:
        st.subheader("Tus preferencias")
        categorias = st.multiselect("Tipo de experiencia",
                                    sorted(exp["Category"].unique()),
                                    default=["Local Experiences"])
        destinos = st.multiselect("Destino preferido (opcional)",
                                  sorted(exp["Destination"].unique()))
        mes = st.select_slider("Mes del viaje", options=list(range(1, 13)),
                               value=8, format_func=lambda m: MESES[m - 1])
        precio_max = st.slider("Presupuesto máx. (€)", 20, 300, 150, 10)
        duracion_max = st.slider("Duración máx. (h)", 1.0, 12.0, 8.0, 0.5)
        feats_pool = sorted({f for fs in exp["Main Features"] for f in fs})
        features = st.multiselect("Características deseadas", feats_pool)
        flexible = st.checkbox("Acepto propuestas en otros meses", True)

        with st.expander("Pesos del motor (α afinidad / β presión / γ rating)"):
            alfa = st.slider("α afinidad", 0.0, 1.0, 0.6, 0.05)
            beta = st.slider("β presión", 0.0, 1.0, 0.3, 0.05)
            gamma = st.slider("γ rating", 0.0, 1.0, 0.1, 0.05)

        buscar = st.button("Recomiéndame", type="primary", use_container_width=True)

    with col_res:
        if buscar:
            pref = Preferencias(categorias=categorias, destinos=destinos, mes=mes,
                                precio_max=precio_max, duracion_max=duracion_max,
                                features=features, flexible_mes=flexible)
            recs = motor.recomendar(pref, alfa, beta, gamma, top_n=5)
            sin = motor.recomendar(pref, alfa, beta, gamma, top_n=5,
                                   con_redistribucion=False)

            st.subheader("Top 5 con redistribución")
            for _, r in recs.iterrows():
                with st.container(border=True):
                    c1, c2 = st.columns([3, 1])
                    c1.markdown(f"**{r['Activity Name']}** — {r['Destination']} · "
                                f"{r['Mes propuesto']} · {r['Price (EUR)']:.0f} € · "
                                f"⭐ {r['Rating']}")
                    c1.caption("💡 " + r["Explicación"])
                    c2.metric("Afinidad", f"{r['Afinidad']}%")
                    c2.progress(int(r["Presión"]), text=f"Presión {r['Presión']}/100")

            with st.expander("Comparar con motor tradicional (sin capa de redistribución)"):
                st.dataframe(sin[["Activity Name", "Destination", "Mes propuesto",
                                  "Afinidad", "Presión", "Rating"]],
                             use_container_width=True, hide_index=True)
                st.caption(f"Presión media top-5 · tradicional: {sin['Presión'].mean():.0f}/100"
                           f" · con redistribución: {recs['Presión'].mean():.0f}/100")
        else:
            st.info("Configura tus preferencias y pulsa **Recomiéndame**.")

# ============================ SIMULADOR ============================
with tab_sim:
    st.subheader("¿Qué pasaría si miles de viajeros usaran el motor?")
    st.caption("Se simulan usuarios con demanda concentrada en verano (60% Jun-Sep). "
               "Baseline = solo afinidad. Motor = afinidad + penalización por presión.")

    c1, c2, c3 = st.columns(3)
    n_usuarios = c1.slider("Nº de usuarios simulados", 200, 3000, 1000, 100)
    beta_sim = c2.slider("β presión (motor)", 0.0, 1.0, 0.3, 0.05, key="bsim")
    seed = c3.number_input("Semilla", value=42)

    if st.button("Ejecutar simulación", type="primary"):
        with st.spinner("Simulando..."):
            res = motor.simular(n_usuarios=n_usuarios, beta=beta_sim, seed=int(seed))
        m_base = Motor.metricas(res["baseline"])
        m_motor = Motor.metricas(res["motor"])

        st.markdown("#### Métricas de concentración")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Gini (experiencias)", f"{m_motor['gini_experiencias']:.3f}",
                  f"{m_motor['gini_experiencias'] - m_base['gini_experiencias']:+.3f}",
                  delta_color="inverse",
                  help="Desigualdad en el reparto de reservas entre experiencias (menos = mejor)")
        k2.metric("% en top 10% experiencias", f"{m_motor['pct_top10_experiencias']:.0%}",
                  f"{(m_motor['pct_top10_experiencias'] - m_base['pct_top10_experiencias']):+.0%}",
                  delta_color="inverse")
        k3.metric("% reservas Jun-Ago", f"{m_motor['pct_verano']:.0%}",
                  f"{(m_motor['pct_verano'] - m_base['pct_verano']):+.0%}",
                  delta_color="inverse",
                  help="Desestacionalización: cuota de reservas en temporada alta")
        k4.metric("Experiencias distintas", m_motor["n_experiencias_distintas"],
                  m_motor["n_experiencias_distintas"] - m_base["n_experiencias_distintas"])

        orden_meses = MESES
        st.markdown("#### Distribución mensual de reservas (antes / después)")
        dist = pd.DataFrame({
            "Baseline": res["baseline"]["Mes"].value_counts().reindex(orden_meses).fillna(0),
            "Con motor": res["motor"]["Mes"].value_counts().reindex(orden_meses).fillna(0),
        })
        st.bar_chart(dist)

        st.markdown("#### Reservas por destino")
        dist_d = pd.DataFrame({
            "Baseline": res["baseline"]["Destination"].value_counts(),
            "Con motor": res["motor"]["Destination"].value_counts(),
        }).fillna(0)
        st.bar_chart(dist_d)

# ============================ DATOS ============================
with tab_datos:
    st.subheader("Catálogo y presión turística")
    st.markdown(f"- **{len(exp):,} experiencias** en {exp['Destination'].nunique()} destinos, "
                f"{exp['Category'].nunique()} categorías\n"
                f"- Presión = 0.5·estacionalidad propia + 0.3·presión del destino "
                f"+ 0.2·masificación percibida en reseñas")
    dest_sel = st.selectbox("Destino", sorted(exp["Destination"].unique()))
    idx = exp["Destination"] == dest_sel
    pres_dest = motor.presion.loc[idx.values].mean()
    st.line_chart(pres_dest.rename("Presión media (0-1)"))
    st.dataframe(exp.loc[idx, ["Activity Name", "Category", "Price (EUR)",
                               "Rating", "pct_masificacion"]].head(50),
                 use_container_width=True, hide_index=True)
