

import ast
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

MESES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun",
         "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]


@dataclass
class Preferencias:
    categorias: list = field(default_factory=list)      # p.ej. ["Local Experiences"]
    destinos: list = field(default_factory=list)        # vacío = cualquiera
    mes: int = 7                                        # 1-12, mes deseado del viaje
    precio_max: float = 200.0
    duracion_max: float = 8.0
    features: list = field(default_factory=list)        # p.ej. ["Free cancellation"]
    rating_min: float = 3.5
    flexible_mes: bool = True                           # aceptar propuestas en otro mes


class Motor:
    def __init__(self, experiencias_csv: str, presion_csv: str):
        self.exp = pd.read_csv(experiencias_csv)
        self.exp["Main Features"] = self.exp["Main Features"].apply(ast.literal_eval)
        self.presion = pd.read_csv(presion_csv).set_index("Experience ID")[MESES]
        self.presion = self.presion.loc[self.exp["Experience ID"]].reset_index(drop=True)
        # Disponibilidad: una experiencia con demanda 0 en un mes está cerrada
        dem_cols = [f"dem_{m}" for m in MESES]
        if all(c in self.exp.columns for c in dem_cols):
            self.abierto = self.exp[dem_cols].values > 0            # (n_exp x 12)
        else:
            self.abierto = np.ones((len(self.exp), 12), dtype=bool)

    # ---------- Capa 1: afinidad ----------
    def afinidad(self, pref: Preferencias) -> np.ndarray:
        e = self.exp
        # Categoría (peso 0.35)
        cat = (e["Category"].isin(pref.categorias).astype(float)
               if pref.categorias else np.ones(len(e)))
        # Destino (peso 0.15): si el usuario indica destinos, priorizarlos sin excluir
        dest = (np.where(e["Destination"].isin(pref.destinos), 1.0, 0.4)
                if pref.destinos else np.ones(len(e)))
        # Precio (peso 0.15): 1 dentro de presupuesto, decae al excederlo
        precio = np.clip(1 - (e["Price (EUR)"] - pref.precio_max) / pref.precio_max, 0, 1)
        # Duración (peso 0.10)
        dur = np.clip(1 - (e["Duration (hrs)"] - pref.duracion_max) / pref.duracion_max, 0, 1)
        # Features (peso 0.15): proporción de features deseadas presentes
        if pref.features:
            feats = e["Main Features"].apply(
                lambda fs: len(set(fs) & set(pref.features)) / len(pref.features))
            feats = feats.values.astype(float)
        else:
            feats = np.ones(len(e))
        # Rating (peso 0.10): escalado 3.5-5 -> 0-1
        rat = np.clip((e["Rating"] - 3.5) / 1.5, 0, 1)

        af = (0.35 * cat + 0.15 * dest + 0.15 * precio +
              0.10 * dur + 0.15 * feats + 0.10 * rat)
        af = np.where(e["Rating"] < pref.rating_min, af * 0.5, af)
        return af.astype(float)

    # ---------- Capa 2: score con presión ----------
    def recomendar(self, pref: Preferencias, alfa=0.6, beta=0.3, gamma=0.1,
                   top_n=5, con_redistribucion=True,
                   presion_extra=None) -> pd.DataFrame:
        """presion_extra: matriz opcional (n_exp x 12) con presión dinámica
        acumulada por reservas recientes (re-ranking dinámico)."""
        af = self.afinidad(pref)
        rat = np.clip((self.exp["Rating"] - 3.5) / 1.5, 0, 1).values

        candidatos = []
        meses_eval = range(12) if (pref.flexible_mes and con_redistribucion) else [pref.mes - 1]
        for m in meses_eval:
            pres = self.presion.iloc[:, m].values
            if presion_extra is not None:
                pres = np.minimum(pres + presion_extra[:, m], 1.5)
            pen_mes = 0.06 * min(abs(m + 1 - pref.mes), 12 - abs(m + 1 - pref.mes))
            score = (alfa * af - (beta * pres if con_redistribucion else 0)
                     + gamma * rat - (pen_mes if con_redistribucion else 0))
            score = np.where(self.abierto[:, m], score, -np.inf)  # cerradas ese mes
            candidatos.append(pd.DataFrame({
                "idx": np.arange(len(af)), "mes": m + 1, "afinidad": af,
                "presion": pres, "score": score}))
        c = pd.concat(candidatos)
        # mejor mes por experiencia y top-N global
        c = c.sort_values("score", ascending=False).drop_duplicates("idx").head(top_n)

        out = self.exp.loc[c["idx"], ["Experience ID", "Activity Name", "Category",
                                      "Destination", "Price (EUR)", "Duration (hrs)",
                                      "Rating", "pct_masificacion"]].copy()
        # Si el catálogo v2 aporta nombres descriptivos, usarlos como nombre visible
        if "Experience Name" in self.exp.columns:
            out["Activity Name"] = self.exp.loc[c["idx"], "Experience Name"].values
        out["Mes propuesto"] = [MESES[m - 1] for m in c["mes"]]
        out["Afinidad"] = (c["afinidad"].values * 100).round(0).astype(int)
        out["Presión"] = (c["presion"].values * 100).round(0).astype(int)
        out["Score"] = c["score"].values.round(3)
        out["Explicación"] = [self._explicar(r, pref) for _, r in out.iterrows()]
        return out.reset_index(drop=True)

    def _explicar(self, r, pref: Preferencias) -> str:
        partes = [f"{r['Afinidad']}% afinidad con tus preferencias"]
        pres_usuario_mes = self.presion.loc[
            self.exp["Experience ID"] == r["Experience ID"], MESES[pref.mes - 1]].iloc[0] * 100
        if r["Mes propuesto"] != MESES[pref.mes - 1]:
            partes.append(f"en {r['Mes propuesto']} la presión baja de "
                          f"{pres_usuario_mes:.0f} a {r['Presión']:.0f} (sobre 100)")
        elif r["Presión"] < 40:
            partes.append(f"presión turística baja ({r['Presión']}/100)")
        if r["Rating"] >= 4.5:
            partes.append(f"valoración excelente ({r['Rating']})")
        if r["pct_masificacion"] < 0.08:
            partes.append("pocas reseñas mencionan masificación")
        return " · ".join(partes)

    # ---------- Simulador de impacto ----------
    def simular(self, n_usuarios=6000, alfa=0.6, beta=0.3, gamma=0.1, seed=42,
                top_pool=5, delta_reserva=0.12):
        """Simula n usuarios eligiendo entre el top-5 recomendado, con y sin
        re-ranking. En el modo 'motor' la presión se actualiza dinámicamente con
        cada reserva simulada (re-ranking dinámico): delta_reserva por reserva
        en esa experiencia-mes, evitando desplazar el pico a otro mes.
        Devuelve reservas por experiencia/destino/mes."""
        rng = np.random.default_rng(seed)
        cats = self.exp["Category"].unique().tolist()
        dests = self.exp["Destination"].unique().tolist()
        feats_pool = sorted({f for fs in self.exp["Main Features"] for f in fs})
        # demanda estival concentrada: 60% de usuarios quieren Jun-Sep
        meses_p = np.array([2, 2, 3, 5, 8, 14, 18, 18, 10, 8, 6, 6], dtype=float)
        meses_p /= meses_p.sum()

        # mismas preferencias para ambos modos (comparación justa)
        prefs = [Preferencias(
            categorias=[rng.choice(cats)],
            destinos=[rng.choice(dests)] if rng.random() < 0.6 else [],
            mes=int(rng.choice(np.arange(1, 13), p=meses_p)),
            precio_max=float(rng.uniform(40, 250)),
            duracion_max=float(rng.uniform(2, 10)),
            features=list(rng.choice(feats_pool, size=2, replace=False)),
            flexible_mes=bool(rng.random() < 0.7),
        ) for _ in range(n_usuarios)]
        elecciones = rng.integers(0, top_pool, size=n_usuarios)

        id2idx = {eid: i for i, eid in enumerate(self.exp["Experience ID"])}
        mes2idx = {m: i for i, m in enumerate(MESES)}

        resultados = {}
        for modo, redis in [("baseline", False), ("motor", True)]:
            presion_extra = np.zeros((len(self.exp), 12)) if redis else None
            reservas = []
            for pref, ele in zip(prefs, elecciones):
                recs = self.recomendar(pref, alfa, beta, gamma,
                                       top_n=top_pool, con_redistribucion=redis,
                                       presion_extra=presion_extra)
                r = recs.iloc[int(ele) % len(recs)]
                reservas.append((r["Experience ID"], r["Destination"],
                                 r["Mes propuesto"]))
                if redis:  # actualización dinámica de presión
                    presion_extra[id2idx[r["Experience ID"]],
                                  mes2idx[r["Mes propuesto"]]] += delta_reserva
            df = pd.DataFrame(reservas, columns=["Experience ID", "Destination", "Mes"])
            resultados[modo] = df
        return resultados

    @staticmethod
    def metricas(df: pd.DataFrame) -> dict:
        por_exp = df["Experience ID"].value_counts()
        por_mes = df["Mes"].value_counts().reindex(MESES).fillna(0)
        return {
            "gini_experiencias": gini(por_exp.values),
            "pct_top10_experiencias": por_exp.head(max(1, int(len(por_exp) * 0.1))).sum() / len(df),
            "gini_mensual": gini(por_mes.values),
            "pct_verano": por_mes[["Jun", "Jul", "Ago"]].sum() / len(df),
            "n_experiencias_distintas": df["Experience ID"].nunique(),
        }


def gini(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))
