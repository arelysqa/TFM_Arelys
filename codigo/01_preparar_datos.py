

import argparse
import ast
import glob
import os

import numpy as np
import pandas as pd

MESES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun",
         "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]

KEYWORDS_MASIFICACION = "crowd|queue|busy|packed|full of tourists"


def preparar(raw_dir: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # ---------- Catálogo ----------
    cat_v2 = os.path.join(raw_dir, "experiences_catalog.csv")
    cat_path = cat_v2 if os.path.exists(cat_v2) else os.path.join(raw_dir, "experiences_catalog_v1.csv")
    print(f"Catálogo: {os.path.basename(cat_path)}")
    cat = pd.read_csv(cat_path)
    cat["Main Features"] = cat["Main Features"].apply(ast.literal_eval)
    demanda = np.array(cat["Monthly Availability"].apply(ast.literal_eval).tolist())  # (n_exp, 12)

    # ---------- masificación percibida ----------
    rev_files = glob.glob(os.path.join(raw_dir, "Review Dataset", "reviews_V1_part*.csv"))
    if rev_files:
        rev = pd.concat([pd.read_csv(f) for f in rev_files], ignore_index=True)
        rev["crowd"] = rev["Review Text"].str.contains(KEYWORDS_MASIFICACION, case=False)
        agg = rev.groupby("Experience ID").agg(
            pct_masificacion=("crowd", "mean"),
            sentimiento_medio=("Sentiment Score", "mean"),
        )
        cat = cat.merge(agg, left_on="Experience ID", right_index=True, how="left")
        cat["pct_masificacion"] = cat["pct_masificacion"].fillna(agg["pct_masificacion"].median())
        cat["sentimiento_medio"] = cat["sentimiento_medio"].fillna(agg["sentimiento_medio"].median())
    else:
        print("AVISO: no se encontraron reseñas; se omite la masificación percibida.")
        cat["pct_masificacion"] = 0.0
        cat["sentimiento_medio"] = 0.5

    # ---------- presión por experiencia y mes ----------
    # Componente 1 (estacionalidad propia): demanda del mes / mes pico de esa experiencia.
    presion_exp = demanda / (demanda.max(axis=1, keepdims=True) + 1e-9)


    if os.path.exists(interes_path):
        MAPEO = {"Malaga": "Málaga", "Seville": "Sevilla"}  # catálogo -> CSV
        it = pd.read_csv(interes_path)
        it["mes"] = pd.to_datetime(it["date"]).dt.month
        media_mensual = it.drop(columns="date").groupby("mes").mean()  # 12 x ciudades
        presion_dest = (media_mensual / media_mensual.max()).T          # ciudades x 12
        presion_dest.columns = MESES
        ciudades = cat["Destination"].map(lambda d: MAPEO.get(d, d))
        presion_dest_exp = presion_dest.loc[ciudades].values
        print("OK: presión de destino con estacionalidad REAL (Google Trends).")
    else:
        df_dem = pd.DataFrame(demanda, columns=MESES)
        df_dem["Destination"] = cat["Destination"].values
        dest_month = df_dem.groupby("Destination").sum()
        presion_dest = dest_month.div(dest_month.max(axis=1), axis=0)
        presion_dest_exp = presion_dest.loc[cat["Destination"]].values
        print("AVISO: sin datos externos; presión de destino con demanda sintética.")

    # masificación percibida- Preferencia: señal REAL por

    masif_path = os.path.join("data", "external", "masificacion_ciudad_mes.csv")
    if os.path.exists(masif_path):
        ms = pd.read_csv(masif_path)
        piv_c = ms.pivot(index="Ciudad", columns="mes", values="pct_masificacion")
        piv_m = ms.pivot(index="Ciudad", columns="mes", values="pct_malo")
        norm = lambda M: (M - M.min().min()) / (M.max().max() - M.min().min() + 1e-9)
        percibida_cm = 0.6 * norm(piv_c) + 0.4 * norm(piv_m)   # ciudades x 12
        presion_percibida = percibida_cm.loc[cat["Destination"]].values
        print("OK: masificación percibida REAL por ciudad-mes (410k reseñas).")
    else:
        pm = cat["pct_masificacion"].values
        presion_percibida = ((pm - pm.min()) / (pm.max() - pm.min() + 1e-9))[:, None]
        print("AVISO: masificación percibida desde reseñas sintéticas de TUI.")

    presion = 0.5 * presion_exp + 0.3 * presion_dest_exp + 0.2 * presion_percibida
    # Reescalar a [0, 1] para que el re-ranking tenga contraste completo
    presion = (presion - presion.min()) / (presion.max() - presion.min())

    presion_df = pd.DataFrame(presion, columns=MESES)
    presion_df.insert(0, "Experience ID", cat["Experience ID"])

    # ---------- Guardar ----------
    cat["Main Features"] = cat["Main Features"].apply(repr)  # serializable
    cat = cat.drop(columns=["Monthly Availability"]).join(
        pd.DataFrame(demanda, columns=[f"dem_{m}" for m in MESES]))
    cat.to_csv(os.path.join(out_dir, "experiencias.csv"), index=False)
    presion_df.to_csv(os.path.join(out_dir, "presion.csv"), index=False)

    print(f"OK: {len(cat)} experiencias -> {out_dir}/experiencias.csv")
    print(f"OK: presión ({presion_df.shape[0]} x 12 meses) -> {out_dir}/presion.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw", help="Carpeta con los datos crudos de TUI")
    ap.add_argument("--out", default="data/processed")
    args = ap.parse_args()
    preparar(args.raw, args.out)
