

import argparse
import ast
import glob
import os

import numpy as np
import pandas as pd

MESES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun",
         "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]

# catálogo -> columna del CSV de Google Trends
MAPEO_TRENDS = {"Malaga": "Málaga", "Seville": "Sevilla"}

# Perfil mensual de viajes por destino (% del año; conocimiento de dominio

PERFIL_DESTINO = {
    "Barcelona":         [6.0, 6.5, 7.5, 8.5, 9.5, 10.0, 10.5, 10.0, 9.5, 8.5, 7.0, 6.5],
    "Madrid":            [7.5, 8.0, 8.5, 9.0, 9.5, 8.5, 7.0, 6.5, 9.0, 9.5, 8.5, 8.5],
    "Seville":           [6.5, 7.5, 9.5, 11.5, 10.5, 7.5, 5.5, 6.0, 9.5, 10.5, 8.0, 7.5],
    "Valencia":          [6.5, 7.0, 8.0, 8.5, 9.0, 9.5, 11.0, 10.5, 9.0, 8.0, 6.5, 6.5],
    "Malaga":            [5.5, 6.0, 7.5, 8.5, 9.5, 10.5, 12.0, 12.0, 10.0, 8.0, 5.5, 5.0],
    "Palma de Mallorca": [3.0, 3.5, 5.0, 7.5, 10.0, 12.5, 14.5, 14.0, 12.0, 9.0, 5.0, 4.0],
    "Tenerife":          [9.5, 9.0, 9.0, 8.0, 7.5, 7.5, 8.5, 9.0, 7.5, 8.0, 8.0, 8.5],
    "Gran Canaria":      [9.5, 9.0, 9.0, 8.0, 7.5, 7.5, 8.5, 9.0, 7.5, 8.0, 8.0, 8.5],
}
PESO_PERFIL = 0.65   # mezcla perfil de dominio vs Google Trends

# Exponente de estacionalidad por categoría (>1 amplifica, <1 aplana).
GAMMA_CATEGORIA = {
    "Excursions & Day Trips": 1.40,
    "Local Experiences": 1.15,
    "Transfers": 1.10,
    "Attractions & Guided Tours": 0.95,
    "Tickets and Events": 0.80,
}

DIAS_MES = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


def pesos_mensuales(cat: pd.DataFrame, trends_csv: str, rng) -> np.ndarray:
    """Matriz (n_exp x 12) de pesos mensuales normalizados."""
    it = pd.read_csv(trends_csv)
    it["mes"] = pd.to_datetime(it["date"]).dt.month
    base_ciudad = it.drop(columns="date").groupby("mes").mean()      # 12 x ciudades

    W = np.zeros((len(cat), 12))
    for i, (dest, categ) in enumerate(zip(cat["Destination"], cat["Category"])):
        col = MAPEO_TRENDS.get(dest, dest)
        trends = base_ciudad[col].values.astype(float)
        trends = trends / trends.sum()
        perfil = np.array(PERFIL_DESTINO[dest]) / 100.0
        w = PESO_PERFIL * perfil + (1 - PESO_PERFIL) * trends        # mezcla realista
        w = w ** GAMMA_CATEGORIA[categ]                              # modulación por categoría
        w = w * rng.lognormal(0.0, 0.25, size=12)                    # idiosincrasia
        W[i] = w / w.sum()

    # 5% de experiencias estacionales puras: cierran sus 5 meses más flojos
    n_estacionales = int(0.05 * len(cat))
    idx = rng.choice(len(cat), size=n_estacionales, replace=False)
    for i in idx:
        cerrados = np.argsort(W[i])[:5]
        W[i, cerrados] = 0.0
        W[i] = W[i] / W[i].sum()
    return W


def generar(raw_dir: str, out_dir: str, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)

    cat = pd.read_csv(os.path.join(raw_dir, "experiences_catalog_v1.csv"))
    demanda_v1 = np.array(cat["Monthly Availability"].apply(ast.literal_eval).tolist())
    total_anual = demanda_v1.sum(axis=1)                             # se conserva el volumen

    trends_csv = os.path.join("data", "external",
                              "interes_turistico_mensual_por_ciudad.csv")
    W = pesos_mensuales(cat, trends_csv, rng)
    demanda_v2 = np.rint(W * total_anual[:, None]).astype(int)

    # ---------- catálogo v2 ----------
    from nombres import generar_nombres
    cat_v2 = cat.copy()
    cat_v2["Monthly Availability"] = [str([int(x) for x in f]) for f in demanda_v2]
    nombres, descs = generar_nombres(cat_v2, seed)
    cat_v2.insert(2, "Experience Name", nombres)
    cat_v2.insert(3, "Short Description", descs)
    cat_v2.to_csv(os.path.join(out_dir, "experiences_catalog.csv"), index=False)

    # ---------- matriz de disponibilidad v2 (consistente con el catálogo) ----------
    filas = []
    for eid, fila in zip(cat["Experience ID"], demanda_v2):
        for m in range(12):
            filas.append((eid, f"2025-{m+1:02d}", bool(fila[m] > 0), int(fila[m])))
    pd.DataFrame(filas, columns=["Experience ID", "Month", "Is Available",
                                 "Average Monthly Bookings"]) \
      .to_csv(os.path.join(out_dir, "availability_matrix.csv"), index=False)

    # ---------- bookings v2: re-muestrear el mes del viaje ----------
    files = sorted(glob.glob(os.path.join(raw_dir, "Customer Bookings", "*.csv")))
    book = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    book["Travel Date"] = pd.to_datetime(book["Travel Date"])
    book["Booking Date"] = pd.to_datetime(book["Booking Date"])
    lead = (book["Travel Date"] - book["Booking Date"]).dt.days.clip(lower=0)

    id2row = {eid: i for i, eid in enumerate(cat["Experience ID"])}
    nuevo_mes = np.empty(len(book), dtype=int)
    for eid, grupo in book.groupby("Experience ID").groups.items():
        w = W[id2row[eid]]
        nuevo_mes[grupo] = rng.choice(12, size=len(grupo), p=w) + 1

    # Se conservan los años originales: histórico hasta diciembre de 2025
    anio = book["Travel Date"].dt.year.values
    dia = np.minimum(book["Travel Date"].dt.day.values,
                     np.array(DIAS_MES)[nuevo_mes - 1])
    travel_v2 = pd.to_datetime(pd.DataFrame({"year": anio, "month": nuevo_mes, "day": dia}))
    book["Travel Date"] = travel_v2.dt.strftime("%Y-%m-%d")
    book["Booking Date"] = (travel_v2 - pd.to_timedelta(lead, unit="D")).dt.strftime("%Y-%m-%d")
    book.to_csv(os.path.join(out_dir, "bookings.csv"), index=False)

    # ---------- resumen ----------
    dist = pd.Series(nuevo_mes).value_counts(normalize=True).sort_index() * 100
    print("Distribución mensual de reservas v2 (%):")
    print("  " + "  ".join(f"{MESES[m-1]} {v:.1f}" for m, v in dist.items()))
    b2 = book.merge(cat[["Experience ID", "Destination"]], on="Experience ID")
    b2["mes"] = pd.to_datetime(b2["Travel Date"]).dt.month
    inv = b2[b2["mes"].isin([11, 12, 1, 2, 3])].groupby("Destination").size()
    tot = b2.groupby("Destination").size()
    print("\n% reservas nov-mar por destino:")
    print(((inv / tot) * 100).round(1).to_string())
    print(f"\nOK -> {out_dir}: experiences_catalog.csv, availability_matrix.csv, bookings.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw", help="Carpeta con los datos TUI originales")
    ap.add_argument("--out", default="../datos")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    generar(args.raw, args.out, args.seed)
