

import argparse
import os

import pandas as pd

CIUDAD_NORM = {
    "Sevilla": "Seville", "Málaga": "Malaga", "Mallorca": "Palma de Mallorca",
}

KEYWORDS = (
    # inglés
    r"crowd|queue|busy|packed|overcrowd|too many people|"
    # español
    r"masificad|mucha gente|lleno de gente|llena de gente|aglomerac|abarrotad|saturad|las colas|hacer cola|"
    # alemán
    r"überfüllt|warteschlange|viele leute|"
    # francés
    r"bondé|foule|beaucoup de monde|file d'attente|"
    # italiano
    r"affollat|troppa gente|"
    # portugués / neerlandés
    r"lotado|wachtrij"
)


def extraer(maestro_csv: str, out_dir: str = "data/external") -> None:
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(maestro_csv, usecols=["Texto", "Rating", "Ciudad", "Fecha", "Fuente"])
    df = df[df["Fuente"] != "TUI"].copy()          # solo reseñas reales
    df["Ciudad"] = df["Ciudad"].replace(CIUDAD_NORM)
    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
    df = df.dropna(subset=["Fecha"])
    df["mes"] = df["Fecha"].dt.month

    df["crowd"] = df["Texto"].astype(str).str.contains(KEYWORDS, case=False, regex=True)
    df["malo"] = df["Rating"] == "Malo"

    agg = (df.groupby(["Ciudad", "mes"])
             .agg(n=("crowd", "size"),
                  pct_masificacion=("crowd", "mean"),
                  pct_malo=("malo", "mean"))
             .reset_index())
    out = os.path.join(out_dir, "masificacion_ciudad_mes.csv")
    agg.to_csv(out, index=False)
    print(f"OK: {len(df):,} reseñas reales -> {out}")
    print(agg.groupby("Ciudad")[["pct_masificacion", "pct_malo"]].mean().round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--maestro", required=True, help="Ruta a dataset_maestro.csv (Smart Touring)")
    ap.add_argument("--out", default="data/external")
    args = ap.parse_args()
    extraer(args.maestro, args.out)
