# eda_cars.py
# -*- coding: utf-8 -*-
import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from textwrap import dedent

NUMERIC_CANDIDATES = [
    "cc", "hp", "max_speed", "0_100_sec", "price", "seats", "torque", "battery_capacity"
]

CATEGORICAL_CANDIDATES = ["company", "model", "engine", "fuel_type"]

def coerce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    # strip espaços, padroniza nomes
    df = df.rename(columns={c: c.strip() for c in df.columns})
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].astype(str).str.strip()

    # força numéricos onde fizer sentido
    for col in NUMERIC_CANDIDATES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # categorias leves
    for col in CATEGORICAL_CANDIDATES:
        if col in df.columns:
            df[col] = df[col].replace({"nan": np.nan})
    return df

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def save_df(df: pd.DataFrame, path: Path):
    df.to_csv(path, index=False, encoding="utf-8")

def describe_dataframe(df: pd.DataFrame, outdir: Path):
    report_md = outdir / "README.md"
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in df.columns if df[c].dtype == "object"]

    # Missing values
    missing = df.isna().sum().sort_values(ascending=False).rename("missing_count").to_frame()
    missing["missing_pct"] = (missing["missing_count"] / len(df) * 100).round(2)
    missing.to_csv(outdir / "missing_values.csv")

    # Basic describe
    desc = df[numeric_cols].describe(percentiles=[.01,.05,.25,.5,.75,.95,.99]).T
    desc.to_csv(outdir / "numeric_describe.csv")

    # Cardinalidade de categóricas
    card = {c: df[c].nunique(dropna=True) for c in cat_cols}
    card_df = pd.DataFrame({"column": list(card.keys()), "n_unique": list(card.values())}).sort_values("n_unique", ascending=False)
    card_df.to_csv(outdir / "categorical_cardinality.csv", index=False)

    # Pequeno sumário em Markdown
    with open(report_md, "w", encoding="utf-8") as f:
        f.write(dedent(f"""
        # EDA – Veículos

        **Linhas:** {len(df)}  
        **Colunas:** {df.shape[1]}

        ## Colunas numéricas
        {", ".join(numeric_cols) if numeric_cols else "(nenhuma detectada)"}

        ## Colunas categóricas
        {", ".join(cat_cols) if cat_cols else "(nenhuma detectada)"}

        Arquivos gerados:
        - `missing_values.csv`
        - `numeric_describe.csv`
        - `categorical_cardinality.csv`
        - `correlation.csv` + `correlation_heatmap.png`
        - pasta `plots/` com histogramas, boxplots e dispersões
        - `derived_metrics.csv`, `grouped_stats_{col}.csv`
        - `toplists.csv`
        - `outliers_zscore.csv`
        """).strip())

def plot_histograms(df: pd.DataFrame, outdir: Path):
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    ensure_dir(outdir)
    for col in numeric_cols:
        try:
            plt.figure()
            df[col].dropna().plot(kind="hist", bins=30, edgecolor="black")
            plt.title(f"Histograma – {col}")
            plt.xlabel(col)
            plt.ylabel("Frequência")
            plt.tight_layout()
            plt.savefig(outdir / f"hist_{col}.png", dpi=150)
            plt.close()
        except Exception:
            plt.close()

def plot_boxplots_by_category(df: pd.DataFrame, category: str, outdir: Path):
    if category not in df.columns:
        return
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    ensure_dir(outdir)
    for col in numeric_cols:
        try:
            plt.figure(figsize=(8, 4))
            # boxplot com pandas/matplotlib
            data = [df.loc[df[category] == cat, col].dropna().values for cat in sorted(df[category].dropna().unique())]
            if len(data) == 0:
                plt.close(); continue
            plt.boxplot(data, labels=sorted(df[category].dropna().unique()))
            plt.title(f"{col} por {category}")
            plt.ylabel(col)
            plt.xlabel(category)
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            plt.savefig(outdir / f"box_{col}_by_{category}.png", dpi=150)
            plt.close()
        except Exception:
            plt.close()

def plot_scatter(df: pd.DataFrame, x: str, y: str, hue: str, outdir: Path):
    if x not in df.columns or y not in df.columns:
        return
    ensure_dir(outdir)
    try:
        plt.figure()
        if hue in df.columns:
            cats = df[hue].dropna().unique()
            for cat in cats:
                mask = df[hue] == cat
                plt.scatter(df.loc[mask, x], df.loc[mask, y], label=str(cat), alpha=0.7)
            plt.legend(title=hue, loc="best")
        else:
            plt.scatter(df[x], df[y], alpha=0.7)
        plt.xlabel(x); plt.ylabel(y)
        plt.title(f"{y} vs {x}")
        plt.tight_layout()
        plt.savefig(outdir / f"scatter_{y}_vs_{x}.png", dpi=150)
        plt.close()
    except Exception:
        plt.close()

def correlation(df: pd.DataFrame, outdir: Path):
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        return
    corr = df[numeric_cols].corr(numeric_only=True)
    corr.to_csv(outdir / "correlation.csv")
    # Heatmap simples (matplotlib)
    try:
        plt.figure(figsize=(0.8*len(numeric_cols)+3, 0.8*len(numeric_cols)+3))
        im = plt.imshow(corr.values, interpolation="nearest")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xticks(ticks=range(len(numeric_cols)), labels=numeric_cols, rotation=45, ha="right")
        plt.yticks(ticks=range(len(numeric_cols)), labels=numeric_cols)
        plt.title("Matriz de Correlação")
        plt.tight_layout()
        plt.savefig(outdir / "correlation_heatmap.png", dpi=150)
        plt.close()
    except Exception:
        plt.close()

def derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # hp por litro (cc em cm³)
    if "hp" in df.columns and "cc" in df.columns:
        df["hp_per_liter"] = df["hp"] / (df["cc"] / 1000.0)
    # torque por litro
    if "torque" in df.columns and "cc" in df.columns:
        df["torque_per_liter"] = df["torque"] / (df["cc"] / 1000.0)
    # preço por hp
    if "price" in df.columns and "hp" in df.columns:
        df["price_per_hp"] = df["price"] / df["hp"]
    # eficiência 0-100 vs hp (menor tempo é melhor): índice simples
    if "0_100_sec" in df.columns and "hp" in df.columns:
        df["accel_efficiency"] = df["hp"] / df["0_100_sec"]
    # Indicativo de EV
    if "battery_capacity" in df.columns:
        df["is_ev"] = df["battery_capacity"].fillna(0) > 0
    return df

def grouped_stats(df: pd.DataFrame, outdir: Path):
    for group_col in ["company", "fuel_type"]:
        if group_col in df.columns:
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if not num_cols:
                continue
            g = df.groupby(group_col, dropna=True)[num_cols].agg(["count","mean","median","std","min","max"])
            # flatten columns
            g.columns = ["_".join(filter(None, map(str, col))).strip("_") for col in g.columns.values]
            g = g.sort_values(by=[c for c in g.columns if c.endswith("_mean")][0] if any(c.endswith("_mean") for c in g.columns) else g.columns[0], ascending=False)
            g.to_csv(outdir / f"grouped_stats_{group_col}.csv")

def toplists(df: pd.DataFrame, outdir: Path):
    # Cria rankings úteis
    cols_exist = df.columns
    tops = []

    def add_top(name, sort_col, ascending, extra_cols=[]):
        use_cols = [c for c in ["company", "model", sort_col] + extra_cols if c in cols_exist]
        if sort_col in cols_exist and len(use_cols) >= 2:
            t = df[use_cols].dropna(subset=[sort_col]).sort_values(sort_col, ascending=ascending).head(20)
            t.insert(0, "list", name)
            tops.append(t)

    add_top("0-100 mais rápidos (menor é melhor)", "0_100_sec", True, ["hp", "max_speed", "price"])
    add_top("Maior velocidade máxima", "max_speed", False, ["hp", "0_100_sec", "price"])
    add_top("Maior potência (hp)", "hp", False, ["cc", "0_100_sec", "price"])
    add_top("Maior torque", "torque", False, ["hp", "price"])
    add_top("Maior hp por litro", "hp_per_liter", False, ["cc", "price"])
    add_top("Menor preço por hp", "price_per_hp", True, ["hp", "cc"])
    if "battery_capacity" in cols_exist:
        add_top("Maior capacidade de bateria (EVs)", "battery_capacity", False, ["price"])

    if tops:
        out = pd.concat(tops, ignore_index=True)
        out.to_csv(outdir / "toplists.csv", index=False)

def detect_outliers_zscore(df: pd.DataFrame, outdir: Path, z_thresh: float = 3.0):
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        return
    z = (df[num_cols] - df[num_cols].mean()) / df[num_cols].std(ddof=0)
    mask = (z.abs() > z_thresh).any(axis=1)
    outliers = df.loc[mask].copy()
    outliers.to_csv(outdir / "outliers_zscore.csv", index=False)

def main():
    parser = argparse.ArgumentParser(description="EDA para CSV de veículos.")
    parser.add_argument("--csv", required=True, help="Caminho para o CSV (com cabeçalho: company,model,engine,cc,hp,max_speed,0_100_sec,price,fuel_type,seats,torque,battery_capacity).")
    parser.add_argument("--out", default="eda_report", help="Pasta de saída para relatórios e gráficos.")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    outdir = Path(args.out)
    plots_dir = outdir / "plots"
    ensure_dir(outdir)
    ensure_dir(plots_dir)

    # Lê CSV
    df = pd.read_csv(csv_path)
    df = coerce_dtypes(df)

    # Métricas derivadas
    df = derived_metrics(df)
    save_df(df, outdir / "derived_metrics.csv")

    # Relatórios tabulares
    describe_dataframe(df, outdir)
    grouped_stats(df, outdir)
    toplists(df, outdir)

    # Correlação
    correlation(df, outdir)

    # Gráficos
    plot_histograms(df, plots_dir / "histograms")
    if "fuel_type" in df.columns:
        plot_boxplots_by_category(df, "fuel_type", plots_dir / "box_by_fuel_type")

    # Dispersões úteis
    plot_scatter(df, "hp", "price", "fuel_type", plots_dir / "scatters")
    plot_scatter(df, "hp", "0_100_sec", "fuel_type", plots_dir / "scatters")
    plot_scatter(df, "max_speed", "price", "fuel_type", plots_dir / "scatters")
    plot_scatter(df, "torque", "hp", "fuel_type", plots_dir / "scatters")
    if "battery_capacity" in df.columns:
        plot_scatter(df, "battery_capacity", "price", "fuel_type", plots_dir / "scatters")

    # Outliers
    detect_outliers_zscore(df, outdir, z_thresh=3.0)

    print(f"✅ EDA concluída. Saída em: {outdir.resolve()}")
    
if __name__ == "__main__":
    main()
