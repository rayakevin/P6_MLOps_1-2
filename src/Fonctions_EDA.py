#uv add numpy pandas scikit-learn matplotlib seaborn category-encoders scipy joblib tqdm

# Fonctions.py
import numpy as np
import pandas as pd
from IPython.display import Markdown, display


def eda_overview(df: pd.DataFrame) -> None:
    """
    Affiche un aperçu EDA (Exploratory Data Analysis) d'un DataFrame.

    Le rapport est organisé en 5 sections :
    1. Vue globale du dataset (taille, doublons)
    2. Qualité des colonnes (types, valeurs manquantes, cardinalité)
    3. Variables numériques
       3.1 Complétude (%NaN+%0, %NaN, %0)
       3.2 Distribution générale (%outliers, nb_outliers_bas/haut, quartiles, skew, kurtosis)
    4. Variables catégorielles (cardinalité, manquants, modalité dominante)
    5. Variables temporelles (min/max, manquants)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame à analyser.

    Returns
    -------
    None
        La fonction affiche des tableaux avec `display` et ne retourne rien.
    """
    # 1) Vue globale du dataset.
    display(Markdown("## 1) Vue globale"))
    n_rows, n_cols = df.shape
    n_dup = df.duplicated().sum()

    global_df = pd.DataFrame(
        {
            "n_lignes": [n_rows],
            "n_colonnes": [n_cols],
            "doublons": [n_dup],
            "%_doublons": [round(n_dup / n_rows * 100, 2) if n_rows else 0],
        }
    )
    display(global_df)

    # 2) Qualité des colonnes : types, manquants, cardinalité.
    display(Markdown("## 2) Qualité des colonnes et cardinalité"))
    col_report = pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "%null": (df.isna().mean() * 100).round(2),
            "non_null": df.notna().sum(),
            "null": df.isna().sum(),
            "n_uniques": df.nunique(dropna=True),
        }
    )
    col_report = col_report.sort_values(["dtype", "%null"], ascending=[True, False])
    display(col_report)

    # 3) Variables numériques.
    num_cols = df.select_dtypes(include=np.number).columns
    display(Markdown(f"## 3) Variables numériques ({len(num_cols)})"))
    if len(num_cols):
        # Metriques de complétude.
        pct_nan = (df[num_cols].isna().mean() * 100).round(2)
        pct_zero = ((df[num_cols] == 0).sum() / n_rows * 100).round(2) if n_rows else 0
        pct_nan_zero = (pct_nan + pct_zero).round(2)

        completion_report = pd.DataFrame(
            {
                "%NaN+%0": pct_nan_zero,
                "%NaN": pct_nan,
                "%0": pct_zero,
            }
        ).sort_values("%NaN+%0", ascending=False)

        display(Markdown("### 3.1) Complétude variables numériques"))
        display(completion_report)

        # Metriques de distribution et outliers (règle IQR).
        q1 = df[num_cols].quantile(0.25, numeric_only=True)
        median = df[num_cols].quantile(0.50, numeric_only=True)
        q3 = df[num_cols].quantile(0.75, numeric_only=True)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outliers_bas = (df[num_cols].lt(lower, axis=1) & df[num_cols].notna()).sum()
        outliers_haut = (df[num_cols].gt(upper, axis=1) & df[num_cols].notna()).sum()
        nb_outliers = outliers_bas + outliers_haut
        non_null_counts = df[num_cols].notna().sum().replace(0, np.nan)

        distribution_report = pd.DataFrame(
            {
                "%outliers (IQR)": (nb_outliers / non_null_counts * 100).round(2).fillna(0),
                "nb_outliers_bas": outliers_bas,
                "nb_outliers_haut": outliers_haut,
                "min": df[num_cols].min(numeric_only=True),
                "Q1": q1,
                "median": median,
                "Q3": q3,
                "max": df[num_cols].max(numeric_only=True),
                "skew": df[num_cols].skew(numeric_only=True),
                "kurtosis": df[num_cols].kurtosis(numeric_only=True),
            }
        ).sort_values("%outliers (IQR)", ascending=False)

        display(Markdown("### 3.2) Distribution générale variables numériques"))
        display(distribution_report)
    else:
        display(Markdown("_Aucune variable numérique._"))

    # 4) Variables catégorielles : densité d'information et dominante.
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns
    display(Markdown(f"## 4) Variables catégorielles ({len(cat_cols)})"))
    if len(cat_cols):
        cat_summary = pd.DataFrame(
            {
                "n_uniques": df[cat_cols].nunique(dropna=True),
                "null": df[cat_cols].isna().sum(),
                "%null": (df[cat_cols].isna().mean() * 100).round(2),
                "modalite_top": [
                    df[c].mode(dropna=True).iloc[0] if not df[c].mode(dropna=True).empty else np.nan
                    for c in cat_cols
                ],
                "nb_modalite_top": [
                    df[c].value_counts(dropna=True).iloc[0] if not df[c].value_counts(dropna=True).empty else 0
                    for c in cat_cols
                ],
            }
        ).sort_values(["%null", "n_uniques"], ascending=[False, False])
        display(cat_summary)
    else:
        display(Markdown("_Aucune variable catégorielle._"))

    # 5) Variables temporelles : bornes et complétude.
    dt_cols = df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns
    display(Markdown(f"## 5) Variables temporelles ({len(dt_cols)})"))
    if len(dt_cols):
        dt_summary = pd.DataFrame(
            {
                "min": df[dt_cols].min(),
                "max": df[dt_cols].max(),
                "null": df[dt_cols].isna().sum(),
                "%null": (df[dt_cols].isna().mean() * 100).round(2),
            }
        )
        display(dt_summary)
    else:
        display(Markdown("_Aucune variable datetime._"))
