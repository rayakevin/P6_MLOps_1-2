#uv add numpy pandas scikit-learn matplotlib seaborn category-encoders scipy joblib tqdm

# Fonctions.py
import numpy as np
import pandas as pd
from IPython.display import display


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
    print("\n## 1) Vue globale")
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
    print("\n## 2) Qualité des colonnes et cardinalité")
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
    print(f"\n## 3) Variables numériques ({len(num_cols)})")
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

        print("\n### 3.1) Complétude variables numériques")
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

        print("\n### 3.2) Distribution générale variables numériques")
        display(distribution_report)
    else:
        print("\nAucune variable numérique.")

    # 4) Variables catégorielles : densité d'information et dominante.
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns
    print(f"\n## 4) Variables catégorielles ({len(cat_cols)})")
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
        print("\nAucune variable catégorielle.")

    # 5) Variables temporelles : bornes et complétude.
    dt_cols = df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns
    print(f"\n## 5) Variables temporelles ({len(dt_cols)})")
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
        print("\nAucune variable datetime.")


def _quote_ident(identifier: str) -> str:
    """
    Quote un identifiant SQL PostgreSQL.

    Cette fonction est utile pour manipuler sans erreur des noms de colonnes
    provenant des CSV Home Credit, qui conservent leur casse d'origine
    (ex. "TARGET", "AMT_CREDIT"). En PostgreSQL, ne pas quoter ces noms
    reviendrait a les convertir en minuscules.
    """
    return '"' + identifier.replace('"', '""') + '"'


def _qualified_table_name(schema: str, table_name: str) -> str:
    """
    Construit un nom de table qualifie `schema.table` correctement quote.
    """
    return f"{_quote_ident(schema)}.{_quote_ident(table_name)}"


def _build_union_all(queries: list[str]) -> str:
    """
    Assemble une liste de SELECT en un seul bloc SQL avec UNION ALL.

    Cette approche permet de calculer un rapport colonne par colonne tout en
    recuperant le resultat dans un seul DataFrame pandas.
    """
    wrapped_queries = [
        f"SELECT * FROM (\n{query}\n) AS subquery_{idx}"
        for idx, query in enumerate(queries, start=1)
    ]
    return "\nUNION ALL\n".join(wrapped_queries)


def _get_table_columns(conn, table_name: str, schema: str) -> pd.DataFrame:
    """
    Récupère les colonnes et types d'une table PostgreSQL.
    """
    columns_query = f"""
        SELECT
            column_name,
            data_type
        FROM information_schema.columns
        WHERE table_schema = '{schema}'
          AND table_name = '{table_name}'
        ORDER BY ordinal_position
    """
    return pd.read_sql(columns_query, conn)


def _get_global_table_stats(conn, table_ref: str, column_names: list[str]) -> pd.DataFrame:
    """
    Calcule les statistiques globales de la table : lignes, colonnes, doublons.
    """
    n_cols = len(column_names)
    row_expr = ", ".join(_quote_ident(col) for col in column_names)
    global_query = f"""
        SELECT
            COUNT(*) AS n_lignes,
            {n_cols} AS n_colonnes,
            COUNT(*) - COUNT(DISTINCT ROW({row_expr})) AS doublons
        FROM {table_ref}
    """
    global_df = pd.read_sql(global_query, conn)
    global_df["%_doublons"] = np.where(
        global_df["n_lignes"] > 0,
        (global_df["doublons"] / global_df["n_lignes"] * 100).round(2),
        0,
    )
    return global_df


def eda_overview_sql(conn, table_name: str, schema: str = "public") -> None:
    """
    Affiche un aperçu EDA d'une table PostgreSQL.

    Le rapport suit la meme logique que `eda_overview`, mais les calculs sont
    effectues directement en SQL sur la base :
    1. Vue globale du dataset
    2. Qualité des colonnes (types, manquants, cardinalité)
    3. Variables numériques
       3.1 Complétude (%NaN+%0, %NaN, %0)
       3.2 Distribution générale (%outliers IQR, quartiles, skew, kurtosis)
    4. Variables catégorielles
    5. Variables temporelles

    Parameters
    ----------
    conn :
        Connexion DB-API ouverte vers PostgreSQL, compatible avec `pandas.read_sql`.
    table_name : str
        Nom de la table à analyser.
    schema : str, default="public"
        Schéma PostgreSQL contenant la table.

    Returns
    -------
    None
        La fonction affiche les tableaux avec `display` et ne retourne rien.

    Pourquoi passer par SQL
    -----------------------
    Pour les grosses tables, il est plus efficace de laisser PostgreSQL faire
    les agrégations et de ne remonter dans pandas que les résultats résumés.
    """
    table_ref = _qualified_table_name(schema, table_name)
    columns_df = _get_table_columns(conn, table_name, schema)

    if columns_df.empty:
        raise ValueError(f"La table {schema}.{table_name} est introuvable.")

    column_names = columns_df["column_name"].tolist()

    # 1) Vue globale : taille du dataset et estimation des doublons exacts.
    print("\n## 1) Vue globale")
    global_df = _get_global_table_stats(conn, table_ref, column_names)
    display(global_df)

    # 2) Rapport colonne : types, manquants, cardinalité.
    print("\n## 2) Qualité des colonnes et cardinalité")
    col_queries = []
    for _, row in columns_df.iterrows():
        col = row["column_name"]
        quoted_col = _quote_ident(col)
        col_queries.append(
            f"""
            SELECT
                '{col}' AS column_name,
                '{row["data_type"]}' AS dtype,
                COUNT(*) FILTER (WHERE {quoted_col} IS NOT NULL) AS non_null,
                COUNT(*) FILTER (WHERE {quoted_col} IS NULL) AS null,
                COUNT(DISTINCT {quoted_col}) AS n_uniques
            FROM {table_ref}
            """
        )

    col_report = pd.read_sql(_build_union_all(col_queries), conn)
    n_rows = int(global_df.loc[0, "n_lignes"])
    col_report["%null"] = np.where(
        n_rows > 0,
        (col_report["null"] / n_rows * 100).round(2),
        0,
    )
    col_report = col_report[["column_name", "dtype", "%null", "non_null", "null", "n_uniques"]]
    col_report = col_report.sort_values(["dtype", "%null"], ascending=[True, False])
    display(col_report.set_index("column_name"))

    numeric_types = {
        "smallint",
        "integer",
        "bigint",
        "numeric",
        "real",
        "double precision",
        "smallserial",
        "serial",
        "bigserial",
    }
    categorical_types = {
        "character varying",
        "character",
        "text",
        "boolean",
    }
    datetime_types = {
        "date",
        "timestamp without time zone",
        "timestamp with time zone",
        "time without time zone",
        "time with time zone",
    }

    numeric_cols = columns_df.loc[columns_df["data_type"].isin(numeric_types), "column_name"].tolist()
    cat_cols = columns_df.loc[columns_df["data_type"].isin(categorical_types), "column_name"].tolist()
    dt_cols = columns_df.loc[columns_df["data_type"].isin(datetime_types), "column_name"].tolist()

    # 3) Variables numeriques.
    print(f"\n## 3) Variables numériques ({len(numeric_cols)})")
    if numeric_cols:
        completion_queries = []
        for col in numeric_cols:
            quoted_col = _quote_ident(col)
            completion_queries.append(
                f"""
                SELECT
                    '{col}' AS column_name,
                    COUNT(*) FILTER (WHERE {quoted_col} IS NULL) AS null_count,
                    COUNT(*) FILTER (WHERE {quoted_col} = 0) AS zero_count
                FROM {table_ref}
                """
            )

        completion_report = pd.read_sql(_build_union_all(completion_queries), conn)
        completion_report["%NaN"] = np.where(
            n_rows > 0,
            (completion_report["null_count"] / n_rows * 100).round(2),
            0,
        )
        completion_report["%0"] = np.where(
            n_rows > 0,
            (completion_report["zero_count"] / n_rows * 100).round(2),
            0,
        )
        completion_report["%NaN+%0"] = (
            completion_report["%NaN"] + completion_report["%0"]
        ).round(2)
        completion_report = completion_report[
            ["column_name", "%NaN+%0", "%NaN", "%0"]
        ].sort_values("%NaN+%0", ascending=False)

        print("\n### 3.1) Complétude variables numériques")
        display(completion_report.set_index("column_name"))

        distribution_queries = []
        for col in numeric_cols:
            quoted_col = _quote_ident(col)
            distribution_queries.append(
                f"""
                WITH stats AS (
                    SELECT
                        COUNT({quoted_col}) AS non_null_count,
                        MIN({quoted_col}) AS min_value,
                        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY {quoted_col}) AS q1,
                        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY {quoted_col}) AS median,
                        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY {quoted_col}) AS q3,
                        MAX({quoted_col}) AS max_value,
                        AVG({quoted_col}) AS mean_value,
                        STDDEV_SAMP({quoted_col}) AS stddev_value
                    FROM {table_ref}
                    WHERE {quoted_col} IS NOT NULL
                ),
                outliers AS (
                    SELECT
                        COUNT(*) FILTER (WHERE t.{quoted_col} < s.q1 - 1.5 * (s.q3 - s.q1)) AS nb_outliers_bas,
                        COUNT(*) FILTER (WHERE t.{quoted_col} > s.q3 + 1.5 * (s.q3 - s.q1)) AS nb_outliers_haut
                    FROM {table_ref} t
                    CROSS JOIN stats s
                    WHERE t.{quoted_col} IS NOT NULL
                ),
                moments AS (
                    SELECT
                        AVG(
                            CASE
                                WHEN s.stddev_value IS NULL OR s.stddev_value = 0 THEN NULL
                                ELSE POWER((t.{quoted_col} - s.mean_value) / s.stddev_value, 3)
                            END
                        ) AS skew_value,
                        AVG(
                            CASE
                                WHEN s.stddev_value IS NULL OR s.stddev_value = 0 THEN NULL
                                ELSE POWER((t.{quoted_col} - s.mean_value) / s.stddev_value, 4)
                            END
                        ) - 3 AS kurtosis_value
                    FROM {table_ref} t
                    CROSS JOIN stats s
                    WHERE t.{quoted_col} IS NOT NULL
                )
                SELECT
                    '{col}' AS column_name,
                    ROUND(
                        100.0 * (o.nb_outliers_bas + o.nb_outliers_haut)
                        / NULLIF(s.non_null_count, 0), 2
                    ) AS "%outliers (IQR)",
                    o.nb_outliers_bas,
                    o.nb_outliers_haut,
                    s.min_value AS min,
                    s.q1 AS "Q1",
                    s.median AS median,
                    s.q3 AS "Q3",
                    s.max_value AS max,
                    m.skew_value AS skew,
                    m.kurtosis_value AS kurtosis
                FROM stats s
                CROSS JOIN outliers o
                CROSS JOIN moments m
                """
            )

        distribution_report = pd.read_sql(_build_union_all(distribution_queries), conn)
        distribution_report = distribution_report.sort_values("%outliers (IQR)", ascending=False)

        print("\n### 3.2) Distribution générale variables numériques")
        display(distribution_report.set_index("column_name"))
    else:
        print("\nAucune variable numérique.")

    # 4) Variables categorielles.
    print(f"\n## 4) Variables catégorielles ({len(cat_cols)})")
    if cat_cols:
        cat_queries = []
        for col in cat_cols:
            quoted_col = _quote_ident(col)
            cat_queries.append(
                f"""
                WITH top_modality AS (
                    SELECT
                        {quoted_col} AS modalite_top,
                        COUNT(*) AS nb_modalite_top
                    FROM {table_ref}
                    WHERE {quoted_col} IS NOT NULL
                    GROUP BY {quoted_col}
                    ORDER BY COUNT(*) DESC, {quoted_col}
                    LIMIT 1
                )
                SELECT
                    '{col}' AS column_name,
                    COUNT(DISTINCT {quoted_col}) AS n_uniques,
                    COUNT(*) FILTER (WHERE {quoted_col} IS NULL) AS null,
                    (SELECT modalite_top FROM top_modality) AS modalite_top,
                    COALESCE((SELECT nb_modalite_top FROM top_modality), 0) AS nb_modalite_top
                FROM {table_ref}
                """
            )

        cat_summary = pd.read_sql(_build_union_all(cat_queries), conn)
        cat_summary["%null"] = np.where(
            n_rows > 0,
            (cat_summary["null"] / n_rows * 100).round(2),
            0,
        )
        cat_summary = cat_summary[
            ["column_name", "n_uniques", "null", "%null", "modalite_top", "nb_modalite_top"]
        ].sort_values(["%null", "n_uniques"], ascending=[False, False])
        display(cat_summary.set_index("column_name"))
    else:
        print("\nAucune variable catégorielle.")

    # 5) Variables temporelles.
    print(f"\n## 5) Variables temporelles ({len(dt_cols)})")
    if dt_cols:
        dt_queries = []
        for col in dt_cols:
            quoted_col = _quote_ident(col)
            dt_queries.append(
                f"""
                SELECT
                    '{col}' AS column_name,
                    MIN({quoted_col}) AS min,
                    MAX({quoted_col}) AS max,
                    COUNT(*) FILTER (WHERE {quoted_col} IS NULL) AS null
                FROM {table_ref}
                """
            )

        dt_summary = pd.read_sql(_build_union_all(dt_queries), conn)
        dt_summary["%null"] = np.where(
            n_rows > 0,
            (dt_summary["null"] / n_rows * 100).round(2),
            0,
        )
        display(dt_summary.set_index("column_name"))
    else:
        print("\nAucune variable datetime.")


def eda_overview_sql_light(conn, table_name: str, schema: str = "public") -> None:
    """
    Affiche une version légère de l'EDA d'une table PostgreSQL.

    Cette version vise l'itération rapide. Elle conserve les éléments les plus
    utiles pour une première exploration, mais évite les calculs coûteux comme :
    - quartiles,
    - détection d'outliers IQR,
    - skew,
    - kurtosis.

    Parameters
    ----------
    conn :
        Connexion DB-API ouverte vers PostgreSQL.
    table_name : str
        Nom de la table à analyser.
    schema : str, default="public"
        Schéma PostgreSQL contenant la table.
    """
    table_ref = _qualified_table_name(schema, table_name)
    columns_df = _get_table_columns(conn, table_name, schema)

    if columns_df.empty:
        raise ValueError(f"La table {schema}.{table_name} est introuvable.")

    column_names = columns_df["column_name"].tolist()
    global_df = _get_global_table_stats(conn, table_ref, column_names)
    n_rows = int(global_df.loc[0, "n_lignes"])

    numeric_types = {
        "smallint",
        "integer",
        "bigint",
        "numeric",
        "real",
        "double precision",
        "smallserial",
        "serial",
        "bigserial",
    }
    categorical_types = {
        "character varying",
        "character",
        "text",
        "boolean",
    }
    datetime_types = {
        "date",
        "timestamp without time zone",
        "timestamp with time zone",
        "time without time zone",
        "time with time zone",
    }

    numeric_cols = columns_df.loc[columns_df["data_type"].isin(numeric_types), "column_name"].tolist()
    cat_cols = columns_df.loc[columns_df["data_type"].isin(categorical_types), "column_name"].tolist()
    dt_cols = columns_df.loc[columns_df["data_type"].isin(datetime_types), "column_name"].tolist()

    print("\n## 1) Vue globale")
    display(global_df)

    print("\n## 2) Qualité des colonnes et cardinalité")
    col_queries = []
    for _, row in columns_df.iterrows():
        col = row["column_name"]
        quoted_col = _quote_ident(col)
        col_queries.append(
            f"""
            SELECT
                '{col}' AS column_name,
                '{row["data_type"]}' AS dtype,
                COUNT(*) FILTER (WHERE {quoted_col} IS NOT NULL) AS non_null,
                COUNT(*) FILTER (WHERE {quoted_col} IS NULL) AS null,
                COUNT(DISTINCT {quoted_col}) AS n_uniques
            FROM {table_ref}
            """
        )

    col_report = pd.read_sql(_build_union_all(col_queries), conn)
    col_report["%null"] = np.where(
        n_rows > 0,
        (col_report["null"] / n_rows * 100).round(2),
        0,
    )
    col_report = col_report[["column_name", "dtype", "%null", "non_null", "null", "n_uniques"]]
    col_report = col_report.sort_values(["dtype", "%null"], ascending=[True, False])
    display(col_report.set_index("column_name"))

    print(f"\n## 3) Variables numériques ({len(numeric_cols)})")
    if numeric_cols:
        numeric_queries = []
        for col in numeric_cols:
            quoted_col = _quote_ident(col)
            numeric_queries.append(
                f"""
                SELECT
                    '{col}' AS column_name,
                    COUNT(*) FILTER (WHERE {quoted_col} IS NULL) AS null_count,
                    COUNT(*) FILTER (WHERE {quoted_col} = 0) AS zero_count,
                    MIN({quoted_col}) AS min,
                    AVG({quoted_col}) AS mean,
                    MAX({quoted_col}) AS max
                FROM {table_ref}
                """
            )

        numeric_report = pd.read_sql(_build_union_all(numeric_queries), conn)
        numeric_report["%NaN"] = np.where(
            n_rows > 0,
            (numeric_report["null_count"] / n_rows * 100).round(2),
            0,
        )
        numeric_report["%0"] = np.where(
            n_rows > 0,
            (numeric_report["zero_count"] / n_rows * 100).round(2),
            0,
        )
        numeric_report["%NaN+%0"] = (
            numeric_report["%NaN"] + numeric_report["%0"]
        ).round(2)
        numeric_report = numeric_report[
            ["column_name", "%NaN+%0", "%NaN", "%0", "min", "mean", "max"]
        ].sort_values("%NaN+%0", ascending=False)
        display(numeric_report.set_index("column_name"))
    else:
        print("\nAucune variable numérique.")

    print(f"\n## 4) Variables catégorielles ({len(cat_cols)})")
    if cat_cols:
        cat_queries = []
        for col in cat_cols:
            quoted_col = _quote_ident(col)
            cat_queries.append(
                f"""
                WITH top_modality AS (
                    SELECT
                        {quoted_col} AS modalite_top,
                        COUNT(*) AS nb_modalite_top
                    FROM {table_ref}
                    WHERE {quoted_col} IS NOT NULL
                    GROUP BY {quoted_col}
                    ORDER BY COUNT(*) DESC, {quoted_col}
                    LIMIT 1
                )
                SELECT
                    '{col}' AS column_name,
                    COUNT(DISTINCT {quoted_col}) AS n_uniques,
                    COUNT(*) FILTER (WHERE {quoted_col} IS NULL) AS null,
                    (SELECT modalite_top FROM top_modality) AS modalite_top,
                    COALESCE((SELECT nb_modalite_top FROM top_modality), 0) AS nb_modalite_top
                FROM {table_ref}
                """
            )

        cat_summary = pd.read_sql(_build_union_all(cat_queries), conn)
        cat_summary["%null"] = np.where(
            n_rows > 0,
            (cat_summary["null"] / n_rows * 100).round(2),
            0,
        )
        cat_summary = cat_summary[
            ["column_name", "n_uniques", "null", "%null", "modalite_top", "nb_modalite_top"]
        ].sort_values(["%null", "n_uniques"], ascending=[False, False])
        display(cat_summary.set_index("column_name"))
    else:
        print("\nAucune variable catégorielle.")

    print(f"\n## 5) Variables temporelles ({len(dt_cols)})")
    if dt_cols:
        dt_queries = []
        for col in dt_cols:
            quoted_col = _quote_ident(col)
            dt_queries.append(
                f"""
                SELECT
                    '{col}' AS column_name,
                    MIN({quoted_col}) AS min,
                    MAX({quoted_col}) AS max,
                    COUNT(*) FILTER (WHERE {quoted_col} IS NULL) AS null
                FROM {table_ref}
                """
            )

        dt_summary = pd.read_sql(_build_union_all(dt_queries), conn)
        dt_summary["%null"] = np.where(
            n_rows > 0,
            (dt_summary["null"] / n_rows * 100).round(2),
            0,
        )
        display(dt_summary.set_index("column_name"))
    else:
        print("\nAucune variable datetime.")
