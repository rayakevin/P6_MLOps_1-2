#uv add numpy pandas scikit-learn matplotlib seaborn category-encoders scipy joblib tqdm

# Fonctions.py
from pathlib import Path
import gc

import numpy as np
import pandas as pd
from IPython.display import display
from scipy.stats import chi2_contingency


RAW_TABLES = {
    "application_train": "application_train.csv",
    "application_test": "application_test.csv",
    "bureau": "bureau.csv",
    "bureau_balance": "bureau_balance.csv",
    "previous_application": "previous_application.csv",
    "pos_cash_balance": "POS_CASH_balance.csv",
    "installments_payments": "installments_payments.csv",
    "credit_card_balance": "credit_card_balance.csv",
}


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

# ---------------------------------------------------------------------------
# Helpers génériques de qualité de données / encodage
# ---------------------------------------------------------------------------

def safe_divide(numerator, denominator):
    """
    Réalise une division sûre en renvoyant `NaN` si le dénominateur vaut 0.
    """
    numerator_arr = np.asarray(numerator, dtype=float)
    denominator_arr = np.asarray(denominator, dtype=float)
    result = np.full_like(numerator_arr, np.nan, dtype=float)
    valid_mask = (~np.isnan(denominator_arr)) & (denominator_arr != 0)
    result[valid_mask] = numerator_arr[valid_mask] / denominator_arr[valid_mask]
    return result


def one_hot_encode_dataframe(
    df: pd.DataFrame,
    nan_as_category: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Encode toutes les colonnes catégorielles d'un DataFrame en one-hot.

    La fonction renvoie le DataFrame encodé ainsi que la liste des colonnes
    nouvellement créées.
    """
    original_columns = list(df.columns)
    categorical_columns = [
        column
        for column in df.columns
        if pd.api.types.is_object_dtype(df[column]) or pd.api.types.is_categorical_dtype(df[column])
    ]

    if not categorical_columns:
        return df.copy(), []

    encoded_df = pd.get_dummies(
        df,
        columns=categorical_columns,
        dummy_na=nan_as_category,
        dtype=np.uint8,
    )
    new_columns = [column for column in encoded_df.columns if column not in original_columns]
    return encoded_df, new_columns


def drop_low_information_columns(
    df: pd.DataFrame,
    protected_columns: list[str] | None = None,
    missing_ratio_threshold: float = 0.995,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Supprime les colonnes objectivement inutiles avant modélisation.

    La fonction applique volontairement des règles prudentes :
    - colonne entièrement vide ;
    - colonne quasi entièrement vide ;
    - colonne constante une fois les valeurs manquantes ignorées.

    Parameters
    ----------
    df : pd.DataFrame
        Tableau à filtrer.
    protected_columns : list[str] | None
        Colonnes à ne jamais supprimer, même si elles respectent un critère.
    missing_ratio_threshold : float
        Seuil au-delà duquel une colonne est considérée comme trop vide.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Le DataFrame filtré et un rapport des colonnes supprimées.
    """
    protected = set(protected_columns or [])

    rows: list[dict[str, float | str]] = []
    columns_to_drop: list[str] = []

    for column in df.columns:
        if column in protected:
            continue

        series = df[column]
        missing_ratio = float(series.isna().mean())
        nunique = int(series.nunique(dropna=True))

        if series.isna().all():
            rows.append(
                {
                    "feature": column,
                    "reason": "all_missing",
                    "missing_ratio": missing_ratio,
                    "nunique": nunique,
                }
            )
            columns_to_drop.append(column)
        elif missing_ratio >= missing_ratio_threshold:
            rows.append(
                {
                    "feature": column,
                    "reason": "quasi_all_missing",
                    "missing_ratio": missing_ratio,
                    "nunique": nunique,
                }
            )
            columns_to_drop.append(column)
        elif nunique <= 1:
            rows.append(
                {
                    "feature": column,
                    "reason": "constant",
                    "missing_ratio": missing_ratio,
                    "nunique": nunique,
                }
            )
            columns_to_drop.append(column)

    filtered_df = df.drop(columns=columns_to_drop, errors="ignore")
    report = pd.DataFrame(rows)
    if not report.empty:
        report = report.sort_values(["reason", "missing_ratio", "feature"]).reset_index(drop=True)
    return filtered_df, report


# Alias plus court conservé pour compatibilité.
one_hot_encoder = one_hot_encode_dataframe


def missing_values_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construit un tableau synthétique des valeurs manquantes.
    """
    report = pd.DataFrame(
        {
            "missing_count": df.isna().sum(),
            "missing_ratio": df.isna().mean(),
            "dtype": df.dtypes.astype(str),
            "nunique": df.nunique(dropna=True),
        }
    )
    return report.loc[report["missing_count"] > 0].sort_values(
        ["missing_ratio", "missing_count"],
        ascending=False,
    )


def duplicate_report(df: pd.DataFrame, key: str) -> dict[str, int | float]:
    """
    Résume le niveau de duplication observé sur une clé métier.
    """
    duplicated_count = int(df.duplicated(subset=[key]).sum())
    total_rows = int(len(df))
    return {
        "rows": total_rows,
        "unique_keys": int(df[key].nunique(dropna=False)),
        "duplicated_keys": duplicated_count,
        "duplicated_ratio": round(duplicated_count / total_rows, 4) if total_rows else 0.0,
    }


def cramers_v_corrected(x: pd.Series, y: pd.Series) -> float:
    """
    Calcule un V de Cramer corrigé entre une variable catégorielle et une cible.
    """
    confusion = pd.crosstab(x.fillna("__MISSING__"), y)
    if confusion.shape[0] < 2 or confusion.shape[1] < 2:
        return np.nan

    chi2 = chi2_contingency(confusion)[0]
    n = confusion.to_numpy().sum()
    if n <= 1:
        return np.nan

    phi2 = chi2 / n
    r, k = confusion.shape
    phi2corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    denom = min(kcorr - 1, rcorr - 1)
    if denom <= 0:
        return np.nan
    return float(np.sqrt(phi2corr / denom))


def summarize_categorical_association(
    df: pd.DataFrame,
    columns: list[str],
    target_col: str,
) -> pd.DataFrame:
    """
    Résume l'association entre plusieurs variables catégorielles et une cible.

    Pour chaque variable, la fonction construit une table de contingence avec la
    cible, calcule le test du chi-2 et le V de Cramer corrigé, puis renvoie un
    tableau trié des variables les plus associées à la cible.
    """
    rows = []

    for col in columns:
        tmp = df[[col, target_col]].copy()
        tmp[col] = tmp[col].astype("string").fillna("__MISSING__")
        contingency = pd.crosstab(tmp[col], tmp[target_col])

        if contingency.shape[0] < 2 or contingency.shape[1] < 2:
            continue

        chi2, p_value, _, _ = chi2_contingency(contingency)
        rows.append(
            {
                "variable": col,
                "chi2": float(chi2),
                "p_value": float(p_value),
                "nb_modalites": int(contingency.shape[0]),
                "cramers_v": cramers_v_corrected(tmp[col], tmp[target_col]),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=["variable", "chi2", "p_value", "nb_modalites", "cramers_v"]
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["cramers_v", "chi2"], ascending=[False, False])
        .reset_index(drop=True)
    )


def summarize_binary_flags(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Compare le taux de cible des variables binaires au taux global.
    """
    global_rate = df[target_col].mean()
    flag_cols = [
        col
        for col in df.columns
        if col != target_col and df[col].dropna().nunique() == 2
    ]

    rows = []
    for col in flag_cols:
        tmp = df[[col, target_col]].copy()
        tmp[col] = tmp[col].astype("string").fillna("__MISSING__")

        stats = (
            tmp.groupby(col, dropna=False)[target_col]
            .agg(nb_obs="size", nb_target_1="sum", taux_target_1="mean")
            .reset_index()
            .rename(columns={col: "modalite"})
        )
        stats["part_obs"] = stats["nb_obs"] / len(tmp)
        stats["ecart_vs_taux_global"] = stats["taux_target_1"] - global_rate
        stats["ratio_vs_taux_global"] = np.where(
            global_rate > 0,
            stats["taux_target_1"] / global_rate,
            np.nan,
        )
        stats["variable"] = col
        rows.append(stats)

    if not rows:
        return pd.DataFrame()

    result = pd.concat(rows, ignore_index=True)
    result["ecart_abs"] = result["ecart_vs_taux_global"].abs()
    return result[
        [
            "variable",
            "modalite",
            "nb_obs",
            "part_obs",
            "nb_target_1",
            "taux_target_1",
            "ecart_vs_taux_global",
            "ratio_vs_taux_global",
            "ecart_abs",
        ]
    ].sort_values(["ecart_abs", "nb_obs"], ascending=[False, False])


def summarize_categorical_modalities(
    df: pd.DataFrame,
    columns: list[str],
    target_col: str,
) -> pd.DataFrame:
    """
    Résume les modalités catégorielles les plus associées à la cible.
    """
    global_rate = df[target_col].mean()
    rows = []

    for col in columns:
        tmp = df[[col, target_col]].copy()
        tmp[col] = tmp[col].astype("string").fillna("__MISSING__")
        stats = (
            tmp.groupby(col, dropna=False)[target_col]
            .agg(nb_obs="size", nb_target_1="sum", taux_target_1="mean")
            .reset_index()
            .rename(columns={col: "modalite"})
        )
        stats["part_obs"] = stats["nb_obs"] / len(tmp)
        stats["ecart_vs_taux_global"] = stats["taux_target_1"] - global_rate
        stats["ratio_vs_taux_global"] = np.where(
            global_rate > 0,
            stats["taux_target_1"] / global_rate,
            np.nan,
        )
        stats["variable"] = col
        rows.append(stats)

    if not rows:
        return pd.DataFrame()

    result = pd.concat(rows, ignore_index=True)
    result["ecart_abs"] = result["ecart_vs_taux_global"].abs()
    return result[
        [
            "variable",
            "modalite",
            "nb_obs",
            "part_obs",
            "nb_target_1",
            "taux_target_1",
            "ecart_vs_taux_global",
            "ratio_vs_taux_global",
            "ecart_abs",
        ]
    ].sort_values(["ecart_abs", "nb_obs"], ascending=[False, False])


# ---------------------------------------------------------------------------
# Helpers projet : agrégation multi-tables Home Credit
# ---------------------------------------------------------------------------

def _read_csv_from_directory(
    data_dir: Path | str,
    file_name: str,
    num_rows: int | None = None,
) -> pd.DataFrame:
    """
    Charge un fichier CSV situé dans un répertoire de données.

    Parameters
    ----------
    data_dir : Path | str
        Répertoire contenant les tables brutes.
    file_name : str
        Nom du fichier CSV à lire.
    num_rows : int | None, optional
        Nombre maximum de lignes à charger. Si None, charge tout le fichier.

    Returns
    -------
    pd.DataFrame
        DataFrame contenant les données lues depuis le CSV.

    Notes
    -----
    Cette fonction sert de petit helper pour centraliser la logique de lecture
    des tables brutes du projet.
    """
    data_path = Path(data_dir) / file_name
    return pd.read_csv(data_path, nrows=num_rows)


def application_train_test(
    data_dir: Path | str,
    num_rows: int | None = None,
    nan_as_category: bool = False,
) -> pd.DataFrame:
    """
    Prépare et fusionne les tables `application_train` et `application_test`.

    Cette fonction :
    - charge les deux tables principales,
    - applique un prétraitement identique aux deux jeux,
    - aligne les colonnes après encodage,
    - concatène train et test dans un unique DataFrame,
    - supprime les colonnes jugées peu informatives.

    Parameters
    ----------
    data_dir : Path | str
        Répertoire contenant les fichiers bruts.
    num_rows : int | None, optional
        Nombre maximum de lignes à charger par table.
    nan_as_category : bool, optional
        Si True, les valeurs manquantes sont traitées comme une catégorie
        lors du one-hot encoding.

    Returns
    -------
    pd.DataFrame
        Table principale combinée train + test.
    """
    # Lecture des tables principales
    train_df = _read_csv_from_directory(data_dir, RAW_TABLES["application_train"], num_rows=num_rows)
    test_df = _read_csv_from_directory(data_dir, RAW_TABLES["application_test"], num_rows=num_rows)

    def preprocess_application_table(df: pd.DataFrame) -> pd.DataFrame:
        """
        Applique le prétraitement métier à une table application.

        Étapes principales :
        - suppression des lignes avec genre anormal ("XNA"),
        - mapping binaire de certaines colonnes catégorielles,
        - gestion de l'anomalie DAYS_EMPLOYED = 365243,
        - création de statistiques sur les colonnes EXT_SOURCE,
        - création de ratios métier,
        - encodage one-hot des variables catégorielles.
        """
        # On exclut la modalité anormale CODE_GENDER = XNA
        output = df.loc[df["CODE_GENDER"] != "XNA"].copy()

        # Transformation de variables binaires textuelles en 0/1
        binary_mappings = {
            "CODE_GENDER": {"F": 0, "M": 1},
            "FLAG_OWN_CAR": {"N": 0, "Y": 1},
            "FLAG_OWN_REALTY": {"N": 0, "Y": 1},
        }
        for column, mapping in binary_mappings.items():
            output[column] = output[column].map(mapping)

        # DAYS_EMPLOYED = 365243 est une valeur sentinelle/anormale dans Home Credit
        # On crée un indicateur de cette anomalie, puis on remplace la valeur par NaN
        output["DAYS_EMPLOYED_ANOM"] = (output["DAYS_EMPLOYED"] == 365243).astype(int)
        output["DAYS_EMPLOYED"] = output["DAYS_EMPLOYED"].replace(365243, np.nan)

        # Construction de features synthétiques à partir des sources externes
        ext_sources = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
        output["EXT_SOURCES_MEAN"] = output[ext_sources].mean(axis=1)
        output["EXT_SOURCES_STD"] = output[ext_sources].std(axis=1)

        # Si l'écart-type n'est pas calculable (ex. trop de NaN), on remplace par la médiane
        output["EXT_SOURCES_STD"] = output["EXT_SOURCES_STD"].fillna(output["EXT_SOURCES_STD"].median())

        # Création de ratios métier
        # safe_divide évite les divisions par zéro / valeurs invalides
        output["CREDIT_TO_ANNUITY_RATIO"] = safe_divide(output["AMT_CREDIT"], output["AMT_ANNUITY"])
        output["CREDIT_TO_GOODS_RATIO"] = safe_divide(output["AMT_CREDIT"], output["AMT_GOODS_PRICE"])
        output["ANNUITY_TO_INCOME_RATIO"] = safe_divide(output["AMT_ANNUITY"], output["AMT_INCOME_TOTAL"])
        output["CREDIT_TO_INCOME_RATIO"] = safe_divide(output["AMT_CREDIT"], output["AMT_INCOME_TOTAL"])
        output["INCOME_PER_PERSON"] = safe_divide(output["AMT_INCOME_TOTAL"], output["CNT_FAM_MEMBERS"])
        output["DAYS_EMPLOYED_PERC"] = safe_divide(output["DAYS_EMPLOYED"], output["DAYS_BIRTH"])
        output["PAYMENT_RATE"] = safe_divide(output["AMT_ANNUITY"], output["AMT_CREDIT"])
        output["PHONE_CHANGE_TO_BIRTH_RATIO"] = safe_divide(output["DAYS_LAST_PHONE_CHANGE"], output["DAYS_BIRTH"])
        output["ID_PUBLISH_TO_BIRTH_RATIO"] = safe_divide(output["DAYS_ID_PUBLISH"], output["DAYS_BIRTH"])

        # One-hot encoding des colonnes catégorielles restantes
        output, _ = one_hot_encode_dataframe(output, nan_as_category=nan_as_category)
        return output

    # Prétraitement identique sur train et test
    train_df = preprocess_application_table(train_df)
    test_df = preprocess_application_table(test_df)

    # Alignement des colonnes pour garantir la même structure train/test
    train_df, test_df = train_df.align(test_df, join="outer", axis=1)

    # Concaténation verticale des deux jeux
    combined = pd.concat([train_df, test_df], axis=0, ignore_index=True, sort=False)

    # Suppression des colonnes peu informatives, sauf colonnes protégées
    combined, _ = drop_low_information_columns(
        combined,
        protected_columns=["SK_ID_CURR", "TARGET"],
    )
    return combined


def bureau_and_balance(
    data_dir: Path | str,
    num_rows: int | None = None,
    nan_as_category: bool = True,
) -> pd.DataFrame:
    """
    Agrège les tables `bureau` et `bureau_balance` à la maille client (`SK_ID_CURR`).

    Logique :
    1. encoder les variables catégorielles,
    2. agréger `bureau_balance` au niveau `SK_ID_BUREAU`,
    3. rattacher cette agrégation à `bureau`,
    4. agréger ensuite `bureau` au niveau client,
    5. produire aussi des sous-agrégations sur crédits actifs et clôturés.

    Returns
    -------
    pd.DataFrame
        Features agrégées par client issues de l'historique bureau.
    """
    bureau = _read_csv_from_directory(data_dir, RAW_TABLES["bureau"], num_rows=num_rows)
    bureau_balance = _read_csv_from_directory(data_dir, RAW_TABLES["bureau_balance"], num_rows=num_rows)

    # One-hot encoding des catégories
    bureau_balance, bureau_balance_cat = one_hot_encode_dataframe(
        bureau_balance,
        nan_as_category=nan_as_category,
    )
    bureau, bureau_cat = one_hot_encode_dataframe(bureau, nan_as_category=nan_as_category)

    # Agrégation de bureau_balance à la maille du crédit bureau
    bureau_balance_agg = {"MONTHS_BALANCE": ["min", "max", "size"]}
    for column in bureau_balance_cat:
        bureau_balance_agg[column] = ["mean"]

    bureau_balance_grouped = bureau_balance.groupby("SK_ID_BUREAU").agg(bureau_balance_agg)
    bureau_balance_grouped.columns = pd.Index(
        [f"{column}_{agg.upper()}" for column, agg in bureau_balance_grouped.columns.tolist()]
    )

    # Jointure du résumé bureau_balance vers bureau
    bureau = bureau.join(bureau_balance_grouped, how="left", on="SK_ID_BUREAU")

    # Après jointure, SK_ID_BUREAU ne sert plus pour l'agrégation client finale
    bureau = bureau.drop(columns=["SK_ID_BUREAU"])

    # Agrégations numériques sur les crédits bureau
    num_aggregations = {
        "DAYS_CREDIT": ["min", "max", "mean", "var"],
        "DAYS_CREDIT_ENDDATE": ["min", "max", "mean"],
        "DAYS_CREDIT_UPDATE": ["mean"],
        "CREDIT_DAY_OVERDUE": ["max", "mean"],
        "AMT_CREDIT_MAX_OVERDUE": ["mean"],
        "AMT_CREDIT_SUM": ["max", "mean", "sum"],
        "AMT_CREDIT_SUM_DEBT": ["max", "mean", "sum"],
        "AMT_CREDIT_SUM_OVERDUE": ["mean"],
        "AMT_CREDIT_SUM_LIMIT": ["mean", "sum"],
        "AMT_ANNUITY": ["max", "mean"],
        "CNT_CREDIT_PROLONG": ["sum"],
        "MONTHS_BALANCE_MIN": ["min"],
        "MONTHS_BALANCE_MAX": ["max"],
        "MONTHS_BALANCE_SIZE": ["mean", "sum"],
    }

    # Pour les variables catégorielles one-hot encodées, la moyenne = fréquence de la modalité
    cat_aggregations = {column: ["mean"] for column in bureau_cat}
    for column in bureau_balance_cat:
        cat_aggregations[f"{column}_MEAN"] = ["mean"]

    # Agrégation finale au niveau client
    bureau_agg = bureau.groupby("SK_ID_CURR").agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(
        [f"BURO_{column}_{agg.upper()}" for column, agg in bureau_agg.columns.tolist()]
    )

    # Nombre total de lignes bureau par client
    bureau_agg = bureau_agg.join(bureau.groupby("SK_ID_CURR").size().rename("BURO_COUNT"))

    # Sous-ensemble des crédits actifs
    active = bureau.loc[bureau.get("CREDIT_ACTIVE_Active", 0) == 1]
    if not active.empty:
        active_agg = active.groupby("SK_ID_CURR").agg(num_aggregations)
        active_agg.columns = pd.Index(
            [f"ACTIVE_{column}_{agg.upper()}" for column, agg in active_agg.columns.tolist()]
        )
        bureau_agg = bureau_agg.join(active_agg, how="left", on="SK_ID_CURR")
        bureau_agg = bureau_agg.join(active.groupby("SK_ID_CURR").size().rename("ACTIVE_COUNT"))

    # Sous-ensemble des crédits clôturés
    closed = bureau.loc[bureau.get("CREDIT_ACTIVE_Closed", 0) == 1]
    if not closed.empty:
        closed_agg = closed.groupby("SK_ID_CURR").agg(num_aggregations)
        closed_agg.columns = pd.Index(
            [f"CLOSED_{column}_{agg.upper()}" for column, agg in closed_agg.columns.tolist()]
        )
        bureau_agg = bureau_agg.join(closed_agg, how="left", on="SK_ID_CURR")
        bureau_agg = bureau_agg.join(closed.groupby("SK_ID_CURR").size().rename("CLOSED_COUNT"))

    # Libération mémoire
    del bureau, bureau_balance, bureau_balance_grouped, active, closed
    gc.collect()
    return bureau_agg


def previous_applications(
    data_dir: Path | str,
    num_rows: int | None = None,
    nan_as_category: bool = True,
) -> pd.DataFrame:
    """
    Agrège la table `previous_application` à la maille client.

    Cette fonction résume l'historique des demandes précédentes :
    - montants,
    - décisions,
    - caractéristiques temporelles,
    - statut approuvé/refusé.

    Returns
    -------
    pd.DataFrame
        Table agrégée au niveau `SK_ID_CURR`.
    """
    prev = _read_csv_from_directory(data_dir, RAW_TABLES["previous_application"], num_rows=num_rows)
    prev, cat_cols = one_hot_encode_dataframe(prev, nan_as_category=nan_as_category)

    # Remplacement des dates sentinelles par NaN
    for column in [
        "DAYS_FIRST_DRAWING",
        "DAYS_FIRST_DUE",
        "DAYS_LAST_DUE_1ST_VERSION",
        "DAYS_LAST_DUE",
        "DAYS_TERMINATION",
    ]:
        prev[column] = prev[column].replace(365243, np.nan)

    # Ratio entre montant demandé et montant accordé
    prev["APP_CREDIT_PERC"] = safe_divide(prev["AMT_APPLICATION"], prev["AMT_CREDIT"])

    num_aggregations = {
        "SK_ID_PREV": ["nunique"],
        "AMT_ANNUITY": ["min", "max", "mean"],
        "AMT_APPLICATION": ["min", "max", "mean"],
        "AMT_CREDIT": ["min", "max", "mean"],
        "APP_CREDIT_PERC": ["min", "max", "mean", "var"],
        "AMT_DOWN_PAYMENT": ["min", "max", "mean"],
        "AMT_GOODS_PRICE": ["min", "max", "mean"],
        "HOUR_APPR_PROCESS_START": ["min", "max", "mean"],
        "RATE_DOWN_PAYMENT": ["min", "max", "mean"],
        "DAYS_DECISION": ["min", "max", "mean"],
        "CNT_PAYMENT": ["mean", "sum"],
    }
    cat_aggregations = {column: ["mean"] for column in cat_cols}

    prev_agg = prev.groupby("SK_ID_CURR").agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(
        [f"PREV_{column}_{agg.upper()}" for column, agg in prev_agg.columns.tolist()]
    )
    prev_agg = prev_agg.join(prev.groupby("SK_ID_CURR").size().rename("PREV_COUNT"))

    # Historique approuvé
    approved = prev.loc[prev.get("NAME_CONTRACT_STATUS_Approved", 0) == 1]
    if not approved.empty:
        approved_agg = approved.groupby("SK_ID_CURR").agg(num_aggregations)
        approved_agg.columns = pd.Index(
            [f"APPROVED_{column}_{agg.upper()}" for column, agg in approved_agg.columns.tolist()]
        )
        prev_agg = prev_agg.join(approved_agg, how="left", on="SK_ID_CURR")
        prev_agg = prev_agg.join(approved.groupby("SK_ID_CURR").size().rename("APPROVED_COUNT"))

    # Historique refusé
    refused = prev.loc[prev.get("NAME_CONTRACT_STATUS_Refused", 0) == 1]
    if not refused.empty:
        refused_agg = refused.groupby("SK_ID_CURR").agg(num_aggregations)
        refused_agg.columns = pd.Index(
            [f"REFUSED_{column}_{agg.upper()}" for column, agg in refused_agg.columns.tolist()]
        )
        prev_agg = prev_agg.join(refused_agg, how="left", on="SK_ID_CURR")
        prev_agg = prev_agg.join(refused.groupby("SK_ID_CURR").size().rename("REFUSED_COUNT"))

    del prev, approved, refused
    gc.collect()
    return prev_agg


def pos_cash(
    data_dir: Path | str,
    num_rows: int | None = None,
    nan_as_category: bool = True,
) -> pd.DataFrame:
    """
    Agrège la table `POS_CASH_balance` à la maille client.

    Cette table contient des informations de suivi sur les crédits POS / cash
    et permet d'extraire notamment des indicateurs de retard.

    Returns
    -------
    pd.DataFrame
        Features agrégées par client.
    """
    pos = _read_csv_from_directory(data_dir, RAW_TABLES["pos_cash_balance"], num_rows=num_rows)
    pos, cat_cols = one_hot_encode_dataframe(pos, nan_as_category=nan_as_category)

    aggregations = {
        "MONTHS_BALANCE": ["max", "mean", "size"],
        "SK_DPD": ["max", "mean"],
        "SK_DPD_DEF": ["max", "mean"],
    }
    for column in cat_cols:
        aggregations[column] = ["mean"]

    pos_agg = pos.groupby("SK_ID_CURR").agg(aggregations)
    pos_agg.columns = pd.Index([f"POS_{column}_{agg.upper()}" for column, agg in pos_agg.columns.tolist()])
    pos_agg = pos_agg.join(pos.groupby("SK_ID_CURR").size().rename("POS_COUNT"))

    del pos
    gc.collect()
    return pos_agg


def installments_payments(
    data_dir: Path | str,
    num_rows: int | None = None,
    nan_as_category: bool = True,
) -> pd.DataFrame:
    """
    Agrège la table `installments_payments` à la maille client.

    Cette fonction calcule plusieurs indicateurs importants liés au paiement
    des échéances :
    - proportion payée,
    - écart entre dû et payé,
    - retard de paiement (DPD),
    - paiement en avance (DBD).

    Returns
    -------
    pd.DataFrame
        Table agrégée par client.
    """
    installments = _read_csv_from_directory(data_dir, RAW_TABLES["installments_payments"], num_rows=num_rows)
    installments, cat_cols = one_hot_encode_dataframe(installments, nan_as_category=nan_as_category)

    # Pourcentage payé par rapport à la mensualité attendue
    installments["PAYMENT_PERC"] = safe_divide(
        installments["AMT_PAYMENT"],
        installments["AMT_INSTALMENT"],
    )

    # Différence brute entre dû et payé
    installments["PAYMENT_DIFF"] = installments["AMT_INSTALMENT"] - installments["AMT_PAYMENT"]

    # Days Past Due : retard de paiement, borné à 0 si paiement à l'heure ou en avance
    installments["DPD"] = (installments["DAYS_ENTRY_PAYMENT"] - installments["DAYS_INSTALMENT"]).clip(lower=0)

    # Days Before Due : paiement en avance, borné à 0 si paiement en retard
    installments["DBD"] = (installments["DAYS_INSTALMENT"] - installments["DAYS_ENTRY_PAYMENT"]).clip(lower=0)

    aggregations = {
        "NUM_INSTALMENT_VERSION": ["nunique"],
        "DPD": ["max", "mean", "sum"],
        "DBD": ["max", "mean", "sum"],
        "PAYMENT_PERC": ["max", "mean", "sum", "var"],
        "PAYMENT_DIFF": ["max", "mean", "sum", "var"],
        "AMT_INSTALMENT": ["max", "mean", "sum"],
        "AMT_PAYMENT": ["min", "max", "mean", "sum"],
        "DAYS_ENTRY_PAYMENT": ["max", "mean", "sum"],
    }
    for column in cat_cols:
        aggregations[column] = ["mean"]

    installments_agg = installments.groupby("SK_ID_CURR").agg(aggregations)
    installments_agg.columns = pd.Index(
        [f"INSTAL_{column}_{agg.upper()}" for column, agg in installments_agg.columns.tolist()]
    )
    installments_agg = installments_agg.join(
        installments.groupby("SK_ID_CURR").size().rename("INSTAL_COUNT")
    )

    del installments
    gc.collect()
    return installments_agg


def credit_card_balance(
    data_dir: Path | str,
    num_rows: int | None = None,
    nan_as_category: bool = True,
) -> pd.DataFrame:
    """
    Agrège la table `credit_card_balance` à la maille client.

    Toutes les colonnes restantes (hors identifiant de demande précédente)
    sont agrégées par client avec plusieurs statistiques descriptives.

    Returns
    -------
    pd.DataFrame
        Table agrégée par client.
    """
    credit_card = _read_csv_from_directory(data_dir, RAW_TABLES["credit_card_balance"], num_rows=num_rows)
    credit_card, _ = one_hot_encode_dataframe(credit_card, nan_as_category=nan_as_category)

    # On supprime l'identifiant de crédit précédent car on veut agréger au niveau client
    credit_card = credit_card.drop(columns=["SK_ID_PREV"])

    credit_card_agg = credit_card.groupby("SK_ID_CURR").agg(["min", "max", "mean", "sum", "var"])
    credit_card_agg.columns = pd.Index(
        [f"CC_{column}_{agg.upper()}" for column, agg in credit_card_agg.columns.tolist()]
    )
    credit_card_agg = credit_card_agg.join(credit_card.groupby("SK_ID_CURR").size().rename("CC_COUNT"))

    del credit_card
    gc.collect()
    return credit_card_agg


def add_post_join_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute des features dérivées après assemblage des différentes tables.

    Ces ratios exploitent simultanément :
    - des variables de la table principale,
    - et des agrégations issues des tables historiques.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame final après jointure des tables enrichies.

    Returns
    -------
    pd.DataFrame
        Copie enrichie avec de nouveaux ratios.
    """
    output = df.copy()

    # Liste de features à créer sous la forme :
    # (nom_nouvelle_feature, numerateur, denominateur)
    feature_pairs = [
        ("NEW_BUREAU_DEBT_RATIO", "BURO_AMT_CREDIT_SUM_DEBT_SUM", "BURO_AMT_CREDIT_SUM_SUM"),
        ("NEW_ACTIVE_DEBT_RATIO", "ACTIVE_AMT_CREDIT_SUM_DEBT_SUM", "ACTIVE_AMT_CREDIT_SUM_SUM"),
        ("NEW_APPROVAL_RATE", "APPROVED_COUNT", "PREV_COUNT"),
        ("NEW_REFUSAL_RATE", "REFUSED_COUNT", "PREV_COUNT"),
        ("NEW_LATE_PAYMENT_RATIO", "INSTAL_DPD_SUM", "INSTAL_COUNT"),
        ("NEW_POS_DPD_RATIO", "POS_SK_DPD_DEF_MEAN", "POS_COUNT"),
        ("NEW_CREDIT_TO_PREV_CREDIT_RATIO", "AMT_CREDIT", "PREV_AMT_CREDIT_MEAN"),
        ("NEW_CURRENT_TO_APPROVED_CREDIT_RATIO", "AMT_CREDIT", "APPROVED_AMT_CREDIT_MEAN"),
        ("NEW_CURRENT_TO_INCOME_CREDIT_RATIO", "AMT_CREDIT", "AMT_INCOME_TOTAL"),
    ]

    # On ne calcule la feature que si les deux colonnes existent
    for new_feature, numerator_feature, denominator_feature in feature_pairs:
        if numerator_feature in output.columns and denominator_feature in output.columns:
            output[new_feature] = safe_divide(output[numerator_feature], output[denominator_feature])

    return output


def build_full_dataset(
    data_dir: Path | str,
    num_rows: int | None = None,
    nan_as_category: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Construit le dataset final train/test enrichi à partir de toutes les tables Home Credit.

    Pipeline global :
    1. préparation de la table principale application,
    2. création des agrégats issus des tables secondaires,
    3. suppression des colonnes peu informatives,
    4. jointure de toutes les tables à la maille client,
    5. création de features post-jointure,
    6. séparation finale en train / test,
    7. production d'un rapport de jointure.

    Parameters
    ----------
    data_dir : Path | str
        Répertoire contenant les tables brutes.
    num_rows : int | None, optional
        Nombre maximum de lignes à charger par table.
    nan_as_category : bool, optional
        Indique si les NaN doivent être traités comme une catégorie lors du one-hot encoding.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        - train_df : dataset d'entraînement enrichi
        - test_df : dataset de test enrichi
        - join_report : rapport synthétique des jointures effectuées
    """
    data_path = Path(data_dir)

    # Construction des différentes briques de features
    base = application_train_test(
        data_dir=data_path,
        num_rows=num_rows,
        nan_as_category=nan_as_category,
    )
    bureau = bureau_and_balance(data_dir=data_path, num_rows=num_rows, nan_as_category=nan_as_category)
    prev = previous_applications(data_dir=data_path, num_rows=num_rows, nan_as_category=nan_as_category)
    pos = pos_cash(data_dir=data_path, num_rows=num_rows, nan_as_category=nan_as_category)
    installments = installments_payments(data_dir=data_path, num_rows=num_rows, nan_as_category=nan_as_category)
    credit_card = credit_card_balance(data_dir=data_path, num_rows=num_rows, nan_as_category=nan_as_category)

    # Suppression des colonnes peu informatives avant les jointures
    base, _ = drop_low_information_columns(
        base,
        protected_columns=["SK_ID_CURR", "TARGET"],
    )
    bureau, _ = drop_low_information_columns(bureau)
    prev, _ = drop_low_information_columns(prev)
    pos, _ = drop_low_information_columns(pos)
    installments, _ = drop_low_information_columns(installments)
    credit_card, _ = drop_low_information_columns(credit_card)

    # Historique des jointures, utile pour audit/debug
    join_steps: list[dict[str, int]] = []

    # Point de départ : table principale
    full_df = base.copy()

    # Jointure successive de chaque bloc de features sur la clé client
    for table_name, features_df in [
        ("bureau", bureau),
        ("previous_application", prev),
        ("pos_cash_balance", pos),
        ("installments_payments", installments),
        ("credit_card_balance", credit_card),
    ]:
        before_rows = len(full_df)

        # Join left : on conserve tous les clients de la table de base
        full_df = full_df.join(features_df, how="left", on="SK_ID_CURR")

        join_steps.append(
            {
                "table_name": table_name,
                "rows_before_join": before_rows,
                "rows_after_join": len(full_df),
                "added_columns": int(features_df.shape[1]),
            }
        )

    # Création de features supplémentaires après assemblage
    full_df = add_post_join_features(full_df)

    # Séparation train / test via la présence ou non de TARGET
    train_df = full_df.loc[full_df["TARGET"].notna()].copy()
    test_df = full_df.loc[full_df["TARGET"].isna()].drop(columns=["TARGET"]).copy()

    # Sécurisation du type de la target
    train_df["TARGET"] = train_df["TARGET"].astype(int)

    # Rapport de jointure
    join_report = pd.DataFrame(join_steps)

    # Libération mémoire
    del base, bureau, prev, pos, installments, credit_card, full_df
    gc.collect()

    return train_df, test_df, join_report