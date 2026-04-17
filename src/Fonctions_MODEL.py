from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import re

import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from time import perf_counter
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from sklearn.model_selection import (
    train_test_split,
    KFold,
    StratifiedKFold,
    GridSearchCV,
    RandomizedSearchCV,
    learning_curve,
)
from sklearn.metrics import (
    r2_score,
    mean_absolute_percentage_error,
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    precision_recall_curve,
    average_precision_score,
    fbeta_score,
)
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from category_encoders import BinaryEncoder

try:
    import lightgbm as lgb
except ImportError:
    lgb = None


def encode_cat_col(
    df: pd.DataFrame,
    col_name: str,
    encoding_type: str,
    ordinal_categories: list | None = None,
) -> tuple[pd.DataFrame, object]:
    """
    Encode une seule variable catégorielle selon la méthode choisie.

    Paramètres
    ----------
    df : pd.DataFrame
        DataFrame source.
    col_name : str
        Nom de la colonne à encoder.
    encoding_type : str
        Type d'encodage à appliquer.
        Valeurs possibles :
        - "onehot"
        - "binary"
        - "ordinal"

    Retours
    -------
    tuple[pd.DataFrame, object]
        - Le DataFrame avec la colonne encodée
        - L'encodeur entraîné

    Notes
    -----
    - `onehot` convient aux variables nominales à faible cardinalité.
    - `binary` convient aux variables nominales à forte cardinalité.
    - `ordinal` convient aux variables ordinales.
    - Pour `binary`, il faut installer `category-encoders`.
    """
    if col_name not in df.columns:
        raise ValueError(f"La colonne '{col_name}' n'existe pas dans le DataFrame.")

    df_encoded = df.copy()

    if encoding_type == "onehot":
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )

        encoded_array = encoder.fit_transform(df_encoded[[col_name]])
        encoded_cols = encoder.get_feature_names_out([col_name])

        encoded_df = pd.DataFrame(
            encoded_array,
            columns=encoded_cols,
            index=df_encoded.index,
        )

        df_encoded = pd.concat(
            [df_encoded.drop(columns=[col_name]), encoded_df],
            axis=1,
        )

        return df_encoded, encoder

    if encoding_type == "binary":
        encoder = BinaryEncoder(cols=[col_name])

        encoded_df = encoder.fit_transform(df_encoded[[col_name]])

        df_encoded = pd.concat(
            [df_encoded.drop(columns=[col_name]), encoded_df],
            axis=1,
        )

        return df_encoded, encoder

    if encoding_type == "ordinal":
        encoder_kwargs = {
            "handle_unknown": "use_encoded_value",
            "unknown_value": -1,
        }

        if ordinal_categories is not None:
            encoder_kwargs["categories"] = [ordinal_categories]

        encoder = OrdinalEncoder(**encoder_kwargs)

        encoded_array = encoder.fit_transform(df_encoded[[col_name]])
        df_encoded[col_name] = encoded_array.astype(int)

        return df_encoded, encoder

    raise ValueError(
        "encoding_type doit être parmi : 'onehot', 'binary', 'ordinal'."
    )


def evaluate_regression_model(model, X, y, test_size=0.2):
    """
    Évalue un modèle de régression à l'aide d'un unique découpage train/test.

    La fonction sépare les données en un jeu d'entraînement et un jeu de test,
    entraîne le modèle sur le jeu d'entraînement, puis calcule plusieurs
    métriques de régression sur les deux sous-ensembles.

    Paramètres
    ----------
    model : estimator object
        Modèle de régression implémentant les méthodes `fit(X, y)` et `predict(X)`.
    X : pd.DataFrame ou array-like
        Matrice des variables explicatives.
    y : pd.Series ou array-like
        Vecteur cible.
    test_size : float, default=0.2
        Proportion des données réservée au jeu de test.

    Retours
    -------
    dict
        Dictionnaire contenant les métriques d'entraînement et de test :
        - "Train R2"
        - "Test R2"
        - "Train MAPE (%)"
        - "Test MAPE (%)"
        - "Train MAE"
        - "Test MAE"
        - "Train RMSE"
        - "Test RMSE"

    Notes
    -----
    - Le découpage est reproductible grâce à `random_state=42`.
    - Le `MAPE` est renvoyé en pourcentage.
    - Le `MAPE` peut devenir instable si la cible contient des valeurs proches de zéro.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    metrics = {
        "Train R2": r2_score(y_train, y_train_pred),
        "Test R2": r2_score(y_test, y_test_pred),

        "Train MAPE (%)": mean_absolute_percentage_error(y_train, y_train_pred) * 100,
        "Test MAPE (%)": mean_absolute_percentage_error(y_test, y_test_pred) * 100,

        "Train MAE": mean_absolute_error(y_train, y_train_pred),
        "Test MAE": mean_absolute_error(y_test, y_test_pred),

        "Train RMSE": np.sqrt(mean_squared_error(y_train, y_train_pred)),
        "Test RMSE": np.sqrt(mean_squared_error(y_test, y_test_pred)),
    }

    return metrics


def evaluate_regression_model_cv(model, X, y, cv):
    """
    Évalue un modèle de régression à l'aide d'une validation croisée K-Fold.

    La fonction réalise une validation croisée mélangée, entraîne le modèle
    sur chaque fold d'entraînement, évalue les performances sur les folds
    d'entraînement et de validation, puis renvoie la moyenne des métriques
    sur l'ensemble des folds.

    Paramètres
    ----------
    model : estimator object
        Modèle de régression implémentant les méthodes `fit(X, y)` et `predict(X)`.
    X : pd.DataFrame
        Matrice des variables explicatives. La fonction utilise `.iloc`,
        un DataFrame pandas est donc attendu.
    y : pd.Series
        Vecteur cible. La fonction utilise `.iloc`,
        une Series pandas est donc attendue.
    cv : int
        Nombre de folds à utiliser pour la validation croisée.

    Retours
    -------
    dict
        Dictionnaire contenant la moyenne des métriques d'entraînement
        et de test sur l'ensemble des folds :
        - "Train R2"
        - "Test R2"
        - "Train MAPE (%)"
        - "Test MAPE (%)"
        - "Train MAE"
        - "Test MAE"
        - "Train RMSE"
        - "Test RMSE"

    Notes
    -----
    - La validation croisée est reproductible grâce à `random_state=42`.
    - Le `MAPE` est renvoyé en pourcentage.
    - Le `MAPE` peut être difficile à interpréter si la cible contient
      des valeurs très faibles.
    - Les métriques d'entraînement sont elles aussi moyennées sur l'ensemble des folds.
    """
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)

    r2_train, r2_test = [], []
    mape_train, mape_test = [], []
    mae_train, mae_test = [], []
    rmse_train, rmse_test = [], []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)

        y_train_pred_real = model.predict(X_train)
        y_test_pred_real = model.predict(X_test)

        r2_train.append(r2_score(y_train, y_train_pred_real))
        r2_test.append(r2_score(y_test, y_test_pred_real))

        mape_train.append(mean_absolute_percentage_error(y_train, y_train_pred_real) * 100)
        mape_test.append(mean_absolute_percentage_error(y_test, y_test_pred_real) * 100)

        mae_train.append(mean_absolute_error(y_train, y_train_pred_real))
        mae_test.append(mean_absolute_error(y_test, y_test_pred_real))

        rmse_train.append(np.sqrt(mean_squared_error(y_train, y_train_pred_real)))
        rmse_test.append(np.sqrt(mean_squared_error(y_test, y_test_pred_real)))

    metrics = {
        "Train R2": np.mean(r2_train),
        "Test R2": np.mean(r2_test),

        "Train MAPE (%)": np.mean(mape_train),
        "Test MAPE (%)": np.mean(mape_test),

        "Train MAE": np.mean(mae_train),
        "Test MAE": np.mean(mae_test),

        "Train RMSE": np.mean(rmse_train),
        "Test RMSE": np.mean(rmse_test),
    }

    return metrics


def evaluate_classification_model_cv(model,X,y,cv,stratify=True,model_name=None,show_confusion_matrix=True,cmap="Blues",):
    """
    Evalue un modele de classification via validation croisee.

    La fonction entraine le modele sur chaque fold, calcule les metriques de
    classification sur les jeux d'entrainement et de validation, puis agrege
    les predictions de validation (out-of-fold) pour produire une matrice de
    confusion globale et un classification report global.

    Parametres
    ----------
    model : estimator object
        Modele de classification implementant `fit(X, y)` et `predict(X)`.
        Si le modele expose `predict_proba(X)` ou `decision_function(X)`,
        un score ROC AUC est aussi calcule.
    X : pd.DataFrame
        Matrice des variables explicatives. La fonction utilise `.iloc`.
    y : pd.Series
        Vecteur cible. La fonction utilise `.iloc`.
    cv : int
        Nombre de folds utilises pour la validation croisee.
    stratify : bool, default=True
        Si `True`, utilise `StratifiedKFold`. Sinon, utilise `KFold`.
    model_name : str | None, default=None
        Nom du modele affiche avant les resultats.
    show_confusion_matrix : bool, default=True
        Si `True`, affiche la matrice de confusion agregee basee sur les
        predictions out-of-fold.
    cmap : str, default="Blues"
        Palette de couleurs utilisee pour la matrice de confusion.

    Retours
    -------
    dict
        Dictionnaire contenant les moyennes et ecarts-types des metriques sur
        l'ensemble des folds :
        - "Train Accuracy"
        - "Test Accuracy"
        - "Train Accuracy Std"
        - "Test Accuracy Std"
        - "Train Precision"
        - "Test Precision"
        - "Train Precision Std"
        - "Test Precision Std"
        - "Train Recall"
        - "Test Recall"
        - "Train Recall Std"
        - "Test Recall Std"
        - "Train F1"
        - "Test F1"
        - "Train F1 Std"
        - "Test F1 Std"
        - "Train ROC AUC"
        - "Test ROC AUC"
        - "Train ROC AUC Std"
        - "Test ROC AUC Std"
        - "Train PR AUC"
        - "Test PR AUC"
        - "Train PR AUC Std"
        - "Test PR AUC Std"
        - "Execution Time Total (s)"
        - "Execution Time Mean Fold (s)"
        - "Execution Time Std Fold (s)"

    Affichage
    ---------
    La fonction affiche, pour chaque modele :
    - le nom du modele si `model_name` est renseigne ;
    - une ligne de metriques `Train` (moyenne +/- ecart-type) ;
    - une ligne de metriques `Test` (moyenne +/- ecart-type) ;
    - une matrice de confusion agregee (si `show_confusion_matrix=True`) ;
    - un `classification_report` agrege sur les predictions out-of-fold.

    Notes
    -----
    - Les metriques `Precision`, `Recall` et `F1` sont calculees avec
      `zero_division=0`.
    - Les scores `ROC AUC` et `PR AUC` sont calcules uniquement si le modele
      fournit un score continu via `predict_proba` ou `decision_function`.
      Sinon, ces metriques sont renvoyees a `np.nan`.
    - Les ecarts-types sont calcules a partir des metriques obtenues sur chaque
      fold, ce qui aide a evaluer la stabilite du modele.
    - La matrice de confusion et le `classification_report` sont calcules sur
      l'ensemble des predictions de validation agregees sur tous les folds.
    """
    splitter = (
        StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        if stratify
        else KFold(n_splits=cv, shuffle=True, random_state=42)
    )

    accuracy_train, accuracy_test = [], []
    precision_train, precision_test = [], []
    recall_train, recall_test = [], []
    f1_train, f1_test = [], []
    roc_auc_train, roc_auc_test = [], []
    pr_auc_train, pr_auc_test = [], []
    execution_times = []
    y_true_oof = pd.Series(index=y.index, dtype=y.dtype)
    y_pred_oof = pd.Series(index=y.index, dtype=y.dtype)

    split_iterator = splitter.split(X, y) if stratify else splitter.split(X)
    total_start_time = perf_counter()

    for train_idx, test_idx in split_iterator:
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        fold_start_time = perf_counter()
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        execution_times.append(perf_counter() - fold_start_time)
        y_true_oof.iloc[test_idx] = y_test.to_numpy()
        y_pred_oof.iloc[test_idx] = y_test_pred

        accuracy_train.append(accuracy_score(y_train, y_train_pred))
        accuracy_test.append(accuracy_score(y_test, y_test_pred))

        precision_train.append(precision_score(y_train, y_train_pred, zero_division=0))
        precision_test.append(precision_score(y_test, y_test_pred, zero_division=0))

        recall_train.append(recall_score(y_train, y_train_pred, zero_division=0))
        recall_test.append(recall_score(y_test, y_test_pred, zero_division=0))

        f1_train.append(f1_score(y_train, y_train_pred, zero_division=0))
        f1_test.append(f1_score(y_test, y_test_pred, zero_division=0))

        train_scores = None
        test_scores = None

        if hasattr(model, "predict_proba"):
            train_scores = model.predict_proba(X_train)[:, 1]
            test_scores = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            train_scores = model.decision_function(X_train)
            test_scores = model.decision_function(X_test)

        if train_scores is not None and test_scores is not None:
            roc_auc_train.append(roc_auc_score(y_train, train_scores))
            roc_auc_test.append(roc_auc_score(y_test, test_scores))
            pr_auc_train.append(average_precision_score(y_train, train_scores))
            pr_auc_test.append(average_precision_score(y_test, test_scores))
        else:
            roc_auc_train.append(np.nan)
            roc_auc_test.append(np.nan)
            pr_auc_train.append(np.nan)
            pr_auc_test.append(np.nan)

    def _format_mean(values):
        """
        Calcule une moyenne et renvoie un float Python arrondi.

        Si toutes les valeurs sont manquantes, la fonction renvoie `nan`.
        """
        valid_values = [value for value in values if not np.isnan(value)]
        if not valid_values:
            return float("nan")
        return round(float(np.mean(valid_values)), 4)

    def _format_std(values):
        """
        Calcule un ecart-type et renvoie un float Python arrondi.

        Si toutes les valeurs sont manquantes, la fonction renvoie `nan`.
        """
        valid_values = [value for value in values if not np.isnan(value)]
        if not valid_values:
            return float("nan")
        return round(float(np.std(valid_values)), 4)

    metrics = {
        "Train Accuracy": _format_mean(accuracy_train),
        "Test Accuracy": _format_mean(accuracy_test),
        "Train Accuracy Std": _format_std(accuracy_train),
        "Test Accuracy Std": _format_std(accuracy_test),

        "Train Precision": _format_mean(precision_train),
        "Test Precision": _format_mean(precision_test),
        "Train Precision Std": _format_std(precision_train),
        "Test Precision Std": _format_std(precision_test),

        "Train Recall": _format_mean(recall_train),
        "Test Recall": _format_mean(recall_test),
        "Train Recall Std": _format_std(recall_train),
        "Test Recall Std": _format_std(recall_test),

        "Train F1": _format_mean(f1_train),
        "Test F1": _format_mean(f1_test),
        "Train F1 Std": _format_std(f1_train),
        "Test F1 Std": _format_std(f1_test),

        "Train ROC AUC": _format_mean(roc_auc_train),
        "Test ROC AUC": _format_mean(roc_auc_test),
        "Train ROC AUC Std": _format_std(roc_auc_train),
        "Test ROC AUC Std": _format_std(roc_auc_test),

        "Train PR AUC": _format_mean(pr_auc_train),
        "Test PR AUC": _format_mean(pr_auc_test),
        "Train PR AUC Std": _format_std(pr_auc_train),
        "Test PR AUC Std": _format_std(pr_auc_test),
        "Execution Time Total (s)": round(float(perf_counter() - total_start_time), 4),
        "Execution Time Mean Fold (s)": _format_mean(execution_times),
        "Execution Time Std Fold (s)": _format_std(execution_times),
    }

    if model_name is not None:
        print(f"\n{model_name}")

    print(
        f"Train | Accuracy: {metrics['Train Accuracy']:.4f} +/- {metrics['Train Accuracy Std']:.4f} | "
        f"Precision: {metrics['Train Precision']:.4f} +/- {metrics['Train Precision Std']:.4f} | "
        f"Recall: {metrics['Train Recall']:.4f} +/- {metrics['Train Recall Std']:.4f} | "
        f"F1: {metrics['Train F1']:.4f} +/- {metrics['Train F1 Std']:.4f} | "
        f"ROC AUC: {metrics['Train ROC AUC']:.4f} +/- {metrics['Train ROC AUC Std']:.4f} | "
        f"PR-AUC: {metrics['Train PR AUC']:.4f} +/- {metrics['Train PR AUC Std']:.4f}"
    )
    print(
        f"Test  | Accuracy: {metrics['Test Accuracy']:.4f} +/- {metrics['Test Accuracy Std']:.4f} | "
        f"Precision: {metrics['Test Precision']:.4f} +/- {metrics['Test Precision Std']:.4f} | "
        f"Recall: {metrics['Test Recall']:.4f} +/- {metrics['Test Recall Std']:.4f} | "
        f"F1: {metrics['Test F1']:.4f} +/- {metrics['Test F1 Std']:.4f} | "
        f"ROC AUC: {metrics['Test ROC AUC']:.4f} +/- {metrics['Test ROC AUC Std']:.4f} | "
        f"PR-AUC: {metrics['Test PR AUC']:.4f} +/- {metrics['Test PR AUC Std']:.4f}"
    )
    print(
        f"Temps | Total CV: {metrics['Execution Time Total (s)']:.4f}s | "
        f"Moyenne par fold: {metrics['Execution Time Mean Fold (s)']:.4f}s +/- "
        f"{metrics['Execution Time Std Fold (s)']:.4f}s"
    )
    if show_confusion_matrix:
        class_labels = np.unique(y_true_oof)
        display_labels = [str(int(label)) for label in class_labels]
        cm = confusion_matrix(y_true_oof, y_pred_oof, labels=class_labels)

        if cm.shape == (2, 2):
            total_count = cm.sum()
            real_first_class_count = cm[0, :].sum()
            real_second_class_count = cm[1, :].sum()

            labels = np.array(
                [
                    [
                        f"{cm[0, 0]}\n({cm[0, 0] / total_count:.1%})",
                        f"{cm[0, 1]}\n({cm[0, 1] / total_count:.1%})",
                    ],
                    [
                        f"{cm[1, 0]}\n({cm[1, 0] / total_count:.1%})",
                        f"{cm[1, 1]}\n({cm[1, 1] / total_count:.1%})",
                    ],
                ]
            )

            plt.figure(figsize=(6, 5))
            sns.heatmap(
                cm,
                annot=labels,
                fmt="",
                cmap=cmap,
                cbar=False,
                xticklabels=display_labels,
                yticklabels=display_labels,
            )
            title = "Matrice de confusion agrégée (validation croisée)"
            if model_name is not None:
                title = f"{title} - {model_name}"
            distribution_text = (
                f"Repartition reelle | {display_labels[0]}: "
                f"{real_first_class_count} ({real_first_class_count / total_count:.1%}) | "
                f"{display_labels[1]}: {real_second_class_count} "
                f"({real_second_class_count / total_count:.1%})"
            )
            plt.title(f"{title}\n{distribution_text}")
            plt.xlabel("Prediction")
            plt.ylabel("Reel")
            plt.tight_layout()
            plt.show()
        else:
            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm,
                display_labels=np.unique(y_true_oof),
            )
            disp.plot(cmap=cmap, values_format="d")
            title = "Matrice de confusion agrégée (validation croisée)"
            if model_name is not None:
                title = f"{title} - {model_name}"
            plt.title(title)
            plt.grid(False)
            plt.show()

    print("\nClassification report (out-of-fold):")
    print(
        classification_report(
            y_true_oof,
            y_pred_oof,
            zero_division=0,
        )
    )

    return metrics


def evaluate_precision_recall_threshold(
    model,
    X_train,
    y_train,
    X_eval,
    y_eval,
    model_name=None,
    plot_curve=True,
):
    """
    Evalue un modele sur une courbe precision-rappel et propose un seuil.

    La fonction entraine une copie du modele sur `X_train`, calcule les scores
    sur `X_eval`, puis identifie le seuil qui maximise le F1-score sur le jeu
    d'evaluation.

    Parametres
    ----------
    model : estimator object
        Modele de classification implementant `fit(X, y)` et exposant
        `predict_proba(X)` ou `decision_function(X)`.
    X_train : pd.DataFrame
        Variables explicatives d'entrainement.
    y_train : pd.Series
        Cible d'entrainement.
    X_eval : pd.DataFrame
        Variables explicatives du jeu d'evaluation.
    y_eval : pd.Series
        Cible du jeu d'evaluation.
    model_name : str | None, default=None
        Nom affiche dans la sortie console et le graphique.
    plot_curve : bool, default=True
        Si `True`, affiche l'evolution de la precision et du recall en
        fonction du seuil.

    Retours
    -------
    dict
        Dictionnaire contenant :
        - "PR AUC"
        - "Meilleur Seuil"
        - "Precision Test Au Meilleur Seuil"
        - "Recall Test Au Meilleur Seuil"
        - "F1 Test Au Meilleur Seuil"
    """
    fitted_model = clone(model)
    fitted_model.fit(X_train, y_train)

    if hasattr(fitted_model, "predict_proba"):
        y_scores = fitted_model.predict_proba(X_eval)[:, 1]
    elif hasattr(fitted_model, "decision_function"):
        y_scores = fitted_model.decision_function(X_eval)
    else:
        raise ValueError(
            "Le modele doit exposer `predict_proba` ou `decision_function`."
        )

    precision, recall, thresholds = precision_recall_curve(y_eval, y_scores)
    average_precision = average_precision_score(y_eval, y_scores)

    if thresholds.size == 0:
        raise ValueError(
            "Impossible de calculer un seuil. Verifiez que `y_eval` contient bien les deux classes."
        )

    precision_for_thresholds = precision[:-1]
    recall_for_thresholds = recall[:-1]
    f1_scores = np.divide(
        2 * precision_for_thresholds * recall_for_thresholds,
        precision_for_thresholds + recall_for_thresholds,
        out=np.zeros_like(precision_for_thresholds),
        where=(precision_for_thresholds + recall_for_thresholds) != 0,
    )

    best_idx = int(np.argmax(f1_scores))
    best_threshold = float(thresholds[best_idx])

    results = {
        "PR AUC": round(float(average_precision), 4),
        "Meilleur Seuil": round(best_threshold, 4),
        "Precision Test Au Meilleur Seuil": round(float(precision_for_thresholds[best_idx]), 4),
        "Recall Test Au Meilleur Seuil": round(float(recall_for_thresholds[best_idx]), 4),
        "F1 Test Au Meilleur Seuil": round(float(f1_scores[best_idx]), 4),
    }

    if model_name is not None:
        print(f"\n{model_name}")

    print(
        f"PR-AUC : {results['PR AUC']:.4f} | "
        f"Meilleur seuil (max F1) : {results['Meilleur Seuil']:.4f}"
    )
    print(
        f"Au seuil {results['Meilleur Seuil']:.4f} | "
        f"Precision test : {results['Precision Test Au Meilleur Seuil']:.4f} | "
        f"Recall test : {results['Recall Test Au Meilleur Seuil']:.4f} | "
        f"F1 test : {results['F1 Test Au Meilleur Seuil']:.4f}"
    )

    if plot_curve:
        plt.figure(figsize=(7, 5))
        plt.plot(
            thresholds,
            precision_for_thresholds,
            label="Precision",
            linewidth=2,
            color="tab:blue",
        )
        plt.plot(
            thresholds,
            recall_for_thresholds,
            label="Recall",
            linewidth=2,
            color="tab:orange",
        )
        plt.axvline(
            x=results["Meilleur Seuil"],
            color="red",
            linestyle="--",
            linewidth=2,
            label=(
                f"Meilleur F1 test = {results['F1 Test Au Meilleur Seuil']:.4f} "
                f"(seuil = {results['Meilleur Seuil']:.4f})"
            ),
        )
        title = "Precision / Recall selon le seuil"
        if model_name is not None:
            title = f"{title} - {model_name}"
        plt.title(title)
        plt.xlabel("Seuil")
        plt.ylabel("Scores Precision/Recall")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    return results


def optimize_classification_hyperparameters(
    model,
    param_grid,
    X,
    y,
    cv=5,
    scoring="f1",
    stratify=True,
    search_type="grid",
    n_iter=20,
    n_jobs=-1,
    model_name=None,
):
    """
    Optimise les hyperparametres d'un modele de classification.

    La fonction lance une recherche d'hyperparametres via `GridSearchCV` ou
    `RandomizedSearchCV`, puis affiche le meilleur score moyen de validation
    croisee et les meilleurs parametres trouves.

    Parametres
    ----------
    model : estimator object
        Modele de classification a optimiser.
    param_grid : dict
        Grille de parametres a tester. Pour un pipeline, utiliser les noms de
        type `etape__parametre`.
    X : pd.DataFrame
        Matrice des variables explicatives.
    y : pd.Series
        Vecteur cible.
    cv : int, default=5
        Nombre de folds pour la validation croisee.
    scoring : str, default="f1"
        Metrique d'optimisation transmise a scikit-learn.
    stratify : bool, default=True
        Si `True`, utilise `StratifiedKFold`. Sinon, utilise `KFold`.
    search_type : {"grid", "random"}, default="grid"
        Type de recherche a utiliser.
    n_iter : int, default=20
        Nombre d'iterations si `search_type="random"`.
    n_jobs : int, default=-1
        Nombre de coeurs utilises par la recherche.
    model_name : str | None, default=None
        Nom du modele affiche dans la sortie console.

    Retours
    -------
    dict
        Dictionnaire contenant :
        - "Best Estimator"
        - "Best Params"
        - "Best CV Score"
        - "Search Object"
    """
    splitter = (
        StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        if stratify
        else KFold(n_splits=cv, shuffle=True, random_state=42)
    )

    if search_type == "grid":
        search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring=scoring,
            cv=splitter,
            n_jobs=n_jobs,
            refit=True,
        )
    elif search_type == "random":
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring=scoring,
            cv=splitter,
            n_jobs=n_jobs,
            random_state=42,
            refit=True,
        )
    else:
        raise ValueError("`search_type` doit valoir 'grid' ou 'random'.")

    search.fit(X, y)

    results = {
        "Best Estimator": search.best_estimator_,
        "Best Params": search.best_params_,
        "Best CV Score": round(float(search.best_score_), 4),
        "Search Object": search,
    }

    if model_name is not None:
        print(f"\n{model_name}")

    search_label = "GridSearchCV" if search_type == "grid" else "RandomizedSearchCV"
    print(
        f"{search_label} | Scoring : {scoring} | "
        f"Meilleur score CV : {results['Best CV Score']:.4f}"
    )
    print(f"Meilleurs hyperparametres : {results['Best Params']}")

    return results


def plot_model_feature_importance(
    model,
    X,
    y=None,
    top_n=15,
    model_name=None,
):
    """
    Affiche l'importance globale des variables via `feature_importances_`.

    Parametres
    ----------
    model : estimator object
        Modele deja entraine, ou modele a entrainer si `y` est fourni.
        Le modele doit exposer l'attribut `feature_importances_`.
    X : pd.DataFrame
        Variables explicatives.
    y : pd.Series | None, default=None
        Cible. Si renseignee, le modele est entraine sur `X, y`.
    top_n : int, default=15
        Nombre maximum de variables affichees.
    model_name : str | None, default=None
        Nom du modele affiche dans le titre.

    Retours
    -------
    pd.DataFrame
        Tableau trie par importance decroissante.
    """
    fitted_model = clone(model) if y is not None else model

    if y is not None:
        fitted_model.fit(X, y)

    if not hasattr(fitted_model, "feature_importances_"):
        raise ValueError("Le modele doit exposer l'attribut `feature_importances_`.")

    importance_df = pd.DataFrame(
        {
            "Variable": X.columns,
            "Importance": fitted_model.feature_importances_,
        }
    ).sort_values("Importance", ascending=False)

    top_importance_df = importance_df.head(top_n).iloc[::-1]

    plt.figure(figsize=(8, max(4, top_n * 0.35)))
    plt.barh(top_importance_df["Variable"], top_importance_df["Importance"], color="steelblue")
    title = "Importance globale des variables"
    if model_name is not None:
        title = f"{title} - {model_name}"
    plt.title(title)
    plt.xlabel("Importance")
    plt.ylabel("Variables")
    plt.tight_layout()
    plt.show()

    return importance_df


def plot_permutation_feature_importance(
    model,
    X_train,
    y_train,
    X_eval,
    y_eval,
    scoring="f1",
    n_repeats=10,
    top_n=15,
    random_state=42,
    model_name=None,
):
    """
    Affiche l'importance globale des variables via permutation importance.

    Parametres
    ----------
    model : estimator object
        Modele de classification a entrainer.
    X_train : pd.DataFrame
        Variables explicatives d'entrainement.
    y_train : pd.Series
        Cible d'entrainement.
    X_eval : pd.DataFrame
        Variables explicatives d'evaluation.
    y_eval : pd.Series
        Cible d'evaluation.
    scoring : str, default="f1"
        Metrique utilisee pour evaluer la perte de performance.
    n_repeats : int, default=10
        Nombre de permutations par variable.
    top_n : int, default=15
        Nombre maximum de variables affichees.
    random_state : int, default=42
        Graine aleatoire.
    model_name : str | None, default=None
        Nom du modele affiche dans le titre.

    Retours
    -------
    pd.DataFrame
        Tableau trie par importance moyenne decroissante.
    """
    fitted_model = clone(model)
    fitted_model.fit(X_train, y_train)

    perm = permutation_importance(
        fitted_model,
        X_eval,
        y_eval,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )

    importance_df = pd.DataFrame(
        {
            "Variable": X_eval.columns,
            "Importance Moyenne": perm.importances_mean,
            "Ecart-Type": perm.importances_std,
        }
    ).sort_values("Importance Moyenne", ascending=False)

    top_importance_df = importance_df.head(top_n).iloc[::-1]

    plt.figure(figsize=(8, max(4, top_n * 0.35)))
    plt.barh(
        top_importance_df["Variable"],
        top_importance_df["Importance Moyenne"],
        xerr=top_importance_df["Ecart-Type"],
        color="darkorange",
    )
    title = f"Importance globale par permutation ({scoring})"
    if model_name is not None:
        title = f"{title} - {model_name}"
    plt.title(title)
    plt.xlabel("Perte de score moyenne")
    plt.ylabel("Variables")
    plt.tight_layout()
    plt.show()

    return importance_df


# ---------------------------------------------------------------------------
# Helpers modélisation / MLflow utilisés dans les notebooks du projet
# ---------------------------------------------------------------------------

DEFAULT_EXPERIMENT_NAME = "P6_HOME_CREDIT_DEFAULT_RISK"
DEFAULT_REGISTERED_MODEL_NAME = "P6_HOME_CREDIT_DEFAULT_RISK_MODEL"


@dataclass(frozen=True)
class MlflowTrackingContext:
    """
    Petit conteneur immuable pour regrouper les informations essentielles
    de configuration MLflow.

    Attributes
    ----------
    client : MlflowClient
        Client MLflow utilisé pour interagir avec le tracking server / backend.
    tracking_uri : str
        URI du backend de tracking MLflow.
    artifact_location : str
        Emplacement racine où seront stockés les artefacts MLflow.
    experiment_name : str
        Nom de l'expérience MLflow active.
    """

    client: MlflowClient
    tracking_uri: str
    artifact_location: str
    experiment_name: str


def configure_mlflow_tracking(
    project_root: Path,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
) -> MlflowTrackingContext:
    """
    Configure un backend MLflow local basé sur SQLite + dossier d'artefacts.

    Parameters
    ----------
    project_root : Path
        Racine du projet.
    experiment_name : str, optional
        Nom de l'expérience MLflow à activer.

    Returns
    -------
    MlflowTrackingContext
        Conteneur avec client, URI de tracking, emplacement des artefacts
        et nom d'expérience.
    """
    tracking_db_path = project_root / "mlflow.db"
    artifact_root = project_root / "mlartifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)

    tracking_uri = f"sqlite:///{tracking_db_path.as_posix()}"
    artifact_location = artifact_root.resolve().as_uri()

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = client.create_experiment(
            name=experiment_name,
            artifact_location=artifact_location,
        )
        experiment = client.get_experiment(experiment_id)

    mlflow.set_experiment(experiment_name)
    return MlflowTrackingContext(
        client=client,
        tracking_uri=tracking_uri,
        artifact_location=experiment.artifact_location,
        experiment_name=experiment_name,
    )


def resolve_lightgbm_device_type(random_state: int = 42) -> str:
    """
    Détecte si LightGBM peut s'exécuter sur GPU dans l'environnement courant.

    Returns
    -------
    str
        "gpu" si un fit de probe LightGBM sur GPU fonctionne, sinon "cpu".
    """
    if lgb is None:
        return "cpu"

    probe_X = pd.DataFrame(
        np.random.default_rng(seed=random_state).random((512, 8), dtype=np.float32)
    )
    probe_y = pd.Series(
        np.random.default_rng(seed=random_state + 1).integers(0, 2, size=512)
    )

    try:
        probe_model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=5,
            random_state=random_state,
            device_type="gpu",
            verbosity=-1,
        )
        probe_model.fit(probe_X, probe_y)
        return "gpu"
    except Exception:
        return "cpu"


def build_lightgbm_estimator(
    random_state: int = 42,
    device_type: str = "cpu",
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
    min_child_samples: int = 20,
    reg_lambda: float = 0.0,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    class_weight: str | dict | None = "balanced",
    verbosity: int = -1,
    gpu_platform_id: int | None = None,
    gpu_device_id: int | None = None,
    **extra_params,
):
    """
    Construit un estimateur LightGBM binaire avec paramètres projet.
    """
    if lgb is None:
        raise ImportError("lightgbm n'est pas installé dans l'environnement.")

    params = {
        "objective": "binary",
        "learning_rate": learning_rate,
        "n_estimators": n_estimators,
        "num_leaves": num_leaves,
        "min_child_samples": min_child_samples,
        "reg_lambda": reg_lambda,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "class_weight": class_weight,
        "random_state": random_state,
        "verbosity": verbosity,
    }

    if device_type == "gpu":
        params["device_type"] = "gpu"
        if gpu_platform_id is not None:
            params["gpu_platform_id"] = gpu_platform_id
        if gpu_device_id is not None:
            params["gpu_device_id"] = gpu_device_id

    params.update(extra_params)
    return lgb.LGBMClassifier(**params)


def build_baseline_benchmark_catalog() -> pd.DataFrame:
    """
    Retourne le catalogue des modèles baseline du projet.
    """
    rows = [
        {
            "model_name": "dummy_classifier",
            "family": "Baseline naïve",
            "benchmark_scope": "full_dataset",
            "why_included": "référence minimale pour montrer qu'une forte accuracy ne suffit pas",
        },
        {
            "model_name": "sgd_log_loss",
            "family": "Baseline linéaire",
            "benchmark_scope": "full_dataset",
            "why_included": "baseline linéaire scalable, mieux adaptée que LogisticRegression au volume du projet",
        },
        {
            "model_name": "hist_gradient_boosting",
            "family": "Booster scikit-learn",
            "benchmark_scope": "full_dataset",
            "why_included": "baseline tabulaire robuste, simple à comparer à LightGBM",
        },
    ]

    if lgb is not None:
        rows.append(
            {
                "model_name": "lightgbm_bonus",
                "family": "Boosting externe",
                "benchmark_scope": "full_dataset",
                "why_included": "benchmark bonus plus puissant et bon candidat pour l'optimisation",
            }
        )

    return pd.DataFrame(rows)


def build_optimization_candidate(
    model_name: str,
    random_state: int = 42,
    lightgbm_device_type: str = "cpu",
):
    """
    Construit le modèle à optimiser à partir de son identifiant projet.
    """
    if model_name == "dummy_classifier":
        return DummyClassifier(strategy="most_frequent")

    if model_name == "sgd_log_loss":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    SGDClassifier(
                        loss="log_loss",
                        alpha=1e-4,
                        max_iter=2000,
                        tol=1e-3,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        )

    if model_name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=300,
            min_samples_leaf=100,
            random_state=random_state,
        )

    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=20,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        )

    if model_name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=20,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        )

    if model_name == "lightgbm_bonus":
        return build_lightgbm_estimator(
            random_state=random_state,
            device_type=lightgbm_device_type,
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            reg_lambda=0.0,
        )

    raise ValueError(f"Modèle non supporté pour l'optimisation : {model_name}")


def get_classification_tuning_grid(model_name: str) -> dict[str, list]:
    """
    Retourne la grille de recherche d'hyperparamètres utilisée en notebook 03.
    """
    grids: dict[str, dict[str, list]] = {
        "dummy_classifier": {},
        "sgd_log_loss": {
            "model__alpha": [1e-5, 1e-4, 1e-3],
            "model__penalty": ["l2", "elasticnet"],
        },
        "hist_gradient_boosting": {
            "learning_rate": [0.03, 0.05],
            "max_iter": [200, 300],
            "min_samples_leaf": [50, 100],
        },
        "random_forest": {
            "n_estimators": [300, 500],
            "min_samples_leaf": [10, 20],
            "max_features": ["sqrt", 0.5],
        },
        "extra_trees": {
            "n_estimators": [300, 500],
            "min_samples_leaf": [10, 20],
            "max_features": ["sqrt", 0.5],
        },
        "lightgbm_bonus": {
            "learning_rate": [0.03, 0.05],
            "num_leaves": [31, 63],
            "min_child_samples": [20, 100],
            "reg_lambda": [0.0, 1.0],
        },
    }

    if model_name not in grids:
        raise ValueError(f"Aucune grille définie pour le modèle : {model_name}")

    return grids[model_name]


def build_classification_tuning_guide(model_name: str) -> pd.DataFrame:
    """
    Produit un tableau pédagogique décrivant les hyperparamètres explorés.
    """
    baseline_model = build_optimization_candidate(
        model_name=model_name,
        random_state=42,
        lightgbm_device_type="cpu",
    )
    param_grid = get_classification_tuning_grid(model_name)

    role_lookup = {
        "learning_rate": "pas d'apprentissage",
        "num_leaves": "complexité des arbres LightGBM",
        "min_child_samples": "taille minimale d'une feuille LightGBM",
        "reg_lambda": "régularisation L2 LightGBM",
        "max_iter": "nombre d'itérations de boosting",
        "min_samples_leaf": "taille minimale d'une feuille",
        "n_estimators": "nombre d'arbres",
        "max_features": "part des variables candidates par split",
        "model__alpha": "force de régularisation linéaire",
        "model__penalty": "type de pénalisation linéaire",
    }

    baseline_params = baseline_model.get_params(deep=True)
    rows = []
    for hyperparameter, search_values in param_grid.items():
        rows.append(
            {
                "model_name": model_name,
                "hyperparameter": hyperparameter,
                "role": role_lookup.get(hyperparameter, "hyperparamètre de tuning"),
                "baseline_value": baseline_params.get(hyperparameter),
                "search_values": json.dumps(search_values, ensure_ascii=False),
            }
        )

    return pd.DataFrame(rows)


def register_model_version(
    model_uri,
    client: MlflowClient,
    registered_model_name: str = DEFAULT_REGISTERED_MODEL_NAME,
    model_version_tags: dict[str, str | int | float] | None = None,
):
    """
    Enregistre une nouvelle version de modèle dans le Model Registry MLflow.

    Parameters
    ----------
    model_uri : str | object
        URI du modèle à enregistrer. Peut être une chaîne ou un objet retourné
        par `mlflow.sklearn.log_model`.
    client : MlflowClient
        Client MLflow configuré sur le même backend.
    registered_model_name : str, optional
        Nom du modèle enregistré.
    model_version_tags : dict | None, optional
        Tags à appliquer à la version créée.

    Returns
    -------
    ModelVersion
        Version créée dans le registry.
    """
    model_version_tags = model_version_tags or {}

    try:
        client.get_registered_model(registered_model_name)
    except Exception:
        client.create_registered_model(registered_model_name)

    if hasattr(model_uri, "model_uri"):
        source_model_uri = model_uri.model_uri
    else:
        source_model_uri = str(model_uri)

    model_version = mlflow.register_model(
        model_uri=source_model_uri,
        name=registered_model_name,
    )

    for key, value in model_version_tags.items():
        client.set_model_version_tag(
            name=registered_model_name,
            version=model_version.version,
            key=str(key),
            value=str(value),
        )

    return model_version


def business_cost(
    y_true,
    y_pred,
    fn_cost: float = 10.0,
    fp_cost: float = 1.0,
) -> dict[str, float]:
    """
    Calcule un coût métier asymétrique à partir des prédictions binaires.

    Parameters
    ----------
    y_true : array-like
        Vraies étiquettes.
    y_pred : array-like
        Prédictions binaires (0/1).
    fn_cost : float, optional
        Coût associé à un faux négatif.
    fp_cost : float, optional
        Coût associé à un faux positif.

    Returns
    -------
    dict[str, float]
        Dictionnaire contenant :
        - le nombre de faux négatifs,
        - le nombre de faux positifs,
        - le coût total métier,
        - le coût moyen par observation.

    Notes
    -----
    Ici, on modélise explicitement un cas où rater un défaut de paiement
    (faux négatif) coûte plus cher que refuser à tort un client solvable
    (faux positif).
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    # Comptage des erreurs selon leur type
    false_negatives = int(((y_true_arr == 1) & (y_pred_arr == 0)).sum())
    false_positives = int(((y_true_arr == 0) & (y_pred_arr == 1)).sum())

    # Coût total pondéré par les coûts métier
    total_cost = false_negatives * fn_cost + false_positives * fp_cost

    # Coût moyen par observation
    average_cost = total_cost / len(y_true_arr) if len(y_true_arr) else np.nan

    return {
        "false_negatives": false_negatives,
        "false_positives": false_positives,
        "business_cost": float(total_cost),
        "business_cost_per_obs": float(average_cost),
    }


def score_to_probability(model, X) -> np.ndarray:
    """
    Convertit la sortie d'un modèle en score continu entre 0 et 1.

    Priorité :
    1. `predict_proba` si disponible ;
    2. `decision_function` sinon, avec normalisation min-max ;
    3. erreur explicite sinon.

    Parameters
    ----------
    model : estimator
        Modèle scikit-learn compatible.
    X : array-like or pd.DataFrame
        Données d'entrée.

    Returns
    -------
    np.ndarray
        Scores continus compris entre 0 et 1.

    Raises
    ------
    ValueError
        Si le modèle ne fournit ni `predict_proba`, ni `decision_function`.

    Notes
    -----
    Attention : la normalisation min-max d'un `decision_function` ne produit
    pas une vraie probabilité calibrée. C'est un score mis à l'échelle.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]

    if hasattr(model, "decision_function"):
        raw_scores = np.asarray(model.decision_function(X), dtype=float)
        min_score = raw_scores.min()
        max_score = raw_scores.max()

        # Mise à l'échelle entre 0 et 1 pour obtenir un score comparable
        return (raw_scores - min_score) / (max_score - min_score + 1e-12)

    raise ValueError("Le modele doit exposer `predict_proba` ou `decision_function`.")


def classification_metrics_at_threshold(
    y_true,
    y_scores,
    threshold: float = 0.5,
    fn_cost: float = 10.0,
    fp_cost: float = 1.0,
) -> dict[str, float]:
    """
    Calcule un ensemble cohérent de métriques de classification pour un seuil donné.

    Parameters
    ----------
    y_true : array-like
        Vérité terrain.
    y_scores : array-like
        Scores continus du modèle.
    threshold : float, optional
        Seuil de conversion score -> prédiction binaire.
    fn_cost : float, optional
        Coût métier d'un faux négatif.
    fp_cost : float, optional
        Coût métier d'un faux positif.

    Returns
    -------
    dict[str, float]
        Dictionnaire de métriques :
        accuracy, precision, recall, f1, roc_auc, pr_auc,
        business_fbeta, coût métier, etc.
    """
    y_true_arr = np.asarray(y_true)
    y_scores_arr = np.asarray(y_scores, dtype=float)

    # Binarisation des scores
    y_pred = (y_scores_arr >= threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true_arr, y_pred)),
        "precision": float(precision_score(y_true_arr, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true_arr, y_pred, zero_division=0)),
        # Beta est dérivé des coûts métier :
        # plus FN coûte cher, plus le rappel est favorisé
        "business_fbeta": float(
            fbeta_score(
                y_true_arr,
                y_pred,
                beta=float(np.sqrt(fn_cost / fp_cost)),
                zero_division=0,
            )
        ),
        "roc_auc": float(roc_auc_score(y_true_arr, y_scores_arr)),
        "pr_auc": float(average_precision_score(y_true_arr, y_scores_arr)),
        "threshold": float(threshold),
    }

    # On ajoute les métriques de coût métier au bloc standard
    metrics.update(business_cost(y_true_arr, y_pred, fn_cost=fn_cost, fp_cost=fp_cost))
    return metrics


def evaluate_classifier_cv(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
    threshold: float = 0.5,
    fn_cost: float = 10.0,
    fp_cost: float = 1.0,
    random_state: int = 42,
    train_metric_sample_size: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Évalue un classifieur en validation croisée stratifiée.

    Parameters
    ----------
    model : estimator
        Modèle scikit-learn compatible.
    X : pd.DataFrame
        Matrice de features.
    y : pd.Series
        Variable cible binaire.
    cv : int, optional
        Nombre de folds.
    threshold : float, optional
        Seuil de décision binaire.
    fn_cost : float, optional
        Coût métier d'un faux négatif.
    fp_cost : float, optional
        Coût métier d'un faux positif.
    random_state : int, optional
        Graine aléatoire pour la reproductibilité.
    train_metric_sample_size : int | None, optional
        Si renseigné, les métriques train sont estimées sur un sous-échantillon
        du train afin de limiter le temps de calcul.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        - folds_df : détail fold par fold et split par split
        - summary_df : résumé moyen et écart-type par split

    Notes
    -----
    Cette fonction entraîne un clone du modèle à chaque fold.
    Elle renvoie à la fois les métriques train et validation, ce qui permet
    d'évaluer l'overfitting.
    """
    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    fold_rows: list[dict[str, float | int | str]] = []

    for fold_idx, (train_idx, valid_idx) in enumerate(splitter.split(X, y), start=1):
        # Découpage du fold courant
        X_train = X.iloc[train_idx]
        X_valid = X.iloc[valid_idx]
        y_train = y.iloc[train_idx]
        y_valid = y.iloc[valid_idx]

        # Clone pour éviter tout partage d'état entre folds
        fitted_model = clone(model)

        # Mesure du temps d'entraînement
        fold_fit_start = perf_counter()
        fitted_model.fit(X_train, y_train)
        fold_fit_duration = perf_counter() - fold_fit_start

        # Option d'accélération : métriques train sur sous-échantillon
        X_train_metrics = X_train
        y_train_metrics = y_train
        if train_metric_sample_size is not None and len(X_train) > train_metric_sample_size:
            sampled_index = y_train.sample(
                n=train_metric_sample_size,
                random_state=random_state + fold_idx,
            ).index
            X_train_metrics = X_train.loc[sampled_index]
            y_train_metrics = y_train.loc[sampled_index]

        # Scores continus
        train_scores = score_to_probability(fitted_model, X_train_metrics)
        valid_scores = score_to_probability(fitted_model, X_valid)

        # Calcul des métriques
        train_metrics = classification_metrics_at_threshold(
            y_true=y_train_metrics,
            y_scores=train_scores,
            threshold=threshold,
            fn_cost=fn_cost,
            fp_cost=fp_cost,
        )
        valid_metrics = classification_metrics_at_threshold(
            y_true=y_valid,
            y_scores=valid_scores,
            threshold=threshold,
            fn_cost=fn_cost,
            fp_cost=fp_cost,
        )

        # Enrichissement des résultats avec du contexte
        train_metrics["fold"] = fold_idx
        train_metrics["split"] = "train"
        train_metrics["fit_time_seconds"] = float(fold_fit_duration)
        train_metrics["train_rows"] = int(len(X_train))
        train_metrics["valid_rows"] = int(len(X_valid))
        train_metrics["metric_rows"] = int(len(X_train_metrics))

        valid_metrics["fold"] = fold_idx
        valid_metrics["split"] = "valid"
        valid_metrics["fit_time_seconds"] = float(fold_fit_duration)
        valid_metrics["train_rows"] = int(len(X_train))
        valid_metrics["valid_rows"] = int(len(X_valid))
        valid_metrics["metric_rows"] = int(len(X_valid))

        fold_rows.extend([train_metrics, valid_metrics])

    # Détail complet fold par fold
    folds_df = pd.DataFrame(fold_rows)

    # Résumé moyen + écart-type par split
    summary_df = (
        folds_df.groupby("split")
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            precision_mean=("precision", "mean"),
            precision_std=("precision", "std"),
            recall_mean=("recall", "mean"),
            recall_std=("recall", "std"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            business_fbeta_mean=("business_fbeta", "mean"),
            business_fbeta_std=("business_fbeta", "std"),
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            pr_auc_mean=("pr_auc", "mean"),
            pr_auc_std=("pr_auc", "std"),
            business_cost_mean=("business_cost", "mean"),
            business_cost_std=("business_cost", "std"),
            business_cost_per_obs_mean=("business_cost_per_obs", "mean"),
            business_cost_per_obs_std=("business_cost_per_obs", "std"),
            fit_time_seconds_mean=("fit_time_seconds", "mean"),
            fit_time_seconds_std=("fit_time_seconds", "std"),
        )
        .reset_index()
    )

    return folds_df, summary_df


def flatten_cv_summary(summary_df: pd.DataFrame) -> dict[str, float]:
    """
    Transforme un résumé CV structuré par split en dictionnaire plat.

    Parameters
    ----------
    summary_df : pd.DataFrame
        DataFrame contenant par exemple une ligne 'train' et une ligne 'valid'.

    Returns
    -------
    dict[str, float]
        Dictionnaire du type :
        {
            "train_accuracy_mean": ...,
            "valid_accuracy_mean": ...,
            ...
        }

    Notes
    -----
    Ce format est particulièrement pratique pour logger des métriques dans MLflow.
    """
    flat_summary: dict[str, float] = {}
    for _, summary_row in summary_df.iterrows():
        split_name = summary_row["split"]
        for column in summary_df.columns:
            if column == "split":
                continue
            flat_summary[f"{split_name}_{column}"] = float(summary_row[column])
    return flat_summary


def evaluate_holdout(
    model,
    X_train,
    y_train,
    X_holdout,
    y_holdout,
    threshold: float = 0.5,
    fn_cost: float = 10.0,
    fp_cost: float = 1.0,
):
    """
    Entraîne un modèle sur tout le train puis l'évalue sur un jeu holdout.

    Returns
    -------
    tuple
        - fitted_model : modèle entraîné
        - holdout_scores : scores continus sur le holdout
        - holdout_metrics : métriques calculées sur le holdout
    """
    fitted_model = clone(model)

    # Entraînement complet
    fit_start = perf_counter()
    fitted_model.fit(X_train, y_train)
    fit_duration = perf_counter() - fit_start

    # Scorage holdout
    holdout_scores = score_to_probability(fitted_model, X_holdout)

    # Évaluation
    holdout_metrics = classification_metrics_at_threshold(
        y_true=y_holdout,
        y_scores=holdout_scores,
        threshold=threshold,
        fn_cost=fn_cost,
        fp_cost=fp_cost,
    )
    holdout_metrics["fit_time_seconds"] = float(fit_duration)
    return fitted_model, holdout_scores, holdout_metrics


def sanitize_model_params(model) -> dict[str, str]:
    """
    Nettoie les paramètres d'un estimateur pour faciliter leur logging.

    Parameters
    ----------
    model : estimator
        Estimateur scikit-learn ou compatible.

    Returns
    -------
    dict[str, str]
        Dictionnaire des paramètres convertis en chaînes courtes.

    Notes
    -----
    Les sous-objets estimateurs sont ignorés pour éviter des logs trop verbeux
    ou non sérialisables.
    """
    clean_params = {}
    for key, value in model.get_params(deep=True).items():
        if hasattr(value, "get_params"):
            continue

        value_str = str(value)
        clean_params[key] = value_str[:200] if len(value_str) > 200 else value_str

    return clean_params


def prepare_modeling_tables(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    id_col: str = "SK_ID_CURR",
    target_col: str = "TARGET",
    missing_threshold: float = 0.95,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    Prépare les matrices de modélisation à partir des DataFrames train et test.

    Étapes :
    - suppression de l'identifiant et de la cible des features ;
    - suppression des colonnes trop manquantes ;
    - suppression des colonnes constantes ;
    - alignement strict train / test ;
    - conversion des booléens texte en numérique ;
    - suppression défensive des colonnes `object` restantes ;
    - normalisation des noms de colonnes ;
    - production d'un rapport qualité.

    Parameters
    ----------
    train_df : pd.DataFrame
        Table train enrichie.
    test_df : pd.DataFrame
        Table test enrichie.
    id_col : str, optional
        Nom de la colonne identifiant.
    target_col : str, optional
        Nom de la colonne cible.
    missing_threshold : float, optional
        Seuil maximal toléré de valeurs manquantes par feature.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]
        - X_train
        - y_train
        - X_test
        - quality_report
    """
    # Remplacement défensif des infinis par NaN
    train_clean = train_df.copy().replace([np.inf, -np.inf], np.nan)
    test_clean = test_df.copy().replace([np.inf, -np.inf], np.nan)

    # Séparation features / target
    X_train = train_clean.drop(columns=[id_col, target_col]).copy()
    y_train = train_clean[target_col].astype(int).copy()
    X_test = test_clean.drop(columns=[id_col]).copy()

    # Suppression des colonnes trop manquantes sur la base du train uniquement
    missing_ratio = X_train.isna().mean()
    columns_above_threshold = missing_ratio.loc[missing_ratio > missing_threshold].index.tolist()
    if columns_above_threshold:
        X_train = X_train.drop(columns=columns_above_threshold)
        X_test = X_test.drop(columns=columns_above_threshold, errors="ignore")

    # Suppression des colonnes constantes
    constant_columns = [
        column
        for column in X_train.columns
        if X_train[column].nunique(dropna=False) <= 1
    ]
    if constant_columns:
        X_train = X_train.drop(columns=constant_columns)
        X_test = X_test.drop(columns=constant_columns, errors="ignore")

    # Alignement strict : X_test reprend l'ordre et les colonnes de X_train
    X_train, X_test = X_train.align(X_test, join="left", axis=1)

    # Gestion des colonnes object restantes
    object_columns = X_train.select_dtypes(include=["object", "string"]).columns.tolist()
    dropped_non_numeric_columns: list[str] = []

    for column in object_columns:
        train_as_str = X_train[column].dropna().astype(str).str.strip().str.lower()
        test_as_str = X_test[column].dropna().astype(str).str.strip().str.lower()
        combined_values = set(train_as_str.unique()).union(set(test_as_str.unique()))

        # Cas simple : booléens encodés en texte
        if combined_values.issubset({"true", "false"}):
            bool_mapping = {"true": 1.0, "false": 0.0}
            X_train[column] = (
                X_train[column].astype("string").str.strip().str.lower().map(bool_mapping).astype(float)
            )
            X_test[column] = (
                X_test[column].astype("string").str.strip().str.lower().map(bool_mapping).astype(float)
            )
        else:
            # Toute autre colonne object est retirée par sécurité
            dropped_non_numeric_columns.append(column)

    if dropped_non_numeric_columns:
        X_train = X_train.drop(columns=dropped_non_numeric_columns)
        X_test = X_test.drop(columns=dropped_non_numeric_columns, errors="ignore")

    # Normalisation des noms de colonnes pour éviter les caractères problématiques
    sanitized_names = []
    seen_names: dict[str, int] = {}

    for column in X_train.columns:
        clean_name = re.sub(r"[^0-9a-zA-Z_]+", "_", str(column).strip().lower())
        clean_name = re.sub(r"_+", "_", clean_name).strip("_") or "feature"

        # Gestion des collisions après nettoyage
        if clean_name in seen_names:
            seen_names[clean_name] += 1
            clean_name = f"{clean_name}_{seen_names[clean_name]}"
        else:
            seen_names[clean_name] = 1

        sanitized_names.append(clean_name)

    X_train.columns = sanitized_names
    X_test.columns = sanitized_names

    # Rapport qualité des features conservées
    quality_report = pd.DataFrame(
        {
            "feature": X_train.columns,
            "missing_ratio_train": X_train.isna().mean().values,
            "dtype": X_train.dtypes.astype(str).values,
            "nunique_train": X_train.nunique(dropna=True).values,
        }
    ).sort_values(["missing_ratio_train", "nunique_train"], ascending=[False, True])

    # On ajoute à la fin les colonnes supprimées car non numériques
    if dropped_non_numeric_columns:
        dropped_report = pd.DataFrame(
            {
                "feature": dropped_non_numeric_columns,
                "missing_ratio_train": np.nan,
                "dtype": "dropped_non_numeric_object",
                "nunique_train": np.nan,
            }
        )
        quality_report = pd.concat([quality_report, dropped_report], ignore_index=True)

    return X_train, y_train, X_test, quality_report


def threshold_search(
    y_true,
    y_scores,
    thresholds: np.ndarray | None = None,
    fn_cost: float = 10.0,
    fp_cost: float = 1.0,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Teste plusieurs seuils de décision et retient le meilleur selon le coût métier.

    Parameters
    ----------
    y_true : array-like
        Vérité terrain.
    y_scores : array-like
        Scores continus.
    thresholds : np.ndarray | None, optional
        Liste de seuils à tester. Si None, utilise une grille par défaut.
    fn_cost : float, optional
        Coût d'un faux négatif.
    fp_cost : float, optional
        Coût d'un faux positif.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        - DataFrame des résultats pour tous les seuils
        - Ligne correspondant au meilleur seuil
    """
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.01)

    rows = [
        classification_metrics_at_threshold(
            y_true=y_true,
            y_scores=y_scores,
            threshold=float(threshold),
            fn_cost=fn_cost,
            fp_cost=fp_cost,
        )
        for threshold in thresholds
    ]

    results = pd.DataFrame(rows)

    # Priorité : coût métier minimal, puis PR AUC, puis recall
    best_row = results.sort_values(
        ["business_cost_per_obs", "pr_auc", "recall"],
        ascending=[True, False, False],
    ).iloc[0]

    return results, best_row


def plot_learning_curve_binary_classifier(
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    scoring: str = "average_precision",
    cv: int = 5,
    random_state: int = 42,
    train_sizes: np.ndarray | None = None,
    n_jobs: int = -1,
):
    """
    Trace une learning curve pour un classifieur binaire.

    Parameters
    ----------
    estimator : estimator
        Modèle scikit-learn compatible.
    X : pd.DataFrame
        Features.
    y : pd.Series
        Cible binaire.
    scoring : str, optional
        Métrique de scoring scikit-learn.
    cv : int, optional
        Nombre de folds.
    random_state : int, optional
        Graine aléatoire.
    train_sizes : np.ndarray | None, optional
        Tailles d'échantillon d'entraînement.
    n_jobs : int, optional
        Nombre de jobs parallèles.

    Returns
    -------
    tuple[pd.DataFrame, matplotlib.figure.Figure]
        - DataFrame résumé de la learning curve
        - Figure matplotlib
    """
    if train_sizes is None:
        train_sizes = np.linspace(0.2, 1.0, 5)

    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    train_sizes_abs, train_scores, valid_scores = learning_curve(
        estimator=estimator,
        X=X,
        y=y,
        cv=splitter,
        train_sizes=train_sizes,
        scoring=scoring,
        n_jobs=n_jobs,
    )

    learning_curve_df = pd.DataFrame(
        {
            "train_size": train_sizes_abs,
            "train_mean": train_scores.mean(axis=1),
            "train_std": train_scores.std(axis=1),
            "valid_mean": valid_scores.mean(axis=1),
            "valid_std": valid_scores.std(axis=1),
        }
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(learning_curve_df["train_size"], learning_curve_df["train_mean"], marker="o", label="Train")
    ax.plot(learning_curve_df["train_size"], learning_curve_df["valid_mean"], marker="o", label="Validation")
    ax.fill_between(
        learning_curve_df["train_size"],
        learning_curve_df["train_mean"] - learning_curve_df["train_std"],
        learning_curve_df["train_mean"] + learning_curve_df["train_std"],
        alpha=0.15,
    )
    ax.fill_between(
        learning_curve_df["train_size"],
        learning_curve_df["valid_mean"] - learning_curve_df["valid_std"],
        learning_curve_df["valid_mean"] + learning_curve_df["valid_std"],
        alpha=0.15,
    )
    ax.set_xlabel("Nombre d'observations d'entrainement")
    ax.set_ylabel(scoring)
    ax.set_title(f"Learning curve - {scoring}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return learning_curve_df, fig


def plot_threshold_diagnostics(threshold_df: pd.DataFrame):
    """
    Affiche deux graphiques pour aider au choix du seuil :
    - coût métier moyen,
    - precision / recall / F1.

    Parameters
    ----------
    threshold_df : pd.DataFrame
        Résultats d'une recherche de seuils.

    Returns
    -------
    matplotlib.figure.Figure
        Figure matplotlib contenant les deux sous-graphiques.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    sns.lineplot(
        data=threshold_df,
        x="threshold",
        y="business_cost_per_obs",
        ax=axes[0],
        color="tab:red",
    )
    axes[0].set_title("Cout metier moyen selon le seuil")
    axes[0].set_xlabel("Seuil")
    axes[0].set_ylabel("Cout moyen")
    axes[0].grid(True, alpha=0.3)

    sns.lineplot(data=threshold_df, x="threshold", y="recall", ax=axes[1], label="Recall")
    sns.lineplot(data=threshold_df, x="threshold", y="precision", ax=axes[1], label="Precision")
    sns.lineplot(data=threshold_df, x="threshold", y="f1", ax=axes[1], label="F1")
    axes[1].set_title("Precision / Recall / F1 selon le seuil")
    axes[1].set_xlabel("Seuil")
    axes[1].set_ylabel("Score")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def save_figure(fig, output_path: Path | str, dpi: int = 150) -> None:
    """
    Sauvegarde une figure matplotlib en créant le dossier parent si besoin.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure à sauvegarder.
    output_path : Path | str
        Chemin de sortie.
    dpi : int, optional
        Résolution de sauvegarde.
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=dpi, bbox_inches="tight")
