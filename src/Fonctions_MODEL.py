from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from time import perf_counter

from sklearn.model_selection import (
    train_test_split,
    KFold,
    StratifiedKFold,
    GridSearchCV,
    RandomizedSearchCV,
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
)
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from category_encoders import BinaryEncoder


def encode_cat_col(
    df: pd.DataFrame,
    col_name: str,
    encoding_type: str,
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
        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )

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
