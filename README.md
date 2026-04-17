# P6_MLOps_1-2

Pipeline pedagogique de scoring du risque de defaut Home Credit, structure autour de trois notebooks metier et d'un suivi MLflow local.

## Structure du projet

- `notebooks/01_PREPARATION_DONNEES.ipynb`
  Preparation, nettoyage, agregations et feature engineering.
- `notebooks/02_MODELISATION_BASELINES_MLFLOW.ipynb`
  Benchmark baseline, validation croisee et tracking MLflow.
- `notebooks/03_OPTIMISATION_SEUIL_EXPLICABILITE.ipynb`
  Optimisation du modele retenu, seuil metier et explicabilite.
- `src/Fonctions_EDA.py`
  Fonctions reutilisables pour l'exploration, la qualite de donnees et la preparation.
- `src/Fonctions_MODEL.py`
  Fonctions reutilisables pour l'evaluation, le tuning, les seuils et MLflow.

## Environnement

- Python `3.12`
- dependances gerees via `uv`

## Suivi des experimentations

Le projet utilise MLflow en local avec :

- un benchmark baseline trace dans le notebook 02 ;
- un workflow d'optimisation et d'explicabilite trace dans le notebook 03 ;
- des captures UI et des rapports HTML produits pour la soutenance.

## Convention de commit

Format recommande :

`type: description courte`

Types utilises :

- `feat`
- `fix`
- `refactor`
- `docs`
- `test`
- `chore`
