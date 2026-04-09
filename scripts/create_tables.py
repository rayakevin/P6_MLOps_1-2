import os
from pathlib import Path

import pandas as pd
import psycopg2
from psycopg2 import sql
from dotenv import load_dotenv


load_dotenv()


# Parametres de connexion a PostgreSQL.
# Les valeurs sont lues depuis le fichier .env pour eviter de hardcoder
# les identifiants dans le code.
DB_CONFIG = {
    "host": os.getenv("PGHOST"),
    "port": int(os.getenv("PGPORT", 5432)),
    "user": os.getenv("PGUSER"),
    "password": os.getenv("PGPASSWORD"),
    "dbname": os.getenv("PGDATABASE", "home_credit"),
}

# Le script est stocke dans scripts/.
# On remonte a la racine du projet pour retrouver data/raw/.
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"

# Nombre de lignes utilisees pour inferer les types de colonnes.
# On echantillonne au lieu de lire tout le CSV pour garder un script rapide.
INFERENCE_ROWS = int(os.getenv("INFERENCE_ROWS", 5000))

# Association table SQL -> fichier CSV source.
# Ici se trouve la partie specifique au dataset Home Credit.
DATASETS = {
    "application_train": "application_train.csv",
    "application_test": "application_test.csv",
    "bureau": "bureau.csv",
    "bureau_balance": "bureau_balance.csv",
    "previous_application": "previous_application.csv",
    "pos_cash_balance": "POS_CASH_balance.csv",
    "installments_payments": "installments_payments.csv",
    "credit_card_balance": "credit_card_balance.csv",
}

# Index utiles pour accelerer les recherches et jointures futures.
# On indexe surtout les identifiants techniques et la cible.
INDEXES = {
    "application_train": ["SK_ID_CURR", "TARGET"],
    "application_test": ["SK_ID_CURR"],
    "bureau": ["SK_ID_BUREAU", "SK_ID_CURR"],
    "bureau_balance": ["SK_ID_BUREAU", "MONTHS_BALANCE"],
    "previous_application": ["SK_ID_PREV", "SK_ID_CURR"],
    "pos_cash_balance": ["SK_ID_PREV", "SK_ID_CURR", "MONTHS_BALANCE"],
    "installments_payments": ["SK_ID_PREV", "SK_ID_CURR"],
    "credit_card_balance": ["SK_ID_PREV", "SK_ID_CURR", "MONTHS_BALANCE"],
}

# Clefs primaires que l'on considere fiables pour ce schema.
# Elles sont separees de la logique de creation afin de garder le script declaratif.
PRIMARY_KEYS = {
    "application_train": ["SK_ID_CURR"],
    "application_test": ["SK_ID_CURR"],
    "bureau": ["SK_ID_BUREAU"],
    "previous_application": ["SK_ID_PREV"],
}

# Definition des relations entre tables.
# On ne met pas de FK sur SK_ID_CURR vers application_train/application_test
# car le parent logique est partage entre deux tables differentes.
#
# Important : dans ce projet, ces relations servent surtout de documentation
# du schema. Le dataset raw n'est pas parfaitement coherent sur toutes les
# references, donc on n'active pas les foreign keys SQL par defaut.
FOREIGN_KEYS = [
    {
        "table": "bureau_balance",
        "columns": ["SK_ID_BUREAU"],
        "reference_table": "bureau",
        "reference_columns": ["SK_ID_BUREAU"],
    },
    {
        "table": "pos_cash_balance",
        "columns": ["SK_ID_PREV"],
        "reference_table": "previous_application",
        "reference_columns": ["SK_ID_PREV"],
    },
    {
        "table": "installments_payments",
        "columns": ["SK_ID_PREV"],
        "reference_table": "previous_application",
        "reference_columns": ["SK_ID_PREV"],
    },
    {
        "table": "credit_card_balance",
        "columns": ["SK_ID_PREV"],
        "reference_table": "previous_application",
        "reference_columns": ["SK_ID_PREV"],
    },
]

# Active eventuellement la creation physique des foreign keys dans PostgreSQL.
# Par defaut on laisse cette option a False pour pouvoir charger la couche raw
# sans etre bloque par les incoherences referentielles du dataset.
ENABLE_FOREIGN_KEYS = os.getenv("ENABLE_FOREIGN_KEYS", "false").lower() == "true"


def infer_postgres_columns(csv_path: Path) -> list[tuple[str, str]]:
    # Lecture d'un echantillon du CSV pour recuperer les noms de colonnes
    # et les types pandas detectes automatiquement.
    sample = pd.read_csv(csv_path, nrows=INFERENCE_ROWS, low_memory=False)
    columns = []

    for column_name, dtype in sample.dtypes.items():
        # Chaque type pandas est converti vers un type PostgreSQL simple.
        pg_type = map_dtype(dtype)
        columns.append((column_name, pg_type))

    return columns


def map_dtype(dtype) -> str:
    # Mapping volontairement simple :
    # BIGINT pour les entiers,
    # DOUBLE PRECISION pour les colonnes numeriques avec decimales,
    # BOOLEAN pour les booleens,
    # TIMESTAMP pour les dates si pandas en detecte,
    # TEXT sinon.
    #
    # Pour une couche raw, on cherche surtout la robustesse plutot
    # qu'une modelisation SQL tres fine.
    if pd.api.types.is_integer_dtype(dtype):
        return "BIGINT"
    if pd.api.types.is_float_dtype(dtype):
        return "DOUBLE PRECISION"
    if pd.api.types.is_bool_dtype(dtype):
        return "BOOLEAN"
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "TIMESTAMP"
    return "TEXT"


def create_table(cur, table_name: str, columns: list[tuple[str, str]]) -> None:
    # Construction dynamique de la liste des colonnes SQL :
    # "nom_colonne TYPE".
    # sql.Identifier protege les noms de colonnes et de tables.
    column_defs = [
        sql.SQL("{} {}").format(sql.Identifier(name), sql.SQL(pg_type))
        for name, pg_type in columns
    ]

    # CREATE TABLE IF NOT EXISTS permet de relancer le script sans erreur
    # si la table existe deja.
    statement = sql.SQL("CREATE TABLE IF NOT EXISTS {} ({})").format(
        sql.Identifier(table_name),
        sql.SQL(", ").join(column_defs),
    )
    cur.execute(statement)


def create_index(cur, table_name: str, column_name: str) -> None:
    # Convention simple de nommage des index pour rester predictible.
    index_name = f"idx_{table_name}_{column_name.lower()}"

    # Les index sont crees separement des tables pour garder la logique claire.
    statement = sql.SQL(
        "CREATE INDEX IF NOT EXISTS {} ON {} ({})"
    ).format(
        sql.Identifier(index_name),
        sql.Identifier(table_name),
        sql.Identifier(column_name),
    )
    cur.execute(statement)


def build_constraint_name(prefix: str, table_name: str, columns: list[str]) -> str:
    # Genere un nom deterministe pour les PK/FK.
    # Exemple : pk_bureau_sk_id_bureau
    return f"{prefix}_{table_name}_{'_'.join(column.lower() for column in columns)}"


def constraint_exists(cur, constraint_name: str) -> bool:
    # On interroge le catalogue systeme PostgreSQL pour verifier si la
    # contrainte existe deja avant de faire un ALTER TABLE.
    cur.execute(
        """
        SELECT 1
        FROM pg_constraint
        WHERE conname = %s
        """,
        (constraint_name,),
    )
    return cur.fetchone() is not None


def add_primary_key(cur, table_name: str, columns: list[str]) -> None:
    # On derive le nom de contrainte a partir du nom de table et des colonnes.
    constraint_name = build_constraint_name("pk", table_name, columns)

    # Si la contrainte existe deja, on ne fait rien.
    if constraint_exists(cur, constraint_name):
        return

    # Ajout de la cle primaire sur une ou plusieurs colonnes.
    cur.execute(
        sql.SQL("ALTER TABLE {} ADD CONSTRAINT {} PRIMARY KEY ({})").format(
            sql.Identifier(table_name),
            sql.Identifier(constraint_name),
            sql.SQL(", ").join(sql.Identifier(column) for column in columns),
        )
    )


def add_foreign_key(
    cur,
    table_name: str,
    columns: list[str],
    reference_table: str,
    reference_columns: list[str],
) -> None:
    # Meme logique que pour les PK, mais appliquee aux foreign keys.
    # Cette fonction n'est appelee que si ENABLE_FOREIGN_KEYS=true.
    constraint_name = build_constraint_name("fk", table_name, columns)
    if constraint_exists(cur, constraint_name):
        return

    # PostgreSQL verifiera ensuite que les valeurs de la table enfant
    # correspondent bien a des valeurs existantes dans la table parent.
    # Sur ce projet, ce controle est utile surtout dans une couche preparee
    # ou nettoyee, pas dans la couche raw importee telle quelle.
    cur.execute(
        sql.SQL(
            "ALTER TABLE {} ADD CONSTRAINT {} FOREIGN KEY ({}) REFERENCES {} ({})"
        ).format(
            sql.Identifier(table_name),
            sql.Identifier(constraint_name),
            sql.SQL(", ").join(sql.Identifier(column) for column in columns),
            sql.Identifier(reference_table),
            sql.SQL(", ").join(sql.Identifier(column) for column in reference_columns),
        )
    )


def validate_dataset_files() -> None:
    # Verification defensive : on arrete le script tout de suite si un CSV
    # attendu est absent du dossier data/raw.
    missing_files = [filename for filename in DATASETS.values() if not (DATA_DIR / filename).exists()]
    if missing_files:
        missing = ", ".join(missing_files)
        raise FileNotFoundError(f"Fichiers manquants dans {DATA_DIR}: {missing}")


def main() -> None:
    # On verifie d'abord que tous les fichiers necessaires sont presents.
    validate_dataset_files()

    # Connexion a la base cible deja creee par create_db.py.
    conn = psycopg2.connect(**DB_CONFIG)

    # On desactive l'autocommit pour garder une transaction maitrisee :
    # si une etape echoue, on rollback tout.
    conn.autocommit = False

    try:
        with conn.cursor() as cur:
            # 1. Creation des tables a partir des CSV
            for table_name, filename in DATASETS.items():
                csv_path = DATA_DIR / filename
                columns = infer_postgres_columns(csv_path)
                create_table(cur, table_name, columns)

                # 2. Creation des index definis dans la configuration
                for column_name in INDEXES.get(table_name, []):
                    create_index(cur, table_name, column_name)

            # 3. Ajout des cles primaires
            for table_name, columns in PRIMARY_KEYS.items():
                add_primary_key(cur, table_name, columns)

            # 4. Ajout optionnel des cles etrangeres
            # Desactive par defaut pour permettre le chargement du raw dataset
            # meme si certaines references sont incoherentes.
            # Si l'on veut un schema plus strict apres nettoyage des donnees,
            # on peut relancer ce script avec ENABLE_FOREIGN_KEYS=true.
            if ENABLE_FOREIGN_KEYS:
                for foreign_key in FOREIGN_KEYS:
                    add_foreign_key(
                        cur,
                        foreign_key["table"],
                        foreign_key["columns"],
                        foreign_key["reference_table"],
                        foreign_key["reference_columns"],
                    )

        # Si tout s'est bien passe, on valide la transaction.
        conn.commit()
        print("Tables PostgreSQL creees pour les datasets Home Credit.")
    except Exception:
        # Si une erreur survient a n'importe quelle etape,
        # on annule toutes les modifications SQL faites dans cette execution.
        conn.rollback()
        raise
    finally:
        # Fermeture propre de la connexion, qu'il y ait succes ou erreur.
        conn.close()


if __name__ == "__main__":
    main()
