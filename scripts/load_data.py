import os
from pathlib import Path

import psycopg2
from psycopg2 import sql
from dotenv import load_dotenv


load_dotenv()


# Parametres de connexion a la base cible.
# Contrairement a create_db.py, ici on se connecte directement a la base
# applicative deja creee.
DB_CONFIG = {
    "host": os.getenv("PGHOST"),
    "port": int(os.getenv("PGPORT", 5432)),
    "user": os.getenv("PGUSER"),
    "password": os.getenv("PGPASSWORD"),
    "dbname": os.getenv("PGDATABASE", "home_credit"),
}

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"

# Ordre de chargement choisi pour respecter autant que possible les
# dependances logiques entre tables.
# Meme si les FK SQL sont desactivees dans la couche raw, il reste plus
# coherent de charger d'abord les tables "parents", puis les historiques.
DATASETS = [
    ("application_train", "application_train.csv"),
    ("application_test", "application_test.csv"),
    ("bureau", "bureau.csv"),
    ("bureau_balance", "bureau_balance.csv"),
    ("previous_application", "previous_application.csv"),
    ("pos_cash_balance", "POS_CASH_balance.csv"),
    ("installments_payments", "installments_payments.csv"),
    ("credit_card_balance", "credit_card_balance.csv"),
]

# Liste explicite des foreign keys potentiellement deja presentes dans la base.
# On les retire avant chargement car le dataset raw contient certaines
# incoherences referentielles.
FOREIGN_KEY_CONSTRAINTS = [
    ("bureau_balance", "fk_bureau_balance_sk_id_bureau"),
    ("pos_cash_balance", "fk_pos_cash_balance_sk_id_prev"),
    ("installments_payments", "fk_installments_payments_sk_id_prev"),
    ("credit_card_balance", "fk_credit_card_balance_sk_id_prev"),
]


def validate_dataset_files() -> None:
    """
    Verifie que tous les fichiers CSV attendus sont presents.

    Pourquoi cette verification est utile
    -------------------------------------
    Sans ce controle, le script pourrait commencer a charger certaines tables
    puis s'arreter plus tard sur un fichier manquant. On prefere echouer tout
    de suite avec un message clair avant meme d'ouvrir la transaction SQL.
    """
    missing_files = [filename for _, filename in DATASETS if not (DATA_DIR / filename).exists()]
    if missing_files:
        missing = ", ".join(missing_files)
        raise FileNotFoundError(f"Fichiers manquants dans {DATA_DIR}: {missing}")


def table_has_rows(cur, table_name: str) -> bool:
    """
    Indique si une table contient deja au moins une ligne.

    Parametres
    ----------
    cur : psycopg2.extensions.cursor
        Curseur PostgreSQL ouvert sur la base cible.
    table_name : str
        Nom de la table a verifier.

    Retour
    ------
    bool
        True si la table n'est pas vide, False sinon.

    Pourquoi on utilise EXISTS
    --------------------------
    On veut un test leger. Un SELECT COUNT(*) parcourrait potentiellement
    beaucoup plus de donnees. Avec EXISTS (...) LIMIT 1, PostgreSQL peut
    s'arreter des qu'une ligne est trouvee.
    """
    cur.execute(
        sql.SQL("SELECT EXISTS (SELECT 1 FROM {} LIMIT 1)").format(
            sql.Identifier(table_name)
        )
    )
    return cur.fetchone()[0]


def drop_constraint_if_exists(cur, table_name: str, constraint_name: str) -> None:
    """
    Supprime une contrainte si elle existe deja sur une table.

    Parametres
    ----------
    cur : psycopg2.extensions.cursor
        Curseur PostgreSQL ouvert.
    table_name : str
        Table sur laquelle la contrainte est definie.
    constraint_name : str
        Nom exact de la contrainte a supprimer.

    Pourquoi cette fonction existe
    ------------------------------
    Si la base a ete creee avec une version precedente du script qui ajoutait
    des foreign keys, ces contraintes bloquent ensuite le chargement du raw
    dataset. Cette fonction permet de nettoyer cet etat avant import.
    """
    cur.execute(
        sql.SQL("ALTER TABLE {} DROP CONSTRAINT IF EXISTS {}").format(
            sql.Identifier(table_name),
            sql.Identifier(constraint_name),
        )
    )


def drop_raw_foreign_keys(cur) -> None:
    """
    Retire les foreign keys incompatibles avec la couche raw du dataset.

    Pourquoi on fait cela
    ---------------------
    Le dataset Home Credit n'est pas parfaitement coherent du point de vue
    referentiel : certaines lignes enfants pointent vers des identifiants
    absents de la table parent. En raw zone, l'objectif principal est de
    charger les fichiers tels quels, pas de forcer une integrite parfaite.
    """
    # Le dataset Home Credit contient des references orphelines dans la couche raw.
    # On retire donc ces FK si elles existent deja pour ne pas bloquer le chargement.
    for table_name, constraint_name in FOREIGN_KEY_CONSTRAINTS:
        drop_constraint_if_exists(cur, table_name, constraint_name)


def load_csv_into_table(cur, table_name: str, csv_path: Path) -> None:
    """
    Charge un fichier CSV dans une table PostgreSQL via COPY.

    Parametres
    ----------
    cur : psycopg2.extensions.cursor
        Curseur PostgreSQL ouvert.
    table_name : str
        Nom de la table cible.
    csv_path : Path
        Chemin vers le fichier CSV source.

    Pourquoi utiliser COPY
    ----------------------
    COPY est la methode native PostgreSQL pour importer rapidement de gros
    volumes de donnees. Elle est beaucoup plus efficace que des INSERT ligne
    par ligne, ce qui est important ici vu la taille du dataset Home Credit.

    Explication de la requete
    -------------------------
    - FORMAT CSV : indique a PostgreSQL que la source est un CSV.
    - HEADER TRUE : la premiere ligne contient les noms de colonnes.
    - DELIMITER ',' : les champs sont separes par des virgules.
    - QUOTE '"' : les champs textes eventuellement quotes utilisent ".

    Pourquoi copy_expert
    --------------------
    psycopg2 expose copy_expert pour executer une commande COPY complete,
    construite dynamiquement. Cela nous laisse plus de controle qu'une
    methode simplifiee.
    """
    copy_query = sql.SQL(
        """
        COPY {} FROM STDIN
        WITH (
            FORMAT CSV,
            HEADER TRUE,
            DELIMITER ',',
            QUOTE '"'
        )
        """
    ).format(sql.Identifier(table_name))

    # Le fichier est ouvert cote Python, puis son contenu est transmis
    # directement a PostgreSQL via STDIN.
    with csv_path.open("r", encoding="utf-8") as csv_file:
        cur.copy_expert(copy_query.as_string(cur.connection), csv_file)


def main() -> None:
    """
    Point d'entree du script de chargement des donnees.

    Deroulement
    -----------
    1. Verification de la presence de tous les CSV.
    2. Connexion a la base cible.
    3. Suppression preventive des foreign keys raw si elles existent.
    4. Parcours des tables source une par une.
    5. Chargement de chaque CSV via COPY.
    6. Commit apres chaque table pour sauvegarder la progression.

    Pourquoi commit table par table
    -------------------------------
    Le dataset est volumineux. Si on attendait la fin complete pour faire un
    seul commit, une erreur tardive ou un arret du processus ferait perdre
    tout le travail deja effectue. En validant apres chaque table, on rend
    le chargement plus robuste et plus facile a reprendre.

    Pourquoi on saute une table deja remplie
    ----------------------------------------
    Cela permet de relancer le script sans dupliquer les donnees des tables
    deja chargees. C'est une logique simple de "resume" adaptee a ce projet.
    """
    validate_dataset_files()

    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = False

    try:
        with conn.cursor() as cur:
            # Nettoyage initial des FK incompatibles avec le raw dataset.
            drop_raw_foreign_keys(cur)

            # On valide ce nettoyage tout de suite pour repartir d'un etat
            # stable avant les imports volumineux.
            conn.commit()

            for table_name, filename in DATASETS:
                csv_path = DATA_DIR / filename

                # Si la table contient deja des lignes, on suppose qu'elle a
                # deja ete chargee lors d'une execution precedente.
                if table_has_rows(cur, table_name):
                    print(f"Table '{table_name}' deja remplie, chargement ignore.", flush=True)
                    continue

                print(f"Chargement de '{filename}' dans '{table_name}'...", flush=True)
                load_csv_into_table(cur, table_name, csv_path)

                # Chaque table est validee separement pour ne pas perdre
                # tout l'avancement en cas d'erreur sur une table suivante.
                conn.commit()
                print(f"Chargement de '{table_name}' valide.", flush=True)

        print("Chargement des donnees termine.", flush=True)
    except Exception:
        # Si une erreur survient pendant le chargement d'une table, on annule
        # uniquement la transaction en cours. Les tables deja commit sont
        # preservees.
        conn.rollback()
        raise
    finally:
        # Fermeture propre de la connexion, que le chargement reussisse ou non.
        conn.close()


if __name__ == "__main__":
    main()
