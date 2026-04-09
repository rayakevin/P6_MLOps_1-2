import os

import psycopg2
from psycopg2 import sql
from dotenv import load_dotenv


load_dotenv()


# Parametres minimaux necessaires pour se connecter au serveur PostgreSQL.
# On ne renseigne pas ici "dbname" car ce script doit d'abord se connecter
# a la base d'administration "postgres" pour pouvoir creer une nouvelle base.
DB_CONFIG = {
    "host": os.getenv("PGHOST"),
    "port": int(os.getenv("PGPORT", 5432)),
    "user": os.getenv("PGUSER"),
    "password": os.getenv("PGPASSWORD"),
}

# Nom de la base applicative a creer.
DB_NAME = os.getenv("PGDATABASE", "home_credit")


def database_exists(cur, db_name: str) -> bool:
    """
    Verifie si une base de donnees existe deja sur le serveur PostgreSQL.

    Parametres
    ----------
    cur : psycopg2.extensions.cursor
        Curseur PostgreSQL deja ouvert sur la base d'administration.
    db_name : str
        Nom de la base que l'on souhaite verifier.

    Retour
    ------
    bool
        True si la base existe deja, False sinon.

    Pourquoi cette fonction existe
    ------------------------------
    PostgreSQL renverrait une erreur si on essayait de creer une base qui
    existe deja. On fait donc d'abord un controle sur le catalogue systeme
    "pg_database" pour rendre le script rejouable sans echec inutile.
    """
    cur.execute(
        "SELECT 1 FROM pg_database WHERE datname = %s",
        (db_name,),
    )
    return cur.fetchone() is not None


def create_database(cur, db_name: str) -> None:
    """
    Cree une nouvelle base PostgreSQL.

    Parametres
    ----------
    cur : psycopg2.extensions.cursor
        Curseur PostgreSQL ouvert avec des droits suffisants pour creer
        une base de donnees.
    db_name : str
        Nom de la base a creer.

    Pourquoi sql.Identifier est utilise
    -----------------------------------
    Le nom de la base est un identifiant SQL, pas une valeur classique.
    On ne peut donc pas utiliser un placeholder %s comme pour un WHERE.
    psycopg2.sql.Identifier permet de construire la requete correctement
    et d'eviter les erreurs de syntaxe ou les injections SQL.
    """
    cur.execute(
        sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db_name))
    )


def main() -> None:
    """
    Point d'entree du script.

    Deroulement
    -----------
    1. Connexion a la base systeme "postgres".
    2. Verification de l'existence de la base cible.
    3. Creation de la base si elle n'existe pas encore.
    4. Fermeture propre de la connexion.

    Pourquoi se connecter a "postgres"
    ----------------------------------
    Pour creer une base, on ne se connecte generalement pas a la base cible
    elle-meme puisqu'elle n'existe pas encore. On se connecte a une base
    d'administration deja presente sur le serveur, souvent "postgres".

    Pourquoi autocommit = True
    --------------------------
    La commande CREATE DATABASE ne peut pas etre executee a l'interieur d'une
    transaction SQL classique. On active donc l'autocommit pour que PostgreSQL
    execute l'ordre immediatement.
    """
    conn = psycopg2.connect(dbname="postgres", **DB_CONFIG)
    conn.autocommit = True

    try:
        with conn.cursor() as cur:
            if database_exists(cur, DB_NAME):
                print(f"Base '{DB_NAME}' existe deja.")
                return

            create_database(cur, DB_NAME)
            print(f"Base '{DB_NAME}' creee.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
