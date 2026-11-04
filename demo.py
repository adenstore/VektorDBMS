# =====================================================================
#  Vektordatabase med Milvus Lite: opsætning og CRUD
# =====================================================================
#
#  SÅDAN KØRER DU FILEN (Linux, macOS eller WSL på Windows)
#    1. python3 -m venv venv && source venv/bin/activate
#    2. pip install -r requirements.txt
#    3. python demo.py            (kører det hele)
#       python demo.py --pause    (stopper efter hvert trin, godt til live-demo)
#
#  Databasen gemmes lokalt som film.db i samme mappe. Den slettes og oprettes
#  igen, hver gang scriptet kører, så demoen altid starter forfra.
#
#  Om vektorerne: i en rigtig app laver en AI-model (embedding-model) tekst
#  eller billeder om til vektorer. Her bruger vi tilfældige tal, fordi fokus
#  er på, hvordan man skriver til og læser fra databasen.
# =====================================================================

import os
import random
import shutil
import sys

from pymilvus import DataType, MilvusClient

from data import FILM

DB_FIL = "film.db"
COLLECTION = "film"
DIM = 8  # antal tal i hver vektor (rigtige modeller bruger fx 384 eller 1536)
PAUSE = "--pause" in sys.argv
random.seed(42)  # samme "tilfældige" vektorer hver gang


def trin(titel):
    if PAUSE:
        input("\n[Tryk Enter for at fortsætte]")
    print("\n" + "=" * 60 + f"\n  {titel}\n" + "=" * 60)


def vektor():
    return [round(random.random(), 3) for _ in range(DIM)]


def antal(res, noegle):
    """Svaret er en dict eller en liste af id'er afhængigt af pymilvus-version."""
    return res[noegle] if isinstance(res, dict) else len(res)


def vis_alle():
    rækker = client.query(COLLECTION, filter="id >= 0",
                          output_fields=["titel", "genre", "aar"])
    for r in sorted(rækker, key=lambda r: r["id"]):
        print(f"  {r['id']:>2}  {r['titel']:<18} {r['genre']:<8} {r['aar']}")


# ---------------------------------------------------------------------
# 1. OPSÆTNING
# ---------------------------------------------------------------------
trin("1. Opsætning: opret database og collection")

# Nyere Milvus Lite gemmer databasen som en mappe, ældre som én fil
if os.path.isdir(DB_FIL):
    shutil.rmtree(DB_FIL)
elif os.path.exists(DB_FIL):
    os.remove(DB_FIL)

# Milvus Lite: én linje, ingen server. Med en URL som "http://localhost:19530"
# taler præcis samme kode med en Milvus Standalone-server i stedet.
client = MilvusClient(DB_FIL)

# Et schema svarer til en tabeldefinition. Den særlige kolonne er vektoren.
schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
schema.add_field("id", DataType.INT64, is_primary=True)
schema.add_field("titel", DataType.VARCHAR, max_length=200)
schema.add_field("genre", DataType.VARCHAR, max_length=50)
schema.add_field("aar", DataType.INT64)
schema.add_field("vektor", DataType.FLOAT_VECTOR, dim=DIM)

# Et indeks på vektoren er påkrævet. COSINE er afstandsmålet.
# FLAT sammenligner med alle rækker. Det er præcist og fint til små mængder.
index = client.prepare_index_params()
index.add_index(field_name="vektor", index_type="FLAT", metric_type="COSINE")

client.create_collection(COLLECTION, schema=schema, index_params=index)
print(f"Collections i databasen: {client.list_collections()}")
print(client.describe_collection(COLLECTION)["fields"][-1])


# ---------------------------------------------------------------------
# 2. CREATE
# ---------------------------------------------------------------------
trin("2. Create: insert")

rækker = [{**f, "vektor": vektor()} for f in FILM]
res = client.insert(COLLECTION, rækker)
print(f"Indsat: {res['insert_count']} rækker")
print(f"Én række ser sådan ud: {rækker[0]}")

# Én række mere
client.insert(COLLECTION, [{"id": 9, "titel": "The Matrix", "genre": "sci-fi",
                            "aar": 1999, "vektor": vektor()}])
vis_alle()


# ---------------------------------------------------------------------
# 3. READ
# ---------------------------------------------------------------------
trin("3a. Read: get på id")
for r in client.get(COLLECTION, ids=[2], output_fields=["titel", "genre", "aar"]):
    print(" ", r)

trin("3b. Read: query med filter (som WHERE i SQL)")
for r in client.query(COLLECTION, filter='genre == "komedie" and aar > 1980',
                      output_fields=["titel", "aar"]):
    print(" ", r)

trin("3c. Read: search (find de nærmeste vektorer)")
# Vi søger med vektoren fra Interstellar. Den nærmeste er sig selv (afstand 1.0).
soeg = client.get(COLLECTION, ids=[1], output_fields=["vektor"])[0]["vektor"]
hits = client.search(COLLECTION, data=[soeg], anns_field="vektor", limit=3,
                     output_fields=["titel"])[0]
for h in hits:
    print(f"  {h['distance']:.3f}  {h['entity']['titel']}")


# ---------------------------------------------------------------------
# 4. UPDATE (upsert)
# ---------------------------------------------------------------------
trin("4. Update: upsert")

# Milvus kan ikke opdatere enkelte felter. Man sender hele rækken igen med
# samme id, og den gamle bliver erstattet (upsert = update eller insert).
res = client.upsert(COLLECTION, [{"id": 5, "titel": "Olsen-banden",
                                  "genre": "krimi", "aar": 1968, "vektor": vektor()}])
print(f"Upsert: {antal(res, 'upsert_count')} række")
for r in client.get(COLLECTION, ids=[5], output_fields=["titel", "genre"]):
    print(" ", r)


# ---------------------------------------------------------------------
# 5. DELETE
# ---------------------------------------------------------------------
trin("5. Delete: på id og med filter")

res = client.delete(COLLECTION, ids=[9])
print(f"Slettet på id: {antal(res, 'delete_count')}")
res = client.delete(COLLECTION, filter='genre == "drama"')
print(f"Slettet med filter (genre == drama): {antal(res, 'delete_count')}")
vis_alle()


# ---------------------------------------------------------------------
# 6. OPRYDNING
# ---------------------------------------------------------------------
trin("6. Oprydning")
client.drop_collection(COLLECTION)
print(f"Collections efter drop: {client.list_collections()}")
client.close()
