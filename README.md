# Vektordatabaser med Milvus Lite

Gruppeopgave i faget Database for udviklere (UCL): undersøgelse og præsentation af en databasetype.
Kategori: **Vector DBMS**. Database: **Milvus Lite**.

Repoet indeholder dokumentation, en guide til at komme i gang og demo-koden fra præsentationen.

| Fil | Indhold |
|---|---|
| `demo.py` | Opsætning og CRUD mod Milvus Lite, trin for trin |
| `data.py` | Eksempeldata (film) |
| `requirements.txt` | Python-pakker |
| `Dockerfile`, `milvus-compose.yml` | Kør demoen i Docker uden at installere noget |

---

## 1. Hvad er en vektordatabase?

En vektordatabase gemmer data som **vektorer**, altså lister af tal som `[0.12, -0.80, 0.33, ...]`.
En AI-model (en *embedding-model*) laver tekst, billeder eller lyd om til vektorer, så ting med
samme betydning får vektorer, der ligger tæt på hinanden.

Databasen er bygget til ét spørgsmål: **"find de rækker, der ligner denne mest"**. Det hedder
*similarity search* eller *nearest neighbour search*. En almindelig database spørger "hvilke rækker
er præcis lig med X?", mens en vektordatabase spørger "hvilke rækker er tættest på X?".

Centrale begreber:

| Begreb | Forklaring |
|---|---|
| Collection | Svarer til en tabel |
| Schema | Felterne i en collection, fx `id`, `titel` og en vektorkolonne med fast antal dimensioner |
| Dimension | Antal tal i hver vektor. Bestemmes af modellen (fx 384, 768 eller 1536) |
| Afstandsmål | Hvordan "tæt på" måles: `COSINE` (vinkel), `L2` (afstand) eller `IP` (prikprodukt) |
| Indeks | Datastruktur der gør søgningen hurtig. `FLAT` sammenligner med alt, `IVF` og `HNSW` er tilnærmede (ANN) og meget hurtigere ved store mængder |
| Skalarfelter | Almindelige felter (tekst, tal), som man kan filtrere på sammen med vektorsøgningen |

## 2. Om Milvus og Milvus Lite

[Milvus](https://milvus.io) er en open source-vektordatabase (Apache 2.0) udviklet af Zilliz og en del
af LF AI & Data Foundation. Den findes i fire udgaver med **samme API**:

| Udgave | Hvordan | Til hvad |
|---|---|---|
| **Milvus Lite** | Python-bibliotek, data gemmes lokalt (`film.db`) | Læring, prototyper, tests, op til ca. en million vektorer |
| Milvus Standalone | Én Docker-container | Mindre produktionsløsninger |
| Milvus Distributed | Kubernetes-klynge | Store mængder (milliarder af vektorer) |
| Zilliz Cloud | Hosted service | Når man ikke vil drive den selv |

Pointen med Lite er, at man kan starte uden server og senere skifte til Standalone ved kun at ændre
én linje: `MilvusClient("film.db")` bliver til `MilvusClient("http://localhost:19530")`.

Begrænsninger i Milvus Lite:
- Kører kun på **Linux og macOS**. På Windows skal man bruge WSL eller Docker (Standalone).
- Lavet til små datamængder. Ældre versioner understøttede kun **FLAT-indeks** (eksakt søgning). Nyere versioner (vi har testet milvus-lite 3.2.2) accepterer også fx HNSW, men tuning af indekser hører hjemme i Standalone og Distributed.
- Ingen brugere, roller eller adgangskontrol (`create_user` giver "Method not implemented") og ingen replikering.

## 3. Kom i gang

### Anbefalet: kør demoen i Docker

Kræver kun [Docker Desktop](https://www.docker.com/products/docker-desktop/). Du skal ikke installere
Python eller Milvus selv, og det virker også på Windows, fordi containeren er Linux.

```bash
docker compose -f milvus-compose.yml build
docker compose -f milvus-compose.yml run --rm demo
```

Demoen stopper efter hvert trin, tryk Enter for at fortsætte. Uden pauser:

```bash
docker compose -f milvus-compose.yml run --rm demo python /app/demo.py
```

Databasen (`film.db`) bliver gemt i mappen `./data` på din egen maskine, så du kan se den bagefter.
Brug `run` og ikke `up`, ellers virker "Tryk Enter" ikke.

Hvad sker der? `Dockerfile` tager et Python-image, installerer `pymilvus` og `milvus-lite` og kopierer
`demo.py` og `data.py` ind. Milvus Lite kører inde i Python-processen i containeren, der er ingen
separat databaseserver.

### Uden Docker: Linux, macOS eller WSL

Milvus Lite kører kun på Linux og macOS. På Windows skal man først installere WSL
(PowerShell som administrator: `wsl --install -d Ubuntu`, genstart, og `sudo apt install -y python3-venv` i Ubuntu).

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python demo.py --pause
```

### Videre: Milvus Standalone (rigtig server) i Docker

Til produktion bruger man Milvus Standalone i stedet for Lite:

```bash
curl -L https://github.com/milvus-io/milvus/releases/latest/download/milvus-standalone-docker-compose.yml -o standalone-compose.yml
docker compose -f standalone-compose.yml up -d
```

Ret derefter `DB_FIL` i `demo.py` til `"http://localhost:19530"` og fjern linjerne, der sletter
databasen. Resten af koden er den samme.

## 4. CRUD i Milvus

Alle eksempler bruger `client = MilvusClient("film.db")`.

### Opsætning: schema og collection

```python
schema = client.create_schema(auto_id=False)
schema.add_field("id", DataType.INT64, is_primary=True)
schema.add_field("titel", DataType.VARCHAR, max_length=200)
schema.add_field("genre", DataType.VARCHAR, max_length=50)
schema.add_field("aar", DataType.INT64)
schema.add_field("vektor", DataType.FLOAT_VECTOR, dim=8)

index = client.prepare_index_params()
index.add_index(field_name="vektor", index_type="FLAT", metric_type="COSINE")

client.create_collection("film", schema=schema, index_params=index)
```

### Create

```python
client.insert("film", [
    {"id": 1, "titel": "Interstellar", "genre": "sci-fi", "aar": 2014,
     "vektor": [0.64, 0.03, 0.28, 0.22, 0.74, 0.68, 0.89, 0.09]},
])
```

### Read

```python
client.get("film", ids=[1])                                     # på primærnøgle
client.query("film", filter='genre == "komedie" and aar > 1980',
             output_fields=["titel", "aar"])                    # filter, som WHERE
client.search("film", data=[min_vektor], anns_field="vektor",
              limit=3, filter="aar < 2000",
              output_fields=["titel"])                          # nærmeste vektorer
```

`search` er det særlige ved en vektordatabase: den returnerer de rækker, hvis vektorer ligger
tættest på søgevektoren, sorteret efter afstand. Den kan kombineres med et almindeligt filter.

### Update

Milvus kan ikke opdatere et enkelt felt med `UPDATE ... SET`. Man bruger **upsert** og sender hele
rækken igen med samme id. Findes id'et, bliver rækken erstattet, ellers bliver den indsat.

```python
client.upsert("film", [{"id": 5, "titel": "Olsen-banden", "genre": "krimi",
                        "aar": 1968, "vektor": [...]}])
```

### Delete

```python
client.delete("film", ids=[9])
client.delete("film", filter='genre == "drama"')
client.drop_collection("film")    # slet hele collection
```

## 5. Fordele og ulemper

**Fordele**
- Lynhurtig similarity search, også med millioner af vektorer (med ANN-indeks i Standalone/Distributed).
- Søgning på **betydning** i stedet for præcise ord, fx "film om rummet" finder "Interstellar".
- Kombinerer vektorsøgning med filtre på almindelige felter.
- Milvus Lite er nem at starte med: `pip install`, ingen server, ingen konfiguration.
- Samme kode fra laptop til klynge. Open source og stort økosystem (LangChain, LlamaIndex).

**Ulemper**
- Skal have en embedding-model ved siden af. Kvaliteten af søgningen afhænger af modellen.
- Ingen joins, ingen rigtige transaktioner og begrænset opdatering (kun upsert af hele rækker).
- Fast dimension: skifter man model, skal alle vektorer laves igen.
- Vektorer fylder meget (1 million × 768 floats ≈ 3 GB) og ANN-indekser bruger meget RAM.
- ANN-søgning er tilnærmet: man bytter lidt præcision for hastighed.
- Milvus Lite: kun Linux/macOS, til små mængder, ingen sikkerhed. Ikke til produktion.

## 6. Refleksion: hvornår er det det rigtige valg?

**Godt valg**
- **RAG** og chatbots, der skal finde relevante tekststykker i egne dokumenter, før en LLM svarer.
- Semantisk søgning i fx produkter, artikler eller support-sager.
- Anbefalinger ("kunder der kunne lide X, kunne også lide Y").
- Billed- og lydsøgning, finde dubletter, opdage afvigelser.

**Vælg noget andet når**
- Data er struktureret med relationer, og man har brug for joins og transaktioner: **relationel database** (PostgreSQL, SQL Server).
- Man gemmer fleksible dokumenter og slår op på præcise værdier: **dokumentdatabase** (MongoDB).
- Man søger på præcise ord og vil have fuldtekstsøgning: **søgemaskine** (Elasticsearch).
- Man har allerede PostgreSQL og få vektorer: udvidelsen **pgvector** kan være nok, så man slipper for en ekstra database.

Typisk bruges en vektordatabase **sammen med** en anden database: den almindelige database er
"sandheden" om data, og vektordatabasen bruges til søgning.

## 7. Yderligere emner

**Skalerbarhed.** Milvus Distributed adskiller lagring og beregning: query nodes, data nodes og
index nodes kan skaleres hver for sig i Kubernetes. Data deles i *shards* og *segments*.
Milvus Lite kan ikke skaleres. Den er én proces på én maskine.

**Performance.** Valget af indeks er den største faktor. `FLAT` er 100 % præcis men langsom ved
mange rækker. Det er den, demoen bruger. `IVF_FLAT` deler vektorerne op i klynger og søger kun i de nærmeste. `HNSW` bygger
en graf og er typisk hurtigst, men bruger mest RAM. Man justerer parametre (fx `nprobe`, `ef`) for
at vælge mellem hastighed og præcision (recall).

**Store datamængder.** Vektorer kan komprimeres med *quantization* (fx `IVF_PQ`, `SQ8`), så de
fylder op til 4 til 32 gange mindre mod lidt lavere præcision. Distributed gemmer data i
objektlager (MinIO/S3).

**Sikkerhed.** Milvus Standalone/Distributed har brugere, roller (RBAC), adgangskode og TLS.
Milvus Lite har ingen af delene: alle, der kan læse filen, kan læse data.

**Data-redundans.** Distributed kan have flere *replicas* i hukommelsen for høj tilgængelighed og
bruger en log (Pulsar/Kafka/Woodpecker) og objektlager til holdbarhed. I Milvus Lite er
backup bare at kopiere `film.db`.

## 8. Eksamensspørgsmål

> *Search engines, wide column stores og vector databaser er hver især bygget til en bestemt slags
> data og en bestemt måde at søge på. Hvad kendetegner de tre typer, og hvordan adskiller de sig fra
> hinanden og fra en relationel database?

### Hvad kendetegner de tre typer?

**Search engine** (fx Elasticsearch, OpenSearch, Solr) er bygget til **tekst**. Den splitter tekst
op i ord (tokens), og laver et *inverteret indeks*: for hvert ord en liste over de dokumenter, ordet
står i. Søgningen finder dokumenter, der indeholder ordene, og *rangerer* dem efter relevans (fx
BM25). Den kan klare stavefejl (fuzzy search), bøjninger af ord og facetter (filtre med antal, fx
"pris 100–200 kr. (34)"). Spørgsmålet er: **"hvilke dokumenter matcher disse ord bedst?"**

**Wide column store** (fx Apache Cassandra, HBase, ScyllaDB) er bygget til **enorme mængder
skrivninger** spredt over mange servere. Data ligger i tabeller med rækker og kolonner, men hver
række kan have sine egne kolonner, og data deles ud på servere efter en *partition key*. Inden for en
partition er rækkerne sorteret efter en *clustering key*. Man designer tabellerne efter de
forespørgsler, man skal køre (query-first), og man duplikerer gerne data. Typisk data: logs,
sensordata, events og beskeder. Spørgsmålet er: **"giv mig rækkerne for denne nøgle, hurtigt, også
med milliarder af rækker"**.

**Vektordatabase** (fx Milvus, Qdrant, Weaviate, Pinecone) er bygget til **embeddings**: vektorer,
som en AI-model har lavet ud fra tekst, billeder eller lyd. Den bruger ANN-indekser (fx HNSW, IVF) til
at finde de vektorer, der ligger tættest på en søgevektor. Den forstår betydning i stedet for præcise
ord. Spørgsmålet er: **"hvad ligner det her mest?"** (se afsnit 1 og 4).

### Hvordan adskiller de sig fra hinanden og fra en relationel database?

| | Relationel (PostgreSQL) | Search engine | Wide column | Vektordatabase |
|---|---|---|---|---|
| Data | Strukturerede rækker med relationer | Tekstdokumenter (JSON) | Rækker med fleksible kolonner, grupperet efter nøgle | Vektorer + metadata |
| Typisk spørgsmål | Præcis match, joins, aggregeringer | Fuldtekst: "indeholder ordene", rangeret | Opslag på partition key, interval på clustering key | Nærmeste naboer: "ligner" |
| Resultat | Præcist sæt af rækker | Rangeret liste efter relevans | Præcist sæt for en nøgle | Rangeret liste efter afstand |
| Skema og joins | Fast skema, joins, normalisering | Fleksibelt, ingen rigtige joins | Query-first, denormaliseret, ingen joins | Fast vektordimension, ingen joins |
| Konsistens | ACID-transaktioner | Næsten realtid, ikke transaktionel | Justerbar (eventual til stærk) | Typisk eventual, ingen transaktioner |
| Skalering | Primært lodret (større server) | Vandret med shards | Vandret, designet til mange servere | Vandret med shards/segmenter |

Forskellen mellem de tre ligger i, **hvad "et godt svar" betyder**. I en relationel database og en
wide column store er svaret præcist: en række matcher, eller den gør ikke. I en search engine og en
vektordatabase er svaret *rangeret*: alle dokumenter får en score, og man får de bedste. Search
engines scorer på ord (leksikalsk), vektordatabaser på betydning (semantisk). Wide column stores
handler slet ikke om søgning, men om at kunne skrive og læse store mængder data på kendte nøgler,
skaleret ud over mange maskiner.

Den relationelle database er den mest generelle: den kan joine, aggregere og sikre konsistens med
transaktioner, men den er hverken god til fuldtekst-relevans, semantisk søgning eller milliarder af
skrivninger i sekundet på tværs af datacentre.


## Kilder

- Milvus dokumentation: https://milvus.io/docs
- Milvus Lite: https://milvus.io/docs/milvus_lite.md
- Indekstyper: https://milvus.io/docs/index.md
- DB-Engines, Vector DBMS: https://db-engines.com/en/ranking/vector+dbms
