# Kører demoen med Milvus Lite i en Linux-container.
# Virker på Windows, macOS og Linux, så længe Docker Desktop/Docker er installeret.
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY demo.py data.py ./

# Databasefilen (film.db) lægges i /app/data, så den kan ses udefra via et volume
WORKDIR /app/data
CMD ["python", "/app/demo.py", "--pause"]
