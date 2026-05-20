# ============================================================
# rag_setup.py — Build the RAG Knowledge Base (Phase 2)
# ============================================================
# What this file teaches you:
#   1. How to turn structured CSV data into searchable text chunks
#   2. What embeddings are and how to generate them locally (free)
#   3. How ChromaDB stores and indexes vectors
#   4. How to query the knowledge base with natural language
#
# Run this file ONCE to build the knowledge base.
# After it runs, a folder called "climate_db" will appear.
# That folder IS your vector database — persisted on disk.
# ============================================================

import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
import os
import json

# ── Step 1: Load your CSV ────────────────────────────────────
# Same data your ML model uses — we're just making it searchable now
print("📂 Loading climate data...")
df = pd.read_csv("DailyDelhiClimateTest.csv")
df["date"] = pd.to_datetime(df["date"])
df["day_of_year"] = df["date"].dt.dayofyear
df["month"] = df["date"].dt.month
df["month_name"] = df["date"].dt.strftime("%B")  # "January", "February" etc.

# Fix pressure outlier (same as wheater.py)
df.loc[df["meanpressure"] < 900, "meanpressure"] = df["meanpressure"].mean()

print(f"✅ Loaded {len(df)} days of climate data")
print(f"   Date range: {df['date'].min().date()} to {df['date'].max().date()}")

# ── Step 2: Chunking ─────────────────────────────────────────
# CONCEPT: A "chunk" is a small piece of text that represents one unit of knowledge.
# For our dataset, one chunk = one day of climate data.
# We convert each CSV row into a human-readable text string.
# Why text? Because embedding models are trained on text — they understand
# "hot dry day" better than raw numbers like [34.5, 17, 14, 1012].
#
# We also add descriptive labels ("Very Hot", "Low humidity") so the
# embedding captures the MEANING, not just the numbers.

def describe_temperature(temp):
    if temp < 10:   return "Very Cold"
    if temp < 18:   return "Cold"
    if temp < 25:   return "Mild"
    if temp < 30:   return "Warm"
    if temp < 35:   return "Hot"
    return "Very Hot"

def describe_humidity(hum):
    if hum < 30:    return "Very Dry"
    if hum < 50:    return "Dry"
    if hum < 70:    return "Moderate humidity"
    if hum < 85:    return "Humid"
    return "Very Humid"

def describe_wind(wind):
    if wind < 5:    return "Calm"
    if wind < 10:   return "Light breeze"
    if wind < 20:   return "Moderate wind"
    return "Strong wind"

def row_to_chunk(row):
    """
    Convert one CSV row into a rich text chunk.
    This is the most important function in RAG — the quality of your
    chunks directly determines the quality of your search results.
    """
    temp_desc  = describe_temperature(row["meantemp"])
    hum_desc   = describe_humidity(row["humidity"])
    wind_desc  = describe_wind(row["wind_speed"])

    return (
        f"Date: {row['date'].strftime('%Y-%m-%d')} "
        f"({row['month_name']}, day {int(row['day_of_year'])} of year). "
        f"Temperature: {row['meantemp']:.1f}°C ({temp_desc}). "
        f"Humidity: {row['humidity']:.1f}% ({hum_desc}). "
        f"Wind speed: {row['wind_speed']:.1f} km/h ({wind_desc}). "
        f"Pressure: {row['meanpressure']:.1f} hPa."
    )

print("\n📝 Creating text chunks from CSV rows...")
chunks = []
metadatas = []  # We store the raw numbers separately for filtering later
ids = []        # ChromaDB needs a unique ID for each chunk

for i, row in df.iterrows():
    chunk_text = row_to_chunk(row)
    chunks.append(chunk_text)

    # Metadata = the original numbers stored alongside the text
    # Useful for filtering: "find days where temp > 30" uses metadata, not embeddings
    metadatas.append({
        "date":        row["date"].strftime("%Y-%m-%d"),
        "meantemp":    float(row["meantemp"]),
        "humidity":    float(row["humidity"]),
        "wind_speed":  float(row["wind_speed"]),
        "pressure":    float(row["meanpressure"]),
        "day_of_year": int(row["day_of_year"]),
        "month":       int(row["month"]),
        "month_name":  row["month_name"],
    })
    ids.append(f"day_{i}")  # Unique ID: "day_0", "day_1", etc.

print(f"✅ Created {len(chunks)} chunks")
print(f"\nExample chunk:\n  {chunks[0]}\n")

# ── Step 3: Embedding model ──────────────────────────────────
# CONCEPT: An embedding model converts text → a list of numbers (vector).
# Similar texts get similar vectors. This is what makes semantic search work.
#
# We use "all-MiniLM-L6-v2" — a small, fast, free model that runs locally.
# It produces 384-dimensional vectors (a list of 384 numbers per chunk).
# First run downloads ~90MB model to your machine. Subsequent runs use cache.
#
# Why local instead of OpenAI embeddings?
#   - Free (no API calls for embeddings)
#   - Fast (no network latency)
#   - Private (your data never leaves your machine)

print("🧠 Loading embedding model (downloads ~90MB on first run)...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ Embedding model ready!")

# Generate embeddings for all chunks
# encode() returns a numpy array of shape (114, 384)
# — 114 chunks, each represented as 384 numbers
print("\n⚡ Generating embeddings for all chunks...")
embeddings = embedding_model.encode(chunks, show_progress_bar=True)
print(f"✅ Generated {len(embeddings)} embeddings, each with {len(embeddings[0])} dimensions")

# ── Step 4: ChromaDB — store everything ─────────────────────
# CONCEPT: ChromaDB is a vector database.
# It stores: the text chunks + their embeddings + metadata
# And lets you search by: semantic similarity OR metadata filters OR both.
#
# PersistentClient saves the database to disk (the "climate_db" folder).
# Next time you load it, everything is already there — no need to re-embed.

print("\n💾 Setting up ChromaDB...")

# Delete old DB if it exists (clean rebuild)
db_path = "./climate_db"
if os.path.exists(db_path):
    import shutil
    shutil.rmtree(db_path)
    print("   Cleared old database")

# Create persistent client — data saved to ./climate_db folder
client = chromadb.PersistentClient(path=db_path)

# A "collection" is like a table in a regular database
# We tell it NOT to use its own embedding function (we pre-computed embeddings above)
collection = client.create_collection(
    name="delhi_climate",
    metadata={"description": "Daily Delhi climate data 2017"}
)

# Add everything to ChromaDB in one call
# ChromaDB stores: id + embedding + text (document) + metadata
collection.add(
    ids=ids,
    embeddings=embeddings.tolist(),   # ChromaDB needs Python lists, not numpy arrays
    documents=chunks,                  # The text chunks
    metadatas=metadatas               # The raw numbers for filtering
)

print(f"✅ ChromaDB ready! Stored {collection.count()} documents in '{db_path}/'")

# ── Step 5: Test the knowledge base ─────────────────────────
# Let's run 3 test queries to confirm everything works.
# This is also a great demonstration of semantic search in action.

print("\n" + "="*55)
print("🔍 Testing semantic search...")
print("="*55)

def search(query, n_results=3):
    """Search the knowledge base with a natural language query."""
    # Step 1: Embed the query using the SAME model used to embed chunks
    # This is crucial — query and chunks must be in the same vector space
    query_embedding = embedding_model.encode([query])[0]

    # Step 2: Find the n most similar chunks using cosine similarity
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=n_results
    )
    return results["documents"][0], results["metadatas"][0]

# Test 1: Semantic search — "hot" maps to high temperature chunks
print("\n📌 Query 1: 'very hot days in Delhi'")
docs, metas = search("very hot days in Delhi", n_results=3)
for doc, meta in zip(docs, metas):
    print(f"   → {meta['date']}: {meta['meantemp']}°C, humidity {meta['humidity']}%")

# Test 2: Semantic search — opposite end of spectrum
print("\n📌 Query 2: 'cold winter days'")
docs, metas = search("cold winter days", n_results=3)
for doc, meta in zip(docs, metas):
    print(f"   → {meta['date']}: {meta['meantemp']}°C, humidity {meta['humidity']}%")

# Test 3: Multi-concept search — both conditions matter
print("\n📌 Query 3: 'humid rainy weather with low wind'")
docs, metas = search("humid rainy weather with low wind", n_results=3)
for doc, meta in zip(docs, metas):
    print(f"   → {meta['date']}: {meta['meantemp']}°C, humidity {meta['humidity']}%, wind {meta['wind_speed']} km/h")

# ── Step 6: Save embedding model path for agent to reuse ─────
# We save the model name to a config file so agent_v2.py can load
# the same model without hardcoding the string in two places.
config = {"embedding_model": "all-MiniLM-L6-v2", "db_path": "./climate_db", "collection": "delhi_climate"}
with open("rag_config.json", "w") as f:
    json.dump(config, f, indent=2)

print("\n" + "="*55)
print("✅ RAG Knowledge Base built successfully!")
print(f"   Database saved to: {db_path}/")
print(f"   Config saved to:   rag_config.json")
print("\n🚀 Next step: run agent_v2.py to use this in your agent!")
print("="*55)