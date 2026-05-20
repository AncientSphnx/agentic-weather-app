# ============================================================
# agent_v2.py — Weather Agent with RAG (Phase 2)
# ============================================================
# What's new vs agent_v1.py:
#   - A third tool: search_climate_history (uses ChromaDB + embeddings)
#   - Agent now has MEMORY of every day in the dataset
#   - Agent can answer evidence-based questions like:
#       "Find days similar to today's conditions"
#       "What were the coldest days on record?"
#       "Find hot dry days and predict if tomorrow will be similar"
#
# The ML prediction tool still works exactly as before.
# RAG is additive — it gives the agent a searchable knowledge base.
# ============================================================

import os
import json
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import tool
from langchain import hub

# RAG imports
import chromadb
from sentence_transformers import SentenceTransformer

# ML imports
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

load_dotenv()

# ── Step 1: Train ML model (same as v1) ─────────────────────
print("🔧 Training ML model...")
df = pd.read_csv("DailyDelhiClimateTest.csv")
df["date"] = pd.to_datetime(df["date"])
df["day_of_year"] = df["date"].dt.dayofyear
df["month"] = df["date"].dt.month
df.loc[df["meanpressure"] < 900, "meanpressure"] = df["meanpressure"].mean()

X = df[["day_of_year", "humidity", "wind_speed", "meanpressure"]]
y = df["meantemp"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
ml_model = LinearRegression()
ml_model.fit(X_train, y_train)
y_pred_test = ml_model.predict(X_test)
model_r2  = round(r2_score(y_test, y_pred_test), 3)
model_mae = round(mean_absolute_error(y_test, y_pred_test), 2)
print(f"✅ ML model ready! R²={model_r2}, MAE=±{model_mae}°C")

# ── Step 2: Load RAG knowledge base ─────────────────────────
# We load the ChromaDB collection we built in rag_setup.py
# and the same embedding model — MUST be the same model, otherwise
# query vectors and stored vectors won't be in the same space
print("📚 Loading RAG knowledge base...")

with open("rag_config.json") as f:
    rag_config = json.load(f)

chroma_client = chromadb.PersistentClient(path=rag_config["db_path"])
collection = chroma_client.get_collection(name=rag_config["collection"])
embedding_model = SentenceTransformer(rag_config["embedding_model"])

print(f"✅ Knowledge base ready! {collection.count()} climate records loaded.")

# ── Step 3: Define Tools ─────────────────────────────────────

@tool
def predict_temperature(input_str: str) -> str:
    """
    Predicts the mean temperature in Delhi given weather conditions.
    Use this tool when the user asks about temperature prediction or forecast.

    Input must be a comma-separated string in this exact format:
        day_of_year, humidity, wind_speed, pressure
    Example:
        200, 65, 10, 1010

    Where:
        day_of_year: Day of the year (1-365)
        humidity: Relative humidity percentage (0-100)
        wind_speed: Wind speed in km/h (0-50)
        pressure: Atmospheric pressure in hPa (900-1050)
    """
    try:
        parts = [p.strip() for p in input_str.split(",")]
        if len(parts) != 4:
            return "Error: provide exactly 4 values as: day_of_year, humidity, wind_speed, pressure"
        day_of_year = int(float(parts[0]))
        humidity    = float(parts[1])
        wind_speed  = float(parts[2])
        pressure    = float(parts[3])
    except ValueError:
        return "Error: all values must be numbers."

    if not (1 <= day_of_year <= 365): return "Error: day_of_year must be 1-365."
    if not (0 <= humidity <= 100):    return "Error: humidity must be 0-100."
    if not (0 <= wind_speed <= 50):   return "Error: wind_speed must be 0-50."
    if not (900 <= pressure <= 1050): return "Error: pressure must be 900-1050."

    input_df = pd.DataFrame(
        [[day_of_year, humidity, wind_speed, pressure]],
        columns=["day_of_year", "humidity", "wind_speed", "meanpressure"]
    )
    predicted_temp = ml_model.predict(input_df)[0]

    if day_of_year < 60 or day_of_year > 330: season = "Winter"
    elif day_of_year < 150: season = "Spring/Pre-monsoon"
    elif day_of_year < 270: season = "Monsoon/Summer"
    else: season = "Autumn/Post-monsoon"

    return (
        f"Predicted temperature: {predicted_temp:.1f}°C | "
        f"Season: {season} | "
        f"Model accuracy: R²={model_r2}, MAE=±{model_mae}°C"
    )


@tool
def search_climate_history(query: str) -> str:
    """
    Searches the historical Delhi climate database using natural language.
    Use this tool when the user asks about:
    - Historical climate patterns ("what were the hottest days?")
    - Finding similar past conditions ("find days like today")
    - Evidence-based questions ("has Delhi ever had temp above 33°C?")
    - Comparing conditions across dates

    Input: a natural language description of what you're looking for.
    Examples:
        "very hot and dry days"
        "cold humid winter mornings"
        "days with strong wind and low pressure"
        "mild comfortable weather in spring"
    """
    # ── The 3-step RAG retrieval process ────────────────────
    # Step A: Embed the query into the same vector space as our chunks
    query_embedding = embedding_model.encode([query])[0]

    # Step B: Find the 5 most semantically similar chunks in ChromaDB
    # ChromaDB computes cosine similarity between query vector and all stored vectors
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=5
    )

    # Step C: Format results for the LLM to read and reason about
    docs      = results["documents"][0]   # The text chunks
    metadatas = results["metadatas"][0]   # The raw numbers

    if not docs:
        return "No matching climate records found."

    # Build a clean summary the LLM can reason over
    output = f"Top {len(docs)} climate records matching '{query}':\n\n"
    for i, (doc, meta) in enumerate(zip(docs, metadatas), 1):
        output += (
            f"{i}. {meta['date']} — "
            f"Temp: {meta['meantemp']:.1f}°C, "
            f"Humidity: {meta['humidity']:.1f}%, "
            f"Wind: {meta['wind_speed']:.1f} km/h, "
            f"Pressure: {meta['pressure']:.1f} hPa\n"
        )

    # Add a quick statistical summary of the retrieved records
    temps = [m["meantemp"] for m in metadatas]
    output += (
        f"\nSummary of retrieved records: "
        f"avg temp {sum(temps)/len(temps):.1f}°C, "
        f"range {min(temps):.1f}–{max(temps):.1f}°C"
    )
    return output


@tool
def get_dataset_stats(query: str) -> str:
    """
    Returns statistics and facts about the Delhi climate dataset.
    Use this when the user asks about overall averages, records,
    dataset size, or date ranges.

    Args:
        query: What the user wants to know about the dataset.
    """
    monthly = df.groupby("month")["meantemp"].mean().round(1).to_dict()
    month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                   7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    monthly_str = ", ".join([f"{month_names[m]}: {t}°C" for m, t in monthly.items()])

    return (
        f"Dataset: {len(df)} days ({df['date'].min().date()} to {df['date'].max().date()}) | "
        f"Temp range: {df['meantemp'].min():.1f}°C to {df['meantemp'].max():.1f}°C | "
        f"Avg temp: {df['meantemp'].mean():.1f}°C | "
        f"Avg humidity: {df['humidity'].mean():.1f}% | "
        f"Hottest day: {df.loc[df['meantemp'].idxmax(), 'date'].date()} | "
        f"Coldest day: {df.loc[df['meantemp'].idxmin(), 'date'].date()} | "
        f"Monthly avg temps → {monthly_str}"
    )


# ── Step 4: Create agent (same pattern as v1) ────────────────
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Now 3 tools — agent picks whichever fits the question
tools = [predict_temperature, search_climate_history, get_dataset_stats]

prompt = hub.pull("hwchase17/react")
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=6,
    handle_parsing_errors=True
)

# ── Step 5: Chat loop ────────────────────────────────────────
print("\n" + "="*55)
print("🌤️  Weather Agent v2 — RAG Enabled!")
print("="*55)
print("New: I can now search through historical climate records!")
print("\nTry these new questions:")
print("  • What were the hottest days on record?")
print("  • Find days with very high humidity and low wind")
print("  • Find the coldest day and predict temperature for similar conditions")
print("  • Were there any days with temperature above 33°C?")
print("  • Find days similar to: humidity 60, wind 5, pressure 1010")
print("\nOr old favourites:")
print("  • Predict temperature for day 100 with humidity 50, wind 10, pressure 1008")
print("\nType 'quit' to exit.\n")

while True:
    user_input = input("You: ").strip()
    if not user_input:
        continue
    if user_input.lower() in ["quit", "exit", "q"]:
        print("👋 Goodbye!")
        break

    print("\n🤖 Agent thinking...\n")
    try:
        response = agent_executor.invoke({"input": user_input})
        print(f"\n💬 Final Answer: {response['output']}\n")
        print("-" * 55)
    except Exception as e:
        print(f"❌ Error: {e}\n")