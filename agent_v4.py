# ============================================================
# agent_v4.py — Specialist Agents + Live Weather (Phase 4)
# ============================================================
# What's new vs agent_v3.py:
#   1. forecast_agent  — chooses between ML model OR live weather API
#   2. analysis_agent  — new node: trend detection + anomaly scoring
#   3. retrieval_agent — now has TWO tools: semantic search + date filter
#   4. Live weather    — Open-Meteo API (free, no key needed)
#   5. Router upgraded — now routes to analysis when needed
#
# Key concept: each agent node is now a mini-agent with its own
# tool list. The node's LLM decides WHICH tool to call.
# The graph still controls the ORDER of nodes.
# ============================================================

import os
import json
import time
import requests
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from typing import TypedDict, List, Optional

from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI

import chromadb
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

load_dotenv()

# ── Setup ────────────────────────────────────────────────────
print("🔧 Setting up...")

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
model_r2  = round(r2_score(y_test, ml_model.predict(X_test)), 3)
model_mae = round(mean_absolute_error(y_test, ml_model.predict(X_test)), 2)
print(f"✅ ML model ready (R²={model_r2})")

with open("rag_config.json") as f:
    rag_config = json.load(f)
chroma_client = chromadb.PersistentClient(path=rag_config["db_path"])
collection    = chroma_client.get_collection(name=rag_config["collection"])
embed_model   = SentenceTransformer(rag_config["embedding_model"])
print(f"✅ RAG ready ({collection.count()} records)")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    max_retries=3,
)
print("✅ LLM ready\n")

# ============================================================
# STATE — upgraded with new fields for analysis
# ============================================================

class WeatherState(TypedDict):
    user_query:     str
    intent:         str            # retrieve | forecast | both | analysis | general
    retrieved_docs: List[str]
    prediction:     Optional[str]
    analysis:       Optional[str]  # NEW: trend/anomaly analysis results
    live_weather:   Optional[str]  # NEW: real-time weather from Open-Meteo
    final_answer:   str

# ============================================================
# TOOL FUNCTIONS
# Each tool is a plain Python function.
# Unlike Phase 1-3 where tools were @tool decorated for ReAct,
# here we call them directly from inside agent nodes.
# The node's LLM decides which one to call via a prompt.
# ============================================================

# ── Tool 1: ML Prediction ────────────────────────────────────
def tool_ml_predict(day_of_year: int, humidity: float,
                    wind_speed: float, pressure: float) -> str:
    """Run the trained LinearRegression model."""
    input_df = pd.DataFrame(
        [[day_of_year, humidity, wind_speed, pressure]],
        columns=["day_of_year", "humidity", "wind_speed", "meanpressure"]
    )
    predicted = ml_model.predict(input_df)[0]

    if day_of_year < 60 or day_of_year > 330: season = "Winter"
    elif day_of_year < 150:                    season = "Spring/Pre-monsoon"
    elif day_of_year < 270:                    season = "Monsoon/Summer"
    else:                                      season = "Autumn"

    return (
        f"ML prediction: {predicted:.1f}°C | Season: {season} | "
        f"R²={model_r2}, MAE=±{model_mae}°C"
    )

# ── Tool 2: Live Weather API ─────────────────────────────────
# Open-Meteo is completely free — no API key needed.
# Delhi coordinates: lat=28.6139, lon=77.2090
def tool_live_weather() -> str:
    """
    Fetch current real weather for Delhi from Open-Meteo API.
    Free, no key required, returns live data.
    """
    try:
        url = (
            "https://api.open-meteo.com/v1/forecast"
            "?latitude=28.6139&longitude=77.2090"
            "&current=temperature_2m,relative_humidity_2m,"
            "wind_speed_10m,surface_pressure,weather_code"
            "&timezone=Asia/Kolkata"
        )
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()["current"]

        # Weather code → human description
        wmo_codes = {
            0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy",
            3: "Overcast", 45: "Foggy", 51: "Light drizzle",
            61: "Light rain", 63: "Moderate rain", 65: "Heavy rain",
            80: "Rain showers", 95: "Thunderstorm"
        }
        code = data.get("weather_code", 0)
        desc = wmo_codes.get(code, f"Code {code}")

        return (
            f"LIVE Delhi weather right now → "
            f"Temp: {data['temperature_2m']}°C | "
            f"Humidity: {data['relative_humidity_2m']}% | "
            f"Wind: {data['wind_speed_10m']} km/h | "
            f"Pressure: {data['surface_pressure']:.1f} hPa | "
            f"Conditions: {desc}"
        )
    except Exception as e:
        return f"Live weather unavailable: {e}"

# ── Tool 3: Semantic Search ───────────────────────────────────
def tool_semantic_search(query: str, n: int = 5) -> List[dict]:
    """Search ChromaDB by semantic similarity."""
    query_vec = embed_model.encode([query])[0]
    results   = collection.query(
        query_embeddings=[query_vec.tolist()],
        n_results=n
    )
    return results["metadatas"][0]

# ── Tool 4: Filter by condition ───────────────────────────────
def tool_filter_records(min_temp=None, max_temp=None,
                        min_humidity=None, max_humidity=None,
                        month=None) -> List[dict]:
    """Filter the dataset by numeric conditions."""
    filtered = df.copy()
    if min_temp    is not None: filtered = filtered[filtered["meantemp"]  >= min_temp]
    if max_temp    is not None: filtered = filtered[filtered["meantemp"]  <= max_temp]
    if min_humidity is not None: filtered = filtered[filtered["humidity"] >= min_humidity]
    if max_humidity is not None: filtered = filtered[filtered["humidity"] <= max_humidity]
    if month       is not None: filtered = filtered[filtered["month"]     == month]

    return filtered[["date","meantemp","humidity","wind_speed","meanpressure"]]\
           .head(5).to_dict("records")

# ── Tool 5: Trend Analysis ────────────────────────────────────
def tool_trend_analysis(records: List[dict]) -> str:
    """
    Analyse a list of climate records for trends and anomalies.
    Compares values against dataset-wide averages.
    """
    if not records:
        return "No records to analyse."

    temps     = [r["meantemp"]   for r in records]
    humids    = [r["humidity"]   for r in records]
    winds     = [r["wind_speed"] for r in records]

    avg_temp_global  = df["meantemp"].mean()
    avg_humid_global = df["humidity"].mean()
    avg_wind_global  = df["wind_speed"].mean()
    std_temp_global  = df["meantemp"].std()

    avg_temp_local   = sum(temps) / len(temps)
    z_score          = (avg_temp_local - avg_temp_global) / std_temp_global

    # Anomaly classification
    if   z_score >  2: anomaly = "🔴 EXTREME anomaly (unusually hot)"
    elif z_score >  1: anomaly = "🟠 Mild anomaly (warmer than normal)"
    elif z_score < -2: anomaly = "🔵 EXTREME anomaly (unusually cold)"
    elif z_score < -1: anomaly = "🟡 Mild anomaly (cooler than normal)"
    else:              anomaly = "🟢 Normal range"

    trend_dir = "rising" if temps[-1] > temps[0] else "falling" if temps[-1] < temps[0] else "stable"

    return (
        f"Analysis of {len(records)} records → "
        f"Avg temp: {avg_temp_local:.1f}°C (global avg: {avg_temp_global:.1f}°C) | "
        f"Anomaly score: {anomaly} (z={z_score:.2f}) | "
        f"Temp trend: {trend_dir} | "
        f"Avg humidity: {sum(humids)/len(humids):.1f}% | "
        f"Avg wind: {sum(winds)/len(winds):.1f} km/h"
    )

# ============================================================
# NODES — now each one uses the LLM to pick & call tools
# ============================================================

def llm_call(prompt: str) -> str:
    """Helper: call LLM with retry on 503."""
    for attempt in range(3):
        try:
            return llm.invoke(prompt).content.strip()
        except Exception as e:
            if ("503" in str(e) or "UNAVAILABLE" in str(e)) and attempt < 2:
                print(f"   ⚠️  Server busy, retrying in 15s...")
                time.sleep(15)
            else:
                raise

# ── Node 1: Planner ──────────────────────────────────────────
def planner_node(state: WeatherState) -> dict:
    print("\n📋 [Planner] Classifying intent...")

    response = llm_call(f"""Classify this weather query into ONE intent:

- "retrieve"  → wants historical records or past data
- "forecast"  → wants a temperature prediction with specific numbers
- "both"      → wants history AND prediction
- "analysis"  → wants trends, anomalies, comparisons, patterns
- "live"      → asks about current/right now/today's weather
- "general"   → general knowledge, no data needed

Query: "{state['user_query']}"

Reply with ONLY the intent word.""")

    valid = {"retrieve","forecast","both","analysis","live","general"}
    intent = response.lower().replace('"','').replace("'","")
    if intent not in valid:
        intent = "general"

    print(f"   Intent: '{intent}'")
    return {"intent": intent}

# ── Node 2: Retrieval Agent ───────────────────────────────────
# NEW: this agent uses the LLM to decide between semantic search
# vs filter-based search depending on the query type.
def retrieval_agent(state: WeatherState) -> dict:
    print("\n🔍 [Retrieval Agent] Deciding search strategy...")

    # Ask LLM: should we use semantic search or numeric filter?
    strategy = llm_call(f"""Given this query, which search strategy is better?

Query: "{state['user_query']}"

- "semantic" → query is descriptive ("hot dry days", "cold winter")
- "filter"   → query has specific numbers ("temp above 30", "humidity below 40")

Reply with ONLY "semantic" or "filter".""")

    strategy = strategy.lower().strip()
    print(f"   Strategy: '{strategy}'")

    if "filter" in strategy:
        # Ask LLM to extract filter parameters
        params_str = llm_call(f"""Extract filter parameters from this query.
Query: "{state['user_query']}"

Reply ONLY with a Python dict like:
{{"min_temp": 30, "max_temp": null, "min_humidity": null, "max_humidity": 40, "month": null}}

Use null for any unspecified parameter.""")
        try:
            params_str = params_str.replace("null", "None")
            params     = eval(params_str)
            records    = tool_filter_records(**{k: v for k, v in params.items() if v is not None})
        except Exception:
            records = tool_semantic_search(state["user_query"])
    else:
        records = tool_semantic_search(state["user_query"])

    # Format for state
    formatted = [
        f"• {r['date'] if isinstance(r.get('date'), str) else str(r.get('date',''))[:10]}: "
        f"Temp={r['meantemp']:.1f}°C, Humidity={r['humidity']:.1f}%, "
        f"Wind={r['wind_speed']:.1f}km/h, " + f"Pressure={r.get('pressure', r.get('meanpressure', 0)):.1f}hPa"
        for r in records
    ]
    print(f"   Found {len(formatted)} records")
    return {"retrieved_docs": formatted}

# ── Node 3: Forecast Agent ────────────────────────────────────
# NEW: chooses between ML model and live weather API
def forecast_agent(state: WeatherState) -> dict:
    print("\n📊 [Forecast Agent] Choosing forecast tool...")

    # Decide: use ML prediction or live weather API?
    tool_choice = llm_call(f"""Which tool should answer this query?

Query: "{state['user_query']}"
Retrieved docs available: {bool(state.get('retrieved_docs'))}

- "ml"   → user gave specific numbers (day, humidity, wind, pressure) OR we have retrieved conditions
- "live" → user asks about current/today/right now weather

Reply with ONLY "ml" or "live".""")

    tool_choice = tool_choice.lower().strip()
    print(f"   Tool choice: '{tool_choice}'")

    if "live" in tool_choice:
        # Fetch real-time weather
        live_result = tool_live_weather()
        print(f"   {live_result}")

        # If intent is "both", also run ML with live conditions
        if state["intent"] == "both" and "unavailable" not in live_result:
            try:
                parts    = live_result.split("|")
                temp_val = float(parts[0].split(":")[1].replace("°C","").strip())
                hum_val  = float(parts[1].split(":")[1].replace("%","").strip())
                wind_val = float(parts[2].split(":")[1].replace("km/h","").strip())
                pres_val = float(parts[3].split(":")[1].replace("hPa","").strip())
                from datetime import datetime
                doy      = datetime.now().timetuple().tm_yday
                ml_check = tool_ml_predict(doy, hum_val, wind_val, pres_val)
                return {"live_weather": live_result, "prediction": ml_check}
            except Exception:
                pass
        return {"live_weather": live_result}

    else:
        # ML prediction path
        if state.get("retrieved_docs") and state["intent"] in ("both",):
            # Extract conditions from retrieved docs
            docs_text  = "\n".join(state["retrieved_docs"])
            params_str = llm_call(f"""From these records, extract conditions of the most relevant day.
Records:
{docs_text}

Original query: "{state['user_query']}"

Reply ONLY with: day_of_year, humidity, wind_speed, pressure
Example: 40, 68.4, 7.9, 1016.4""")
        else:
            # Extract directly from query
            params_str = llm_call(f"""Extract weather parameters from this query.
Query: "{state['user_query']}"

Reply ONLY with: day_of_year, humidity, wind_speed, pressure
Use Delhi averages for missing values: 180, 60, 8, 1010""")

        print(f"   Params: {params_str}")
        try:
            parts    = [p.strip() for p in params_str.split(",")]
            result   = tool_ml_predict(
                int(float(parts[0])), float(parts[1]),
                float(parts[2]),      float(parts[3])
            )
        except Exception as e:
            result = f"Prediction failed: {e}"

        print(f"   Result: {result}")
        return {"prediction": result}

# ── Node 4: Analysis Agent ────────────────────────────────────
# NEW node — didn't exist in Phase 3
def analysis_agent(state: WeatherState) -> dict:
    print("\n🔬 [Analysis Agent] Running trend & anomaly analysis...")

    # First retrieve relevant records if not already done
    if not state.get("retrieved_docs"):
        records_raw = tool_semantic_search(state["user_query"])
    else:
        # Re-fetch raw dicts for numeric analysis
        records_raw = tool_semantic_search(state["user_query"])

    # Run trend analysis tool
    analysis_result = tool_trend_analysis(records_raw)
    print(f"   {analysis_result}")

    # Ask LLM to interpret the raw analysis in context of the query
    interpretation = llm_call(f"""Interpret this climate analysis for the user's question.

User question: "{state['user_query']}"
Raw analysis: {analysis_result}

Write 1-2 sentences interpreting what this means for the user.
Be specific with numbers.""")

    return {"analysis": f"{analysis_result}\n\nInterpretation: {interpretation}"}

# ── Node 5: Responder ─────────────────────────────────────────
def responder_node(state: WeatherState) -> dict:
    print("\n✍️  [Responder] Writing final answer...")

    parts = []
    if state.get("live_weather"):   parts.append(f"Live weather:\n{state['live_weather']}")
    if state.get("retrieved_docs"): parts.append(f"Historical records:\n" + "\n".join(state["retrieved_docs"]))
    if state.get("prediction"):     parts.append(f"ML prediction:\n{state['prediction']}")
    if state.get("analysis"):       parts.append(f"Analysis:\n{state['analysis']}")

    context = "\n\n".join(parts) if parts else "No data tools were used."

    answer = llm_call(f"""You are a helpful Delhi weather assistant.
Answer clearly using the context below. Cite specific numbers.
Keep it to 3-4 sentences max.

Question: "{state['user_query']}"
Context:
{context}""")

    return {"final_answer": answer}

# ============================================================
# ROUTER — upgraded with new intents
# ============================================================

def route_after_planner(state: WeatherState) -> str:
    intent = state["intent"]
    print(f"\n🔀 [Router] '{intent}' → ", end="")

    routes = {
        "retrieve":  "retrieval_agent",
        "forecast":  "forecast_agent",
        "both":      "retrieval_agent",
        "analysis":  "retrieval_agent",   # analysis needs data first
        "live":      "forecast_agent",    # live goes straight to forecast
        "general":   "responder_node",
    }
    target = routes.get(intent, "responder_node")
    print(target)
    return target

def route_after_retrieval(state: WeatherState) -> str:
    intent = state["intent"]
    if intent == "both":     return "forecast_agent"
    if intent == "analysis": return "analysis_agent"
    return "responder_node"

def route_after_forecast(state: WeatherState) -> str:
    # After forecast, check if analysis is also needed
    if state["intent"] == "analysis":
        return "analysis_agent"
    return "responder_node"

# ============================================================
# BUILD THE GRAPH
# ============================================================

print("🏗️  Building graph...")
g = StateGraph(WeatherState)

g.add_node("planner_node",    planner_node)
g.add_node("retrieval_agent", retrieval_agent)
g.add_node("forecast_agent",  forecast_agent)
g.add_node("analysis_agent",  analysis_agent)
g.add_node("responder_node",  responder_node)

g.add_edge(START, "planner_node")
g.add_edge("analysis_agent",  "responder_node")
g.add_edge("responder_node",  END)

g.add_conditional_edges("planner_node", route_after_planner, {
    "retrieval_agent": "retrieval_agent",
    "forecast_agent":  "forecast_agent",
    "responder_node":  "responder_node",
})
g.add_conditional_edges("retrieval_agent", route_after_retrieval, {
    "forecast_agent":  "forecast_agent",
    "analysis_agent":  "analysis_agent",
    "responder_node":  "responder_node",
})
g.add_conditional_edges("forecast_agent", route_after_forecast, {
    "analysis_agent":  "analysis_agent",
    "responder_node":  "responder_node",
})

graph = g.compile()
print("✅ Graph compiled!\n")

# ============================================================
# CHAT LOOP
# ============================================================

def run_query(query: str) -> str:
    initial: WeatherState = {
        "user_query":     query,
        "intent":         "",
        "retrieved_docs": [],
        "prediction":     None,
        "analysis":       None,
        "live_weather":   None,
        "final_answer":   "",
    }
    print("\n" + "─"*55)
    print(f"🌤️  {query}")
    print("─"*55)
    final = graph.invoke(initial)
    return final["final_answer"]


print("="*55)
print("🌤️  Weather Agent v4 — Specialist Agents!")
print("="*55)
print("New capabilities:")
print("  🌐 Live weather  → 'What's the weather in Delhi right now?'")
print("  🔬 Analysis      → 'Is Delhi getting hotter over time?'")
print("  🔍 Smart search  → 'Find days where humidity was below 30%'")
print("  📊 Forecast      → 'Predict temp for day 200, humidity 60, wind 8, pressure 1010'")
print("  🔁 Combined      → 'Find coldest day and predict temperature'")
print("\nType 'quit' to exit.\n")

while True:
    user_input = input("You: ").strip()
    if not user_input: continue
    if user_input.lower() in ["quit","exit","q"]:
        print("👋 Goodbye!")
        break
    answer = run_query(user_input)
    print(f"\n💬 {answer}\n")
    print("="*55)
    time.sleep(6)