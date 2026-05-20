# ============================================================
# agent_v5.py — Memory + Explainability (Phase 5)
# ============================================================
# What's new vs agent_v4.py:
#   1. Conversation memory — agent remembers last 6 turns
#   2. Follow-up questions work ("what about humidity on that day?")
#   3. Explainability — responder cites specific records + explains why
#   4. Citation builder — shows which data points support the answer
#   5. Memory summary — type 'memory' to see conversation history
#
# Key concept: memory lives OUTSIDE the graph (in a Python list).
# It gets injected INTO state on each turn, so every node can
# see what was discussed before. After each turn it gets updated.
# The graph itself is stateless — memory is managed by the caller.
# ============================================================

import os
import json
import time
import requests
import pandas as pd
from dotenv import load_dotenv
from typing import TypedDict, List, Optional
from datetime import datetime

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
# STEP 1: State — now includes conversation_history
# ============================================================
# conversation_history is a list of dicts:
# [
#   {"role": "user",      "content": "Find the hottest day"},
#   {"role": "assistant", "content": "April 20, 2017 — 34.5°C",
#    "retrieved_docs": [...], "prediction": "..."},
#   ...
# ]
# It flows through state so every node can reference it.

class WeatherState(TypedDict):
    user_query:           str
    intent:               str
    conversation_history: List[dict]   # NEW — full history injected each turn
    retrieved_docs:       List[str]
    prediction:           Optional[str]
    analysis:             Optional[str]
    live_weather:         Optional[str]
    citations:            List[str]    # NEW — specific records that support the answer
    final_answer:         str

# ============================================================
# TOOL FUNCTIONS (same as v4, unchanged)
# ============================================================

def tool_ml_predict(day_of_year: int, humidity: float,
                    wind_speed: float, pressure: float) -> str:
    input_df  = pd.DataFrame(
        [[day_of_year, humidity, wind_speed, pressure]],
        columns=["day_of_year", "humidity", "wind_speed", "meanpressure"]
    )
    predicted = ml_model.predict(input_df)[0]
    if day_of_year < 60 or day_of_year > 330: season = "Winter"
    elif day_of_year < 150:                    season = "Spring/Pre-monsoon"
    elif day_of_year < 270:                    season = "Monsoon/Summer"
    else:                                      season = "Autumn"
    return (f"ML prediction: {predicted:.1f}°C | Season: {season} | "
            f"R²={model_r2}, MAE=±{model_mae}°C | "
            f"Inputs: day={day_of_year}, humidity={humidity}%, "
            f"wind={wind_speed}km/h, pressure={pressure}hPa")

def tool_live_weather() -> str:
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
        wmo  = {0:"Clear sky",1:"Mainly clear",2:"Partly cloudy",3:"Overcast",
                45:"Foggy",51:"Light drizzle",61:"Light rain",63:"Moderate rain",
                65:"Heavy rain",80:"Rain showers",95:"Thunderstorm"}
        desc = wmo.get(data.get("weather_code", 0), "Unknown")
        return (f"LIVE Delhi → Temp: {data['temperature_2m']}°C | "
                f"Humidity: {data['relative_humidity_2m']}% | "
                f"Wind: {data['wind_speed_10m']}km/h | "
                f"Pressure: {data['surface_pressure']:.1f}hPa | "
                f"Conditions: {desc}")
    except Exception as e:
        return f"Live weather unavailable: {e}"

def tool_semantic_search(query: str, n: int = 5) -> List[dict]:
    vec     = embed_model.encode([query])[0]
    results = collection.query(query_embeddings=[vec.tolist()], n_results=n)
    return results["metadatas"][0]

def tool_filter_records(min_temp=None, max_temp=None,
                        min_humidity=None, max_humidity=None,
                        month=None) -> List[dict]:
    filtered = df.copy()
    if min_temp     is not None: filtered = filtered[filtered["meantemp"]  >= min_temp]
    if max_temp     is not None: filtered = filtered[filtered["meantemp"]  <= max_temp]
    if min_humidity is not None: filtered = filtered[filtered["humidity"]  >= min_humidity]
    if max_humidity is not None: filtered = filtered[filtered["humidity"]  <= max_humidity]
    if month        is not None: filtered = filtered[filtered["month"]     == month]
    return filtered[["date","meantemp","humidity","wind_speed","meanpressure"]]\
           .head(5).to_dict("records")

def tool_trend_analysis(records: List[dict]) -> str:
    if not records: return "No records to analyse."
    temps  = [r.get("meantemp", 0)   for r in records]
    humids = [r.get("humidity", 0)   for r in records]
    winds  = [r.get("wind_speed", 0) for r in records]
    avg_g  = df["meantemp"].mean()
    std_g  = df["meantemp"].std()
    avg_l  = sum(temps) / len(temps)
    z      = (avg_l - avg_g) / std_g
    if   z >  2: anomaly = "🔴 EXTREME (unusually hot)"
    elif z >  1: anomaly = "🟠 Mild (warmer than normal)"
    elif z < -2: anomaly = "🔵 EXTREME (unusually cold)"
    elif z < -1: anomaly = "🟡 Mild (cooler than normal)"
    else:        anomaly = "🟢 Normal range"
    trend = "rising" if temps[-1] > temps[0] else "falling" if temps[-1] < temps[0] else "stable"
    return (f"Analysis of {len(records)} records → "
            f"Avg temp: {avg_l:.1f}°C (global avg: {avg_g:.1f}°C) | "
            f"Anomaly: {anomaly} (z={z:.2f}) | Trend: {trend} | "
            f"Avg humidity: {sum(humids)/len(humids):.1f}% | "
            f"Avg wind: {sum(winds)/len(winds):.1f}km/h")

# ============================================================
# HELPER: format conversation history for prompt injection
# ============================================================
# This is the core of memory — we turn the history list into
# a readable string that gets prepended to every LLM prompt.
# The LLM reads this and "remembers" what was discussed.

def format_history(history: List[dict], max_turns: int = 6) -> str:
    """Convert conversation history list to a prompt-ready string."""
    if not history:
        return "No previous conversation."

    # Only keep the last max_turns entries (sliding window)
    # This prevents the prompt from growing infinitely
    recent = history[-max_turns:]

    lines = []
    for turn in recent:
        lines.append(f"User: {turn['user']}")
        lines.append(f"Assistant: {turn['assistant']}")
        # If this turn had retrieved docs, include a summary
        # so follow-up questions can reference "that day" etc.
        if turn.get("retrieved_docs"):
            lines.append(f"[Data shown: {turn['retrieved_docs'][0] if turn['retrieved_docs'] else 'none'}]")
    return "\n".join(lines)


# ============================================================
# LLM HELPER
# ============================================================

def llm_call(prompt: str) -> str:
    for attempt in range(3):
        try:
            return llm.invoke(prompt).content.strip()
        except Exception as e:
            if ("503" in str(e) or "UNAVAILABLE" in str(e)) and attempt < 2:
                print(f"   ⚠️  Server busy, retrying in 15s...")
                time.sleep(15)
            else:
                raise


# ============================================================
# NODES
# ============================================================

# ── Node 1: Planner — now memory-aware ───────────────────────
# KEY CHANGE: we inject conversation_history into the planner prompt.
# This lets the planner understand follow-up questions like
# "what about humidity on that day?" by reading prior context.

def planner_node(state: WeatherState) -> dict:
    print("\n📋 [Planner] Classifying intent (with memory)...")

    history_str = format_history(state.get("conversation_history", []))

    # The history is injected directly into the classification prompt
    # The LLM can now see what was discussed and classify correctly
    response = llm_call(f"""You are a query classifier for a weather AI system.
Use the conversation history to understand follow-up questions.

CONVERSATION HISTORY:
{history_str}

CURRENT QUERY: "{state['user_query']}"

Classify into ONE intent:
- "retrieve"  → wants historical records or past data
- "forecast"  → wants a temperature prediction with specific numbers
- "both"      → wants history AND prediction
- "analysis"  → wants trends, anomalies, patterns
- "live"      → asks about current/right now weather
- "followup"  → refers to something from conversation history ("that day", "those conditions", "what about...")
- "general"   → general knowledge, no data needed

Reply with ONLY the intent word.""")

    valid  = {"retrieve","forecast","both","analysis","live","followup","general"}
    intent = response.lower().replace('"','').replace("'","").strip()
    if intent not in valid:
        intent = "general"

    print(f"   Intent: '{intent}'")
    return {"intent": intent}


# ── Node 2: Memory Resolver — NEW ────────────────────────────
# When intent is "followup", this node extracts context from
# conversation history and rewrites the query to be self-contained.
# Example:
#   History: "Apr 20 was hottest day — 34.5°C, humidity 27.5%"
#   Query:   "what was the humidity on that day?"
#   Resolved: "What was the humidity on April 20, 2017?"

def memory_resolver_node(state: WeatherState) -> dict:
    print("\n🧠 [Memory Resolver] Resolving follow-up from history...")

    history_str = format_history(state.get("conversation_history", []))

    resolved = llm_call(f"""The user asked a follow-up question that references something from earlier.
Rewrite it as a fully self-contained question using context from history.

CONVERSATION HISTORY:
{history_str}

FOLLOW-UP QUERY: "{state['user_query']}"

Write the resolved, self-contained question. No explanation.""")

    print(f"   Resolved to: '{resolved}'")

    # Re-classify the resolved query
    reclassified = llm_call(f"""Classify this query:
"{resolved}"

Options: retrieve | forecast | both | analysis | live | general
Reply with ONLY the intent word.""")

    valid  = {"retrieve","forecast","both","analysis","live","general"}
    intent = reclassified.lower().strip()
    if intent not in valid:
        intent = "retrieve"

    print(f"   Re-classified as: '{intent}'")
    # Update user_query to the resolved version so downstream nodes use it
    return {"user_query": resolved, "intent": intent}


# ── Node 3: Retrieval Agent (same as v4) ─────────────────────
def retrieval_agent(state: WeatherState) -> dict:
    print("\n🔍 [Retrieval Agent] Searching...")

    strategy = llm_call(f"""Search strategy for: "{state['user_query']}"
- "semantic" → descriptive query
- "filter"   → has specific numbers/thresholds
Reply ONLY "semantic" or "filter".""")

    if "filter" in strategy.lower():
        params_str = llm_call(f"""Extract filter params from: "{state['user_query']}"
Reply ONLY as Python dict:
{{"min_temp": null, "max_temp": null, "min_humidity": null, "max_humidity": null, "month": null}}""")
        try:
            params  = eval(params_str.replace("null","None"))
            records = tool_filter_records(**{k: v for k,v in params.items() if v is not None})
        except Exception:
            records = tool_semantic_search(state["user_query"])
    else:
        records = tool_semantic_search(state["user_query"])

    formatted = [
        f"• {str(r.get('date',''))[:10]}: "
        f"Temp={r['meantemp']:.1f}°C, Humidity={r['humidity']:.1f}%, "
        f"Wind={r['wind_speed']:.1f}km/h, "
        f"Pressure={r.get('pressure', r.get('meanpressure', 0)):.1f}hPa"
        for r in records
    ]
    print(f"   Found {len(formatted)} records")

    # ── Citation builder ─────────────────────────────────────
    # NEW: we build citations here — specific records that will
    # be shown to the user as sources for the answer.
    # This is the "grounding" part of RAG — every claim has a source.
    citations = [f"[Source {i+1}] {doc}" for i, doc in enumerate(formatted)]

    return {"retrieved_docs": formatted, "citations": citations}


# ── Node 4: Forecast Agent (same as v4) ──────────────────────
def forecast_agent(state: WeatherState) -> dict:
    print("\n📊 [Forecast Agent] Forecasting...")

    tool_choice = llm_call(f"""Tool for: "{state['user_query']}"
- "live" → current/today/right now
- "ml"   → specific numbers or retrieved conditions
Reply ONLY "ml" or "live".""")

    if "live" in tool_choice.lower():
        live = tool_live_weather()
        print(f"   {live}")
        return {"live_weather": live}
    else:
        if state.get("retrieved_docs") and state["intent"] in ("both",):
            docs_text  = "\n".join(state["retrieved_docs"])
            params_str = llm_call(f"""From records below, extract conditions of most relevant day.
Records: {docs_text}
Query: "{state['user_query']}"
Reply ONLY: day_of_year, humidity, wind_speed, pressure""")
        else:
            params_str = llm_call(f"""Extract params from: "{state['user_query']}"
Reply ONLY: day_of_year, humidity, wind_speed, pressure
Defaults if missing: 180, 60, 8, 1010""")

        print(f"   Params: {params_str}")
        try:
            p      = [x.strip() for x in params_str.split(",")]
            result = tool_ml_predict(int(float(p[0])), float(p[1]), float(p[2]), float(p[3]))
        except Exception as e:
            result = f"Prediction failed: {e}"
        print(f"   {result}")
        return {"prediction": result}


# ── Node 5: Analysis Agent (same as v4) ──────────────────────
def analysis_agent(state: WeatherState) -> dict:
    print("\n🔬 [Analysis Agent] Analysing...")
    records = tool_semantic_search(state["user_query"])
    result  = tool_trend_analysis(records)
    interp  = llm_call(f"""Interpret for user: "{state['user_query']}"
Analysis: {result}
Write 1-2 sentences with specific numbers.""")
    print(f"   {result}")
    return {"analysis": f"{result}\n\nInterpretation: {interp}"}


# ── Node 6: Responder — now with citations + explainability ──
# KEY CHANGE: the responder prompt now explicitly asks the LLM to:
#   1. Cite which specific records support each claim
#   2. Explain WHY the prediction makes sense given the data
#   3. Reference the conversation history where relevant

def responder_node(state: WeatherState) -> dict:
    print("\n✍️  [Responder] Writing cited answer...")

    history_str = format_history(state.get("conversation_history", []))

    # Build context block
    parts = []
    if state.get("live_weather"):
        parts.append(f"Live weather data:\n{state['live_weather']}")
    if state.get("retrieved_docs"):
        parts.append("Historical records retrieved:\n" + "\n".join(state["retrieved_docs"]))
    if state.get("prediction"):
        parts.append(f"ML model prediction:\n{state['prediction']}")
    if state.get("analysis"):
        parts.append(f"Statistical analysis:\n{state['analysis']}")

    context    = "\n\n".join(parts) if parts else "No data tools were used."
    citations  = state.get("citations", [])
    cite_block = "\n".join(citations) if citations else "No citations."

    # The explainability prompt — this is what makes Phase 5 different
    answer = llm_call(f"""You are a Delhi weather assistant that explains its reasoning.

CONVERSATION HISTORY (for context):
{history_str}

CURRENT QUESTION: "{state['user_query']}"

DATA RETRIEVED:
{context}

CITATIONS:
{cite_block}

Instructions:
1. Answer the question directly with specific numbers
2. Explain WHY — what in the data supports this answer
3. If you made a prediction, explain what historical patterns support it
4. Cite at least one specific date/record if data was retrieved
5. If this is a follow-up, connect it to the previous answer naturally
6. Keep it to 4-5 sentences max. Be conversational but precise.""")

    return {"final_answer": answer}


# ============================================================
# ROUTING
# ============================================================

def route_after_planner(state: WeatherState) -> str:
    intent = state["intent"]
    print(f"\n🔀 [Router] '{intent}' → ", end="")
    routes = {
        "retrieve":  "retrieval_agent",
        "forecast":  "forecast_agent",
        "both":      "retrieval_agent",
        "analysis":  "retrieval_agent",
        "live":      "forecast_agent",
        "followup":  "memory_resolver_node",  # NEW route
        "general":   "responder_node",
    }
    target = routes.get(intent, "responder_node")
    print(target)
    return target

def route_after_memory_resolver(state: WeatherState) -> str:
    """After resolving a follow-up, route based on the new intent."""
    intent = state["intent"]
    print(f"\n🔀 [Router post-resolve] '{intent}' → ", end="")
    routes = {
        "retrieve": "retrieval_agent",
        "forecast": "forecast_agent",
        "both":     "retrieval_agent",
        "analysis": "retrieval_agent",
        "live":     "forecast_agent",
        "general":  "responder_node",
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
    if state["intent"] == "analysis": return "analysis_agent"
    return "responder_node"


# ============================================================
# BUILD GRAPH
# ============================================================

print("🏗️  Building graph...")
g = StateGraph(WeatherState)

g.add_node("planner_node",        planner_node)
g.add_node("memory_resolver_node",memory_resolver_node)   # NEW
g.add_node("retrieval_agent",     retrieval_agent)
g.add_node("forecast_agent",      forecast_agent)
g.add_node("analysis_agent",      analysis_agent)
g.add_node("responder_node",      responder_node)

g.add_edge(START,              "planner_node")
g.add_edge("analysis_agent",   "responder_node")
g.add_edge("responder_node",   END)

g.add_conditional_edges("planner_node", route_after_planner, {
    "retrieval_agent":     "retrieval_agent",
    "forecast_agent":      "forecast_agent",
    "memory_resolver_node":"memory_resolver_node",
    "responder_node":      "responder_node",
})
g.add_conditional_edges("memory_resolver_node", route_after_memory_resolver, {
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
# CHAT LOOP — with persistent memory across turns
# ============================================================
# conversation_history lives HERE, outside the graph.
# It persists across all graph.invoke() calls in this session.
# Each turn: inject → invoke → update.

conversation_history: List[dict] = []   # persists across turns

def run_query(query: str) -> str:
    global conversation_history

    initial: WeatherState = {
        "user_query":           query,
        "intent":               "",
        "conversation_history": conversation_history[-6:],  # last 6 turns
        "retrieved_docs":       [],
        "prediction":           None,
        "analysis":             None,
        "live_weather":         None,
        "citations":            [],
        "final_answer":         "",
    }

    print("\n" + "─"*55)
    print(f"🌤️  {query}")
    print("─"*55)

    final = graph.invoke(initial)
    answer = final["final_answer"]

    # ── Update memory after each turn ────────────────────────
    # We store the full turn including what data was retrieved
    # so follow-up questions can reference "that day", "those conditions" etc.
    conversation_history.append({
        "user":          query,
        "assistant":     answer,
        "retrieved_docs": final.get("retrieved_docs", []),
        "prediction":    final.get("prediction"),
        "live_weather":  final.get("live_weather"),
    })

    # Keep memory bounded — max 20 turns total
    if len(conversation_history) > 20:
        conversation_history.pop(0)

    return answer


print("="*55)
print("🌤️  Weather Agent v5 — Memory + Explainability!")
print("="*55)
print("New in this version:")
print("  🧠 Memory      → follow-up questions now work!")
print("  📎 Citations   → answers cite specific records")
print("  💡 Explains    → tells you WHY, not just what")
print("\nTry this conversation flow:")
print("  1. 'What were the hottest days on record?'")
print("  2. 'What was the humidity on that day?'     ← follow-up!")
print("  3. 'Predict temperature for those conditions' ← chained!")
print("\nOther commands:")
print("  • 'memory' → show conversation history")
print("  • 'clear'  → clear memory and start fresh")
print("  • 'quit'   → exit")
print()

while True:
    user_input = input("You: ").strip()
    if not user_input:
        continue

    # Special commands
    if user_input.lower() == "quit":
        print("👋 Goodbye!")
        break

    if user_input.lower() == "memory":
        print("\n📜 Conversation History:")
        if not conversation_history:
            print("   (empty)")
        for i, turn in enumerate(conversation_history, 1):
            print(f"\n  Turn {i}:")
            print(f"    You:   {turn['user']}")
            print(f"    Agent: {turn['assistant'][:120]}...")
        print()
        continue

    if user_input.lower() == "clear":
        conversation_history.clear()
        print("🗑️  Memory cleared!\n")
        continue

    answer = run_query(user_input)
    print(f"\n💬 {answer}\n")
    print("="*55)
    time.sleep(6)