# ============================================================
# agent_v3.py — Weather Agent with LangGraph (Phase 3)
# ============================================================
# What's new vs agent_v2.py:
#   - Replaced black-box ReAct loop with an explicit state machine
#   - Every step is a named node you can see and control
#   - State flows between nodes as a typed Python dict
#   - Router decides which node runs next based on query intent
#   - Much easier to debug, extend, and explain in interviews
#
# Graph structure:
#   [START] → planner → router ──► retrieval_node ──► responder → [END]
#                               ──► forecast_node  ──► responder → [END]
#                               ──► both_nodes     ──► responder → [END]
# ============================================================

import os
import json
import time
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from typing import TypedDict, List, Optional

# LangGraph imports
from langgraph.graph import StateGraph, START, END

# LangChain LLM
from langchain_google_genai import ChatGoogleGenerativeAI

# RAG imports
import chromadb
from sentence_transformers import SentenceTransformer

# ML imports
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

load_dotenv()

# ── Setup: ML model ─────────────────────────────────────────
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
model_r2  = round(r2_score(y_test, ml_model.predict(X_test)), 3)
model_mae = round(mean_absolute_error(y_test, ml_model.predict(X_test)), 2)
print(f"✅ ML model ready! R²={model_r2}, MAE=±{model_mae}°C")

# ── Setup: RAG ───────────────────────────────────────────────
print("📚 Loading RAG knowledge base...")
with open("rag_config.json") as f:
    rag_config = json.load(f)
chroma_client = chromadb.PersistentClient(path=rag_config["db_path"])
collection    = chroma_client.get_collection(name=rag_config["collection"])
embed_model   = SentenceTransformer(rag_config["embedding_model"])
print(f"✅ RAG ready! {collection.count()} records loaded.")

# ── Setup: LLM ───────────────────────────────────────────────
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# ============================================================
# STEP 1: Define the State
# ============================================================
# The State is a TypedDict — a typed Python dict that flows through
# every node in the graph. Every node receives the full state,
# does its work, and returns ONLY the keys it updated.
# LangGraph merges those updates back into the shared state.
#
# Think of it as a baton in a relay race — passed node to node,
# each runner adds their contribution before passing it on.

class WeatherState(TypedDict):
    user_query:     str            # Original question from user (never changes)
    intent:         str            # What the planner decided: "retrieve", "forecast", "both", "general"
    retrieved_docs: List[str]      # Results from the RAG knowledge base
    prediction:     Optional[str]  # Result from the ML forecast tool
    final_answer:   str            # The answer shown to the user

# ============================================================
# STEP 2: Define the Nodes
# ============================================================
# Each node is a plain Python function that:
#   - Takes the full state as input
#   - Returns a dict of ONLY the keys it wants to update
# LangGraph handles the merging automatically.

# ── Node 1: Planner ─────────────────────────────────────────
# The planner's only job: classify the user's intent.
# It uses the LLM to decide which tool(s) are needed.
# This replaces the ad-hoc "which tool should I call?" guessing
# in the ReAct loop with an explicit, inspectable decision.

def planner_node(state: WeatherState) -> dict:
    """Classify the user's query intent using the LLM."""
    print("\n📋 [Planner] Classifying intent...")

    prompt = f"""You are a query classifier for a weather AI system.
Classify the user's query into exactly one of these intents:

- "retrieve"  → user wants historical data, past records, similar days, patterns
- "forecast"  → user wants a temperature prediction (will give specific numbers: day, humidity, wind, pressure)
- "both"      → user wants historical data AND a prediction (e.g. "find coldest day then predict")
- "general"   → general weather knowledge question, no data lookup needed

User query: "{state['user_query']}"

Reply with ONLY the intent word. No explanation."""

    response = llm.invoke(prompt)
    # Clean up the response — LLM sometimes adds quotes or spaces
    intent = response.content.strip().lower().replace('"', '').replace("'", "")

    # Validate — default to "general" if LLM returns something unexpected
    valid_intents = {"retrieve", "forecast", "both", "general"}
    if intent not in valid_intents:
        intent = "general"

    print(f"   Intent: '{intent}'")
    return {"intent": intent}  # Only update the "intent" key in state


# ── Node 2: Retrieval ────────────────────────────────────────
# Searches the ChromaDB vector store for relevant historical records.
# Only runs when intent is "retrieve" or "both".

def retrieval_node(state: WeatherState) -> dict:
    """Search the RAG knowledge base for relevant climate records."""
    print("\n🔍 [Retrieval] Searching knowledge base...")

    query_embedding = embed_model.encode([state["user_query"]])[0]
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=5
    )

    docs      = results["documents"][0]
    metadatas = results["metadatas"][0]

    # Format results into clean text for the responder to use
    formatted = []
    for doc, meta in zip(docs, metadatas):
        formatted.append(
            f"• {meta['date']}: Temp={meta['meantemp']:.1f}°C, "
            f"Humidity={meta['humidity']:.1f}%, "
            f"Wind={meta['wind_speed']:.1f}km/h, "
            f"Pressure={meta['pressure']:.1f}hPa"
        )

    retrieved_text = "\n".join(formatted)
    print(f"   Found {len(docs)} matching records")
    return {"retrieved_docs": formatted}


# ── Node 3: Forecast ─────────────────────────────────────────
# Runs the ML model to make a temperature prediction.
# If called after retrieval (intent="both"), it extracts the
# conditions from retrieved_docs automatically.

def forecast_node(state: WeatherState) -> dict:
    """Run the ML model to predict temperature."""
    print("\n📊 [Forecast] Running ML prediction...")

    # Case 1: intent is "both" — extract conditions from retrieved docs
    # The LLM figures out which day's conditions to use
    if state["intent"] == "both" and state.get("retrieved_docs"):
        docs_text = "\n".join(state["retrieved_docs"])

        prompt = f"""From these climate records, extract the conditions of the MOST RELEVANT day
(e.g. coldest, hottest, or most similar to what was asked).

Records:
{docs_text}

Original query: "{state['user_query']}"

Reply with ONLY a comma-separated line in this exact format:
day_of_year, humidity, wind_speed, pressure

Example: 40, 68.4, 7.9, 1016.4
No explanation, no other text."""

        response = llm.invoke(prompt)
        params_str = response.content.strip()
        print(f"   Extracted params: {params_str}")

    # Case 2: intent is "forecast" — parse directly from user query
    else:
        prompt = f"""Extract weather parameters from this query for temperature prediction.

Query: "{state['user_query']}"

Reply with ONLY a comma-separated line in this exact format:
day_of_year, humidity, wind_speed, pressure

If any value is missing, use these Delhi averages: day_of_year=180, humidity=60, wind_speed=8, pressure=1010
No explanation, no other text."""

        response = llm.invoke(prompt)
        params_str = response.content.strip()
        print(f"   Extracted params: {params_str}")

    # Run the actual ML model
    try:
        parts      = [p.strip() for p in params_str.split(",")]
        day        = int(float(parts[0]))
        humidity   = float(parts[1])
        wind       = float(parts[2])
        pressure   = float(parts[3])

        input_df   = pd.DataFrame(
            [[day, humidity, wind, pressure]],
            columns=["day_of_year", "humidity", "wind_speed", "meanpressure"]
        )
        predicted  = ml_model.predict(input_df)[0]

        if day < 60 or day > 330:       season = "Winter"
        elif day < 150:                  season = "Spring/Pre-monsoon"
        elif day < 270:                  season = "Monsoon/Summer"
        else:                            season = "Autumn"

        result = (
            f"Predicted temperature: {predicted:.1f}°C | "
            f"Season: {season} | "
            f"Inputs: day={day}, humidity={humidity}%, wind={wind}km/h, pressure={pressure}hPa | "
            f"Model: R²={model_r2}, MAE=±{model_mae}°C"
        )
    except Exception as e:
        result = f"Prediction failed: {e}"

    print(f"   Result: {result}")
    return {"prediction": result}


# ── Node 4: Responder ────────────────────────────────────────
# The final node. Always runs last.
# Takes everything in the state and uses the LLM to write
# a clear, grounded final answer for the user.

def responder_node(state: WeatherState) -> dict:
    """Generate the final answer using all gathered context."""
    print("\n✍️  [Responder] Writing final answer...")

    # Build a context block from whatever data was collected
    context_parts = []
    if state.get("retrieved_docs"):
        context_parts.append("Historical records found:\n" + "\n".join(state["retrieved_docs"]))
    if state.get("prediction"):
        context_parts.append("ML prediction result:\n" + state["prediction"])

    context = "\n\n".join(context_parts) if context_parts else "No data tools were used."

    prompt = f"""You are a helpful weather assistant for Delhi climate data.
Answer the user's question clearly and concisely using the provided context.
Always cite specific dates/numbers from the context.
If a prediction was made, explain what it means in plain language.

User question: "{state['user_query']}"

Context:
{context}

Write a clear, friendly answer in 2-4 sentences."""

    response = llm.invoke(prompt)
    answer = response.content.strip()
    return {"final_answer": answer}


# ============================================================
# STEP 3: Define the Router
# ============================================================
# A router is a special function that returns the NAME of the
# next node to run. LangGraph uses this to decide which edge
# to follow after the planner.
#
# This is a "conditional edge" — the path through the graph
# changes based on the state. This is impossible in ReAct.

def route_after_planner(state: WeatherState) -> str:
    """Decide which node runs after the planner based on intent."""
    intent = state["intent"]
    print(f"\n🔀 [Router] Intent='{intent}' → routing to: ", end="")

    if intent == "retrieve":
        print("retrieval_node")
        return "retrieval_node"
    elif intent == "forecast":
        print("forecast_node")
        return "forecast_node"
    elif intent == "both":
        print("retrieval_node (then forecast)")
        return "retrieval_node"   # retrieval runs first, then forecast
    else:  # "general"
        print("responder_node (direct)")
        return "responder_node"


def route_after_retrieval(state: WeatherState) -> str:
    """After retrieval: go to forecast if intent is 'both', else go to responder."""
    if state["intent"] == "both":
        return "forecast_node"
    return "responder_node"


# ============================================================
# STEP 4: Build the Graph
# ============================================================
# Now we wire everything together.
# add_node() registers a function as a named node.
# add_edge() creates a fixed connection (always goes A → B).
# add_conditional_edges() creates a router (goes to different nodes based on logic).

print("\n🏗️  Building LangGraph...")

graph_builder = StateGraph(WeatherState)

# Register all nodes
graph_builder.add_node("planner_node",   planner_node)
graph_builder.add_node("retrieval_node", retrieval_node)
graph_builder.add_node("forecast_node",  forecast_node)
graph_builder.add_node("responder_node", responder_node)

# Fixed edges (always happen)
graph_builder.add_edge(START, "planner_node")          # Entry point
graph_builder.add_edge("responder_node", END)          # Exit point

# Conditional edge after planner — router decides where to go
graph_builder.add_conditional_edges(
    "planner_node",           # After this node...
    route_after_planner,      # ...call this function to decide the next node
    {                         # Map return values to node names
        "retrieval_node":  "retrieval_node",
        "forecast_node":   "forecast_node",
        "responder_node":  "responder_node",
    }
)

# Conditional edge after retrieval — go to forecast or responder
graph_builder.add_conditional_edges(
    "retrieval_node",
    route_after_retrieval,
    {
        "forecast_node":  "forecast_node",
        "responder_node": "responder_node",
    }
)

# Fixed edge: forecast always goes to responder
graph_builder.add_edge("forecast_node", "responder_node")

# Compile the graph — this validates all connections and creates the runnable
graph = graph_builder.compile()
print("✅ Graph compiled!")

# ============================================================
# STEP 5: Chat loop
# ============================================================

def run_query(query: str) -> str:
    """Run a query through the LangGraph state machine."""
    # Initial state — only user_query is set, everything else is empty
    initial_state: WeatherState = {
        "user_query":     query,
        "intent":         "",
        "retrieved_docs": [],
        "prediction":     None,
        "final_answer":   "",
    }

    print("\n" + "─"*55)
    print(f"🌤️  Query: {query}")
    print("─"*55)

    # graph.invoke() runs the state through the graph until END
    # It returns the FINAL state after all nodes have run
    final_state = graph.invoke(initial_state)

    return final_state["final_answer"]


print("\n" + "="*55)
print("🌤️  Weather Agent v3 — LangGraph Powered!")
print("="*55)
print("Nodes: Planner → Router → Retrieval / Forecast → Responder")
print("\nTry these:")
print("  • What were the hottest days on record?")
print("  • Predict temp for day 150, humidity 55, wind 12, pressure 1008")
print("  • Find the coldest day and predict temperature for those conditions")
print("  • How does monsoon season affect Delhi temperatures?")
print("\nType 'quit' to exit.\n")

while True:
    user_input = input("You: ").strip()
    if not user_input:
        continue
    if user_input.lower() in ["quit", "exit", "q"]:
        print("👋 Goodbye!")
        break

    answer = run_query(user_input)
    print(f"\n💬 Answer: {answer}\n")
    print("="*55)
    time.sleep(5)