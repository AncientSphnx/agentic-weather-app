# ============================================================
# app_v2.py — Full Agentic Weather App (Phase 6)
# ============================================================
# What's new vs app.py:
#   - All original endpoints preserved (/, /api/predict, etc.)
#   - New endpoint: POST /api/chat  ← talks to your agent_v5 graph
#   - New endpoint: POST /api/chat/clear  ← clears conversation memory
#   - Serves new chat UI at /chat
#   - Per-session conversation memory (Flask sessions)
# ============================================================

from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS
import pandas as pd
import os
import json
import time
import requests as http_requests
from datetime import datetime
from typing import TypedDict, List, Optional
from dotenv import load_dotenv

# LangGraph + LangChain
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI

# RAG
import chromadb
from sentence_transformers import SentenceTransformer

# ML
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)   # needed for Flask sessions
CORS(app)

# ── Global variables ─────────────────────────────────────────
ml_model   = None
df         = None
graph      = None
embed_mdl  = None
chroma_col = None
llm        = None
model_r2   = None
model_mae  = None

# Per-session conversation memory stored server-side
# key = session_id, value = list of conversation turns
session_memories = {}

# ============================================================
# SETUP — runs once at startup
# ============================================================

def setup_all():
    global ml_model, df, graph, embed_mdl, chroma_col, llm, model_r2, model_mae

    print("🔧 Loading everything...")

    # ML model
    df = pd.read_csv("DailyDelhiClimateTest.csv")
    df["date"] = pd.to_datetime(df["date"])
    df["day_of_year"] = df["date"].dt.dayofyear
    df["month"] = df["date"].dt.month
    df.loc[df["meanpressure"] < 900, "meanpressure"] = df["meanpressure"].mean()
    X = df[["day_of_year","humidity","wind_speed","meanpressure"]]
    y = df["meantemp"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    ml_model = LinearRegression()
    ml_model.fit(X_train, y_train)
    model_r2  = round(r2_score(y_test, ml_model.predict(X_test)), 3)
    model_mae = round(mean_absolute_error(y_test, ml_model.predict(X_test)), 2)
    print(f"✅ ML model (R²={model_r2})")

    # RAG
    with open("rag_config.json") as f:
        rag_config = json.load(f)
    chroma_client = chromadb.PersistentClient(path=rag_config["db_path"])
    chroma_col    = chroma_client.get_collection(name=rag_config["collection"])
    embed_mdl     = SentenceTransformer(rag_config["embedding_model"])
    print(f"✅ RAG ({chroma_col.count()} records)")

    # LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        max_retries=3,
    )
    print("✅ LLM ready")

    # Build graph
    graph = build_graph()
    print("✅ Agent graph compiled\n")


# ============================================================
# AGENT STATE + TOOLS (from agent_v5.py — condensed)
# ============================================================

class WeatherState(TypedDict):
    user_query:           str
    intent:               str
    conversation_history: List[dict]
    retrieved_docs:       List[str]
    prediction:           Optional[str]
    analysis:             Optional[str]
    live_weather:         Optional[str]
    citations:            List[str]
    final_answer:         str

def llm_call(prompt: str) -> str:
    for attempt in range(3):
        try:
            return llm.invoke(prompt).content.strip()
        except Exception as e:
            if ("503" in str(e) or "UNAVAILABLE" in str(e)) and attempt < 2:
                time.sleep(15)
            else:
                raise

def tool_ml_predict(day, humidity, wind, pressure):
    inp = pd.DataFrame([[day, humidity, wind, pressure]],
                       columns=["day_of_year","humidity","wind_speed","meanpressure"])
    pred = ml_model.predict(inp)[0]
    if day < 60 or day > 330: season = "Winter"
    elif day < 150:            season = "Spring/Pre-monsoon"
    elif day < 270:            season = "Monsoon/Summer"
    else:                      season = "Autumn"
    return (f"ML prediction: {pred:.1f}°C | Season: {season} | "
            f"R²={model_r2}, MAE=±{model_mae}°C")

def tool_live_weather():
    try:
        url = ("https://api.open-meteo.com/v1/forecast"
               "?latitude=28.6139&longitude=77.2090"
               "&current=temperature_2m,relative_humidity_2m,"
               "wind_speed_10m,surface_pressure,weather_code"
               "&timezone=Asia/Kolkata")
        data = http_requests.get(url, timeout=10).json()["current"]
        wmo  = {0:"Clear sky",1:"Mainly clear",2:"Partly cloudy",3:"Overcast",
                45:"Foggy",61:"Light rain",63:"Moderate rain",65:"Heavy rain",
                80:"Rain showers",95:"Thunderstorm"}
        desc = wmo.get(data.get("weather_code",0),"Unknown")
        return (f"LIVE Delhi → Temp: {data['temperature_2m']}°C | "
                f"Humidity: {data['relative_humidity_2m']}% | "
                f"Wind: {data['wind_speed_10m']}km/h | "
                f"Conditions: {desc}")
    except Exception as e:
        return f"Live weather unavailable: {e}"

def tool_semantic_search(query, n=5):
    vec = embed_mdl.encode([query])[0]
    res = chroma_col.query(query_embeddings=[vec.tolist()], n_results=n)
    return res["metadatas"][0]

def tool_filter_records(min_temp=None, max_temp=None,
                        min_humidity=None, max_humidity=None, month=None):
    f = df.copy()
    if min_temp     is not None: f = f[f["meantemp"]  >= min_temp]
    if max_temp     is not None: f = f[f["meantemp"]  <= max_temp]
    if min_humidity is not None: f = f[f["humidity"]  >= min_humidity]
    if max_humidity is not None: f = f[f["humidity"]  <= max_humidity]
    if month        is not None: f = f[f["month"]     == month]
    return f[["date","meantemp","humidity","wind_speed","meanpressure"]].head(5).to_dict("records")

def tool_trend_analysis(records):
    if not records: return "No records."
    temps = [r.get("meantemp",0) for r in records]
    avg_g = df["meantemp"].mean(); std_g = df["meantemp"].std()
    avg_l = sum(temps)/len(temps); z = (avg_l-avg_g)/std_g
    if z > 2: a = "🔴 EXTREME (hot)"
    elif z > 1: a = "🟠 Warmer than normal"
    elif z < -2: a = "🔵 EXTREME (cold)"
    elif z < -1: a = "🟡 Cooler than normal"
    else: a = "🟢 Normal range"
    trend = "rising" if temps[-1]>temps[0] else "falling" if temps[-1]<temps[0] else "stable"
    return (f"Avg: {avg_l:.1f}°C (global: {avg_g:.1f}°C) | "
            f"Anomaly: {a} (z={z:.2f}) | Trend: {trend}")

def format_history(history, max_turns=6):
    if not history: return "No previous conversation."
    lines = []
    for t in history[-max_turns:]:
        lines.append(f"User: {t['user']}")
        lines.append(f"Assistant: {t['assistant']}")
        if t.get("retrieved_docs"):
            lines.append(f"[Data: {t['retrieved_docs'][0]}]")
    return "\n".join(lines)

# ── Nodes ────────────────────────────────────────────────────

def planner_node(state):
    history = format_history(state.get("conversation_history",[]))
    resp = llm_call(f"""Classify weather query. Use history for follow-ups.

HISTORY:
{history}

QUERY: "{state['user_query']}"

Intents: retrieve | forecast | both | analysis | live | followup | general
Reply ONLY the intent word.""")
    valid = {"retrieve","forecast","both","analysis","live","followup","general"}
    intent = resp.lower().strip().replace('"','')
    if intent not in valid: intent = "general"
    return {"intent": intent}

def memory_resolver_node(state):
    history = format_history(state.get("conversation_history",[]))
    resolved = llm_call(f"""Rewrite this follow-up as self-contained question using history.
HISTORY: {history}
FOLLOW-UP: "{state['user_query']}"
Write only the resolved question.""")
    reclass = llm_call(f"""Classify: "{resolved}"
Options: retrieve|forecast|both|analysis|live|general
Reply only the intent.""")
    valid = {"retrieve","forecast","both","analysis","live","general"}
    intent = reclass.lower().strip()
    if intent not in valid: intent = "retrieve"
    return {"user_query": resolved, "intent": intent}

def retrieval_agent(state):
    strategy = llm_call(f"""Search strategy for: "{state['user_query']}"
"semantic" = descriptive, "filter" = has numbers/thresholds
Reply only "semantic" or "filter".""")
    if "filter" in strategy.lower():
        try:
            p = llm_call(f"""Extract filter params from: "{state['user_query']}"
Reply ONLY as dict: {{"min_temp":null,"max_temp":null,"min_humidity":null,"max_humidity":null,"month":null}}""")
            params = eval(p.replace("null","None"))
            records = tool_filter_records(**{k:v for k,v in params.items() if v is not None})
        except Exception:
            records = tool_semantic_search(state["user_query"])
    else:
        records = tool_semantic_search(state["user_query"])

    formatted = [
        f"• {str(r.get('date',''))[:10]}: Temp={r['meantemp']:.1f}°C, "
        f"Humidity={r['humidity']:.1f}%, Wind={r['wind_speed']:.1f}km/h, "
        f"Pressure={r.get('pressure',r.get('meanpressure',0)):.1f}hPa"
        for r in records
    ]
    citations = [f"[Source {i+1}] {d}" for i,d in enumerate(formatted)]
    return {"retrieved_docs": formatted, "citations": citations}

def forecast_agent(state):
    choice = llm_call(f"""Tool for: "{state['user_query']}"
"live"=current/today, "ml"=specific numbers or retrieved conditions
Reply only "ml" or "live".""")
    if "live" in choice.lower():
        return {"live_weather": tool_live_weather()}
    else:
        if state.get("retrieved_docs") and state["intent"] == "both":
            p = llm_call(f"""From records extract most relevant day's conditions.
Records: {chr(10).join(state['retrieved_docs'])}
Query: "{state['user_query']}"
Reply only: day_of_year, humidity, wind_speed, pressure""")
        else:
            p = llm_call(f"""Extract params from: "{state['user_query']}"
Reply only: day_of_year, humidity, wind_speed, pressure
Defaults: 180,60,8,1010""")
        try:
            parts = [x.strip() for x in p.split(",")]
            result = tool_ml_predict(int(float(parts[0])),float(parts[1]),
                                     float(parts[2]),float(parts[3]))
        except Exception as e:
            result = f"Prediction failed: {e}"
        return {"prediction": result}

def analysis_agent(state):
    records = tool_semantic_search(state["user_query"])
    result  = tool_trend_analysis(records)
    interp  = llm_call(f"""Interpret for: "{state['user_query']}"
Analysis: {result}
1-2 sentences with numbers.""")
    return {"analysis": f"{result}\n{interp}"}

def responder_node(state):
    history = format_history(state.get("conversation_history",[]))
    parts   = []
    if state.get("live_weather"):   parts.append(f"Live weather:\n{state['live_weather']}")
    if state.get("retrieved_docs"): parts.append("Records:\n" + "\n".join(state["retrieved_docs"]))
    if state.get("prediction"):     parts.append(f"ML prediction:\n{state['prediction']}")
    if state.get("analysis"):       parts.append(f"Analysis:\n{state['analysis']}")
    context = "\n\n".join(parts) if parts else "No data."
    citations = "\n".join(state.get("citations",[])) or "None."

    answer = llm_call(f"""Delhi weather assistant. Answer with citations and reasoning.

HISTORY: {history}
QUESTION: "{state['user_query']}"
DATA: {context}
CITATIONS: {citations}

Rules: cite records, explain why, 3-4 sentences, be precise.""")
    return {"final_answer": answer}

# ── Routing ──────────────────────────────────────────────────

def route_after_planner(state):
    routes = {"retrieve":"retrieval_agent","forecast":"forecast_agent",
              "both":"retrieval_agent","analysis":"retrieval_agent",
              "live":"forecast_agent","followup":"memory_resolver_node",
              "general":"responder_node"}
    return routes.get(state["intent"],"responder_node")

def route_after_resolver(state):
    routes = {"retrieve":"retrieval_agent","forecast":"forecast_agent",
              "both":"retrieval_agent","analysis":"retrieval_agent",
              "live":"forecast_agent","general":"responder_node"}
    return routes.get(state["intent"],"responder_node")

def route_after_retrieval(state):
    if state["intent"] == "both":     return "forecast_agent"
    if state["intent"] == "analysis": return "analysis_agent"
    return "responder_node"

def route_after_forecast(state):
    return "analysis_agent" if state["intent"] == "analysis" else "responder_node"

def build_graph():
    g = StateGraph(WeatherState)
    g.add_node("planner_node",         planner_node)
    g.add_node("memory_resolver_node", memory_resolver_node)
    g.add_node("retrieval_agent",      retrieval_agent)
    g.add_node("forecast_agent",       forecast_agent)
    g.add_node("analysis_agent",       analysis_agent)
    g.add_node("responder_node",       responder_node)
    g.add_edge(START, "planner_node")
    g.add_edge("analysis_agent", "responder_node")
    g.add_edge("responder_node", END)
    g.add_conditional_edges("planner_node", route_after_planner,
        {"retrieval_agent":"retrieval_agent","forecast_agent":"forecast_agent",
         "memory_resolver_node":"memory_resolver_node","responder_node":"responder_node"})
    g.add_conditional_edges("memory_resolver_node", route_after_resolver,
        {"retrieval_agent":"retrieval_agent","forecast_agent":"forecast_agent","responder_node":"responder_node"})
    g.add_conditional_edges("retrieval_agent", route_after_retrieval,
        {"forecast_agent":"forecast_agent","analysis_agent":"analysis_agent","responder_node":"responder_node"})
    g.add_conditional_edges("forecast_agent", route_after_forecast,
        {"analysis_agent":"analysis_agent","responder_node":"responder_node"})
    return g.compile()


# ============================================================
# FLASK ROUTES — original + new chat endpoints
# ============================================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat')
def chat_page():
    """New chat UI page."""
    return render_template('chat.html')

# ── Original endpoints (unchanged) ──────────────────────────

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data       = request.get_json()
        date_str   = data.get('date')
        humidity   = float(data.get('humidity'))
        wind_speed = float(data.get('wind_speed'))
        pressure   = float(data.get('pressure'))
        date_obj   = datetime.strptime(date_str, '%Y-%m-%d')
        day_of_year= date_obj.timetuple().tm_yday
        inp        = pd.DataFrame([[day_of_year,humidity,wind_speed,pressure]],
                                  columns=['day_of_year','humidity','wind_speed','meanpressure'])
        pred_temp  = ml_model.predict(inp)[0]
        conditions = {range(0,10):"🧥 Cold",range(10,20):"🌤️ Cool",
                      range(20,30):"☀️ Warm",range(30,35):"🔥 Hot"}
        condition  = next((v for k,v in conditions.items() if int(pred_temp) in k), "🌋 Very Hot")
        return jsonify({"temperature": round(float(pred_temp),2),
                        "weather_condition": condition,
                        "inputs": {"day_of_year":day_of_year,"humidity":humidity,
                                   "wind_speed":wind_speed,"pressure":pressure}})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/model-info', methods=['GET'])
def model_info():
    from sklearn.metrics import mean_squared_error, r2_score as r2
    X = df[['day_of_year','humidity','wind_speed','meanpressure']]
    y_true = df['meantemp']
    y_pred = ml_model.predict(X)
    return jsonify({
        'algorithm': 'Linear Regression',
        'metrics': {'r2_score': round(float(r2(y_true,y_pred)),3),
                    'mse': round(float(mean_squared_error(y_true,y_pred)),3)},
        'dataset_info': {'total_samples': len(df),
                         'date_range': f"{df['date'].min().date()} to {df['date'].max().date()}"}
    })

@app.route('/api/historical-data', methods=['GET'])
def historical_data():
    chart = df.iloc[::7].copy()
    data  = [{'date': r['date'].strftime('%Y-%m-%d'),
               'actual_temp': float(r['meantemp']),
               'humidity': float(r['humidity'])}
              for _,r in chart.iterrows()]
    return jsonify({'data': data, 'total_points': len(data)})

# ── NEW: Chat endpoints ──────────────────────────────────────

@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Main chat endpoint — runs user query through the agent graph.
    Uses session ID to maintain per-user conversation memory.
    """
    try:
        data       = request.get_json()
        user_query = data.get('message', '').strip()
        session_id = data.get('session_id', 'default')

        if not user_query:
            return jsonify({'error': 'No message provided'}), 400

        # Get or create memory for this session
        if session_id not in session_memories:
            session_memories[session_id] = []
        history = session_memories[session_id]

        # Build initial state
        initial: WeatherState = {
            "user_query":           user_query,
            "intent":               "",
            "conversation_history": history[-6:],
            "retrieved_docs":       [],
            "prediction":           None,
            "analysis":             None,
            "live_weather":         None,
            "citations":            [],
            "final_answer":         "",
        }

        # Run graph
        final = graph.invoke(initial)
        answer = final["final_answer"]

        # Update session memory
        history.append({
            "user":          user_query,
            "assistant":     answer,
            "retrieved_docs": final.get("retrieved_docs",[]),
            "prediction":    final.get("prediction"),
        })
        if len(history) > 20:
            history.pop(0)

        return jsonify({
            "answer":        answer,
            "intent":        final.get("intent",""),
            "retrieved_docs": final.get("retrieved_docs",[]),
            "prediction":    final.get("prediction"),
            "live_weather":  final.get("live_weather"),
            "citations":     final.get("citations",[]),
        })

    except Exception as e:
        return jsonify({'error': f'Agent error: {str(e)}'}), 500

@app.route('/api/chat/clear', methods=['POST'])
def chat_clear():
    """Clear conversation memory for a session."""
    data       = request.get_json() or {}
    session_id = data.get('session_id', 'default')
    if session_id in session_memories:
        session_memories[session_id].clear()
    return jsonify({'status': 'cleared'})

@app.route('/api/chat/history', methods=['GET'])
def chat_history():
    """Get conversation history for a session."""
    session_id = request.args.get('session_id', 'default')
    history    = session_memories.get(session_id, [])
    return jsonify({'history': history, 'turn_count': len(history)})


# ============================================================
# RUN
# ============================================================

if __name__ == '__main__':
    print("🌤️  Starting Agentic Weather App...")
    setup_all()
    print("📡 Endpoints:")
    print("   GET  /          → Original weather UI")
    print("   GET  /chat      → New AI chat interface")
    print("   POST /api/chat  → Agent endpoint")
    print("   POST /api/predict, GET /api/model-info (original)")
    print("\n🚀 http://localhost:5000\n")
    app.run(debug=False, host='0.0.0.0', port=5000)