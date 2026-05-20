# weather-agent

A full-stack **Agentic AI + RAG** weather prediction system built on Delhi climate data. Evolved from a simple Linear Regression model into a multi-agent LangGraph system with semantic search, live weather, conversation memory, and a chat UI.

## What this project is

Most ML weather apps are one-shot: input numbers → get a prediction. This project replaces that with an agent that **reasons**, **retrieves**, and **explains** — choosing the right tool for each query, searching historical records semantically, and remembering context across conversation turns.

## Architecture

```
User query
    ↓
Planner node        — classifies intent (retrieve / forecast / analysis / live / followup)
    ↓
Router              — conditional edges based on intent
    ↓
┌─────────────────────────────────────────┐
│  Retrieval Agent   → ChromaDB semantic  │
│  Forecast Agent    → ML model / API     │
│  Analysis Agent    → z-score / trends   │
│  Memory Resolver   → resolves follow-ups│
└─────────────────────────────────────────┘
    ↓
Responder node      — cited, explained answer
```

## Features

- **Multi-agent LangGraph graph** — explicit state machine with conditional routing
- **RAG knowledge base** — 114 days of Delhi climate data embedded in ChromaDB, searchable by natural language
- **Dual forecast tools** — Linear Regression ML model + live Open-Meteo API (no key needed)
- **Anomaly detection** — z-score analysis against dataset averages with trend direction
- **Conversation memory** — sliding window of last 6 turns, follow-up questions work naturally
- **Explainability** — every answer cites specific historical records that support it
- **Chat UI** — dark weather-themed interface with intent badges, citations panel, suggestion chips
- **Original ML UI** — original prediction form preserved at `/`

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt
pip install langchain langchain-google-genai langgraph chromadb sentence-transformers python-dotenv requests

# 2. Add your Gemini API key (free at aistudio.google.com)
echo "GOOGLE_API_KEY=your_key_here" > .env

# 3. Build the RAG knowledge base (run once)
python rag_setup.py

# 4. Start the server
python app_v2.py
```

Open `http://localhost:5000/chat` for the AI chat interface.  
Open `http://localhost:5000` for the original ML prediction form.

## Project Structure

```
├── app_v2.py               # Full Flask app — original endpoints + /api/chat
├── app.py                  # Original Flask app (preserved)
├── wheater.py              # Original ML training script
├── rag_setup.py            # Builds ChromaDB vector knowledge base (run once)
├── rag_config.json         # RAG configuration
├── agent_v1.py             # Phase 1: first ReAct agent
├── agent_v2.py             # Phase 2: agent + RAG search tool
├── agent_v3.py             # Phase 3: LangGraph state machine
├── agent_v4.py             # Phase 4: specialist agents + live weather API
├── agent_v5.py             # Phase 5: memory + explainability
├── climate_db/             # ChromaDB vector database (auto-generated)
├── DailyDelhiClimateTest.csv
├── requirements.txt
└── templates/
    ├── index.html          # Original weather UI
    └── chat.html           # Agentic chat UI
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Original ML prediction UI |
| GET | `/chat` | Agentic chat UI |
| POST | `/api/chat` | Agent endpoint — runs query through LangGraph |
| POST | `/api/chat/clear` | Clear conversation memory for session |
| GET | `/api/chat/history` | Get conversation history |
| POST | `/api/predict` | Direct ML prediction (original) |
| GET | `/api/model-info` | Model metrics |
| GET | `/api/historical-data` | Historical chart data |

## Example queries

```
"What's the weather in Delhi right now?"
"Find the hottest days on record"
"Predict temperature for day 200, humidity 60, wind 8, pressure 1010"
"Is the temperature in the dataset unusually high?"
"Find days where humidity was below 30%"

# Follow-up memory:
"What were the coldest days?"
"What was the humidity on that day?"       ← agent remembers context
"Predict temperature for those conditions" ← chained reasoning
```

## How it was built — phases

| Phase | What was added |
|---|---|
| 1 | ReAct agent wrapping LinearRegression as a LangChain tool |
| 2 | RAG knowledge base — ChromaDB + sentence-transformers embeddings |
| 3 | LangGraph state machine replacing the black-box ReAct loop |
| 4 | Specialist agents with tool selection + live Open-Meteo API |
| 5 | Sliding window conversation memory + citation-based explainability |
| 6 | Flask integration + dark chat UI with intent badges |

## ML Model

- **Algorithm**: Linear Regression (scikit-learn)
- **Features**: day of year, humidity, wind speed, mean pressure
- **Dataset**: Delhi daily climate data (114 days)
- **Performance**: R²=0.817, MAE=±2.24°C

## Stack

**Backend**: Python, Flask, LangGraph, LangChain, scikit-learn  
**AI**: Gemini 2.5 Flash Lite, sentence-transformers (all-MiniLM-L6-v2)  
**Vector DB**: ChromaDB  
**Live data**: Open-Meteo API  
**Frontend**: HTML, CSS, JavaScript