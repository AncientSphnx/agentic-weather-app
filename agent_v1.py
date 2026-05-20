# ============================================================
# agent_v1.py — Your First Weather Agent (Phase 1)
# ============================================================
# What this file teaches you:
#   1. How to wrap an ML model as a Tool an LLM can call
#   2. How the ReAct loop works (Reason → Act → Observe)
#   3. How an agent decides WHEN to call a tool vs answer directly
# ============================================================

import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# LangChain imports — broken down so you know what each does
from langchain_google_genai import ChatGoogleGenerativeAI  # The LLM brain (Gemini)
from langchain.agents import create_react_agent, AgentExecutor  # The ReAct loop engine
from langchain.tools import tool                                 # Decorator to turn a function into a Tool
from langchain import hub                                        # Pulls the ReAct prompt template from LangChain hub

# ML imports (your existing model)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# ── Step 1: Load environment variables (.env file) ──────────
# This reads your GOOGLE_API_KEY from the .env file
# Never hardcode API keys in your source code
load_dotenv()

# ── Step 2: Train your ML model once at startup ─────────────
# We train it here so it's ready before any agent call
# Think of this as "loading your tool" before handing it to the agent

print("🔧 Training ML model...")

df = pd.read_csv("DailyDelhiClimateTest.csv")
df["date"] = pd.to_datetime(df["date"])
df["day_of_year"] = df["date"].dt.dayofyear
df.loc[df["meanpressure"] < 900, "meanpressure"] = df["meanpressure"].mean()

X = df[["day_of_year", "humidity", "wind_speed", "meanpressure"]]
y = df["meantemp"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

ml_model = LinearRegression()
ml_model.fit(X_train, y_train)

y_pred = ml_model.predict(X_test)
model_r2  = round(r2_score(y_test, y_pred), 3)
model_mae = round(mean_absolute_error(y_test, y_pred), 2)

print(f"✅ Model ready! R²={model_r2}, MAE=±{model_mae}°C")

# ── Step 3: Define Tools ─────────────────────────────────────
# A Tool = a Python function the LLM is ALLOWED to call
# The @tool decorator + docstring is what the LLM reads to decide
# WHETHER and HOW to use this function.
# The docstring is literally the LLM's instruction manual for the tool.

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
    # Parse the comma-separated input string
    # This is needed because ReAct agents pass tool input as a single string
    try:
        parts = [p.strip() for p in input_str.split(",")]
        if len(parts) != 4:
            return "Error: provide exactly 4 values as: day_of_year, humidity, wind_speed, pressure"
        day_of_year = int(float(parts[0]))
        humidity    = float(parts[1])
        wind_speed  = float(parts[2])
        pressure    = float(parts[3])
    except ValueError:
        return "Error: all values must be numbers. Format: day_of_year, humidity, wind_speed, pressure"

    # Validate inputs before running the model
    if not (1 <= day_of_year <= 365):
        return "Error: day_of_year must be between 1 and 365."
    if not (0 <= humidity <= 100):
        return "Error: humidity must be between 0 and 100."
    if not (0 <= wind_speed <= 50):
        return "Error: wind_speed must be between 0 and 50."
    if not (900 <= pressure <= 1050):
        return "Error: pressure must be between 900 and 1050."

    # Run the ML model
    input_df = pd.DataFrame(
        [[day_of_year, humidity, wind_speed, pressure]],
        columns=["day_of_year", "humidity", "wind_speed", "meanpressure"]
    )
    predicted_temp = ml_model.predict(input_df)[0]

    # Determine season from day_of_year (Delhi seasons)
    if day_of_year < 60 or day_of_year > 330:
        season = "Winter"
    elif day_of_year < 150:
        season = "Spring/Pre-monsoon"
    elif day_of_year < 270:
        season = "Monsoon/Summer"
    else:
        season = "Autumn/Post-monsoon"

    return (
        f"Predicted temperature: {predicted_temp:.1f}°C | "
        f"Season: {season} | "
        f"Model accuracy: R²={model_r2}, MAE=±{model_mae}°C"
    )


@tool
def get_dataset_stats(query: str) -> str:
    """
    Returns statistics and facts about the Delhi climate dataset.
    Use this when the user asks about historical data, averages, records,
    or anything about the dataset itself.

    Args:
        query: What the user wants to know about the dataset (e.g. 'hottest month', 'average humidity')
    """
    stats = {
        "total_days": len(df),
        "date_range": f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}",
        "avg_temp": round(df["meantemp"].mean(), 1),
        "max_temp": round(df["meantemp"].max(), 1),
        "min_temp": round(df["meantemp"].min(), 1),
        "avg_humidity": round(df["humidity"].mean(), 1),
        "avg_wind": round(df["wind_speed"].mean(), 1),
        "hottest_day": df.loc[df["meantemp"].idxmax(), "date"].strftime("%Y-%m-%d"),
        "coldest_day": df.loc[df["meantemp"].idxmin(), "date"].strftime("%Y-%m-%d"),
    }

    # Monthly averages — useful for "what's the hottest month" type questions
    df["month"] = df["date"].dt.month
    monthly = df.groupby("month")["meantemp"].mean().round(1).to_dict()
    month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                   7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    monthly_str = ", ".join([f"{month_names[m]}: {t}°C" for m, t in monthly.items()])

    return (
        f"Dataset: {stats['total_days']} days ({stats['date_range']}) | "
        f"Temp range: {stats['min_temp']}°C to {stats['max_temp']}°C | "
        f"Avg temp: {stats['avg_temp']}°C | Avg humidity: {stats['avg_humidity']}% | "
        f"Hottest day: {stats['hottest_day']} | Coldest day: {stats['coldest_day']} | "
        f"Monthly avg temps → {monthly_str}"
    )


# ── Step 4: Create the LLM (the brain) ──────────────────────
# This is the Gemini model that does all the reasoning
# temperature=0 means deterministic — same input, same output
# (higher temperature = more creative/random)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# ── Step 5: Bundle tools into a list ────────────────────────
# The agent will only have access to tools in this list
# This is how you control what the agent CAN and CANNOT do
tools = [predict_temperature, get_dataset_stats]

# ── Step 6: Load the ReAct prompt ───────────────────────────
# This is a pre-built prompt from LangChain that teaches the LLM
# HOW to do the Reason → Act → Observe loop
# It tells the LLM: "Think step by step, pick a tool, observe the result, repeat"
prompt = hub.pull("hwchase17/react")

# ── Step 7: Create the agent ────────────────────────────────
# create_react_agent wires the LLM + tools + prompt together
# AgentExecutor is the loop runner — it keeps running until the agent says "Final Answer"
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,      # ← IMPORTANT: set True so you can SEE the ReAct loop happen
    max_iterations=5,  # Safety limit — stops infinite loops
    handle_parsing_errors=True
)

# ── Step 8: Chat loop ────────────────────────────────────────
# Simple terminal chat so you can talk to your agent right now
print("\n" + "="*55)
print("🌤️  Weather Agent v1 — Phase 1 Complete!")
print("="*55)
print("Ask me anything about Delhi weather.")
print("Examples:")
print("  • What will the temperature be on day 200 with 65% humidity, wind 10, pressure 1010?")
print("  • What is the hottest month in Delhi?")
print("  • Is 38°C normal for Delhi in summer?")
print("Type 'quit' to exit.\n")

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