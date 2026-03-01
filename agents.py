"""
Multi-Agent Intent-Based AI Assistant
Stack: Groq + LLaMA3.3 | LangGraph | Tavily | OpenWeatherMap | ChromaDB | Gradio

Streaming:
  stream_query() uses LangGraph stream_mode=["updates","messages"]
    - "updates"  → one event per completed node  → live status lines
    - "messages" → LLM token chunks from analysis_agent → token-by-token answer
"""

import os
import json
import re
from typing import TypedDict, Optional, List, Generator
from datetime import datetime

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from tavily import TavilyClient
from pyowm import OWM
import chromadb
from chromadb.config import Settings

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════
GROQ_API_KEY        = os.getenv("GROQ_API_KEY",        "xxxx")
TAVILY_API_KEY      = os.getenv("TAVILY_API_KEY",      "xxxx")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "xxxx")

# Node name used for the analysis LLM — used to filter "messages" stream events
ANALYSIS_NODE = "analysis"


# ══════════════════════════════════════════════════════════════════════════════
#  PYDANTIC SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════
class SubTask(BaseModel):
    task_id: int
    description: str
    agent: str
    condition: Optional[str] = None

class IntentPlan(BaseModel):
    city: Optional[str] = None
    needs_city: bool = False
    tasks: List[SubTask] = Field(default_factory=list)
    summary: str = ""
    budget_min: Optional[float] = None
    budget_max: Optional[float] = None


# ══════════════════════════════════════════════════════════════════════════════
#  GRAPH STATE
# ══════════════════════════════════════════════════════════════════════════════
class AgentState(TypedDict):
    user_query:           str
    city:                 Optional[str]
    needs_city:           bool
    intent_plan:          Optional[dict]
    weather_result:       Optional[str]
    outdoor_result:       Optional[str]
    entertainment_result: Optional[str]
    food_result:          Optional[str]
    general_result:       Optional[str]
    price_result:         Optional[str]
    final_answer:         Optional[str]
    memory_context:       Optional[str]
    active_tasks:         List[str]
    weather_condition:    Optional[str]
    budget_min:           Optional[float]
    budget_max:           Optional[float]


# ══════════════════════════════════════════════════════════════════════════════
#  LLM
#  analysis_agent uses streaming=True so LangGraph "messages" mode captures
#  its tokens automatically — no change needed in the node itself.
# ══════════════════════════════════════════════════════════════════════════════
def get_llm(temperature: float = 0.3, streaming: bool = False):
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile",
        temperature=temperature,
        streaming=streaming,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  CHROMADB MEMORY
# ══════════════════════════════════════════════════════════════════════════════
chroma_client     = chromadb.Client(Settings(anonymized_telemetry=False))
memory_collection = chroma_client.get_or_create_collection("user_memory")

def save_to_memory(query: str, result: str, city: str = ""):
    doc_id = f"{datetime.now().isoformat()}_{hash(query) % 10000}"
    memory_collection.add(
        documents=[f"Query: {query}\nCity: {city}\nResult: {result[:500]}"],
        ids=[doc_id],
        metadatas=[{"city": city, "timestamp": datetime.now().isoformat()}]
    )

def recall_from_memory(query: str, n_results: int = 3) -> str:
    try:
        results = memory_collection.query(
            query_texts=[query],
            n_results=min(n_results, memory_collection.count())
        )
        if results["documents"] and results["documents"][0]:
            return "\n---\n".join(results["documents"][0])
    except Exception:
        pass
    return ""


# ══════════════════════════════════════════════════════════════════════════════
#  TOOLS
# ══════════════════════════════════════════════════════════════════════════════
def get_weather(city: str) -> dict:
    try:
        owm = OWM(OPENWEATHER_API_KEY)
        mgr = owm.weather_manager()
        obs = mgr.weather_at_place(city)
        w   = obs.weather
        return {
            "status":     w.detailed_status,
            "temp_c":     w.temperature("celsius")["temp"],
            "feels_like": w.temperature("celsius")["feels_like"],
            "humidity":   w.humidity,
            "wind_kmh":   round(w.wind()["speed"] * 3.6, 1),
        }
    except Exception as e:
        return {"error": str(e), "status": "unknown"}


def tavily_search(query: str, max_results: int = 5) -> tuple[str, list[dict]]:
    try:
        client   = TavilyClient(api_key=TAVILY_API_KEY)
        response = client.search(query, search_depth="advanced", max_results=max_results)
        results  = response.get("results", [])
        parts = []
        for r in results[:max_results]:
            title   = r.get("title", "")
            content = r.get("content", "")
            url     = r.get("url", "")
            parts.append(f"### {title}\n{content}\nSource: {url}")
        return "\n\n".join(parts), results
    except Exception as e:
        return f"Search error: {e}", []


# ══════════════════════════════════════════════════════════════════════════════
#  NODE 1 – MEMORY
# ══════════════════════════════════════════════════════════════════════════════
def memory_node(state: AgentState) -> AgentState:
    context = recall_from_memory(state["user_query"])
    state["memory_context"] = context if context else "No previous context."
    return state


# ══════════════════════════════════════════════════════════════════════════════
#  NODE 2 – INTENT CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════
INTENT_SYSTEM = """You are an Intent Classification Agent. Parse the user query into a JSON plan.

Available agents:
- "weather"       → weather / forecast
- "outdoor"       → hiking, parks, sports, outdoor painting
- "entertainment" → movies, concerts, shows, museums, art studios
- "food"          → restaurants, cafes, dining
- "price"         → ANY mention of budget, price, cost, "under ₹X", "below $X", "cheap", "affordable", "free"
- "general"       → anything else

Rules:
1. Extract city. If missing, set needs_city=true.
2. If query mentions a price/budget constraint ALWAYS add a "price" task AND extract budget_min / budget_max.
3. If a task is conditional (e.g. "if it rains"), note the condition.
4. ALWAYS include "weather" if any condition depends on weather.
5. Output ONLY valid JSON:
{
  "city": "<city or null>",
  "needs_city": <true|false>,
  "budget_min": <number or null>,
  "budget_max": <number or null>,
  "tasks": [
    {"task_id": 1, "description": "...", "agent": "weather",  "condition": null},
    {"task_id": 2, "description": "...", "agent": "outdoor",  "condition": "if sunny"},
    {"task_id": 3, "description": "...", "agent": "price",    "condition": null}
  ],
  "summary": "One-line summary"
}
"""

def intent_agent(state: AgentState) -> AgentState:
    llm         = get_llm(temperature=0.1)
    memory_hint = f"\nUser's past preferences:\n{state['memory_context']}" if state.get("memory_context") else ""
    messages = [
        SystemMessage(content=INTENT_SYSTEM),
        HumanMessage(content=f"User query: {state['user_query']}{memory_hint}")
    ]
    response = llm.invoke(messages)
    raw      = response.content.strip()
    json_match = re.search(r'\{.*\}', raw, re.DOTALL)
    if json_match:
        raw = json_match.group()
    try:
        plan = json.loads(raw)
    except json.JSONDecodeError:
        plan = {
            "city": None, "needs_city": True, "budget_min": None, "budget_max": None,
            "tasks": [{"task_id": 1, "description": state["user_query"],
                       "agent": "general", "condition": None}],
            "summary": state["user_query"]
        }
    state["intent_plan"]  = plan
    state["city"]         = plan.get("city") or state.get("city")
    state["needs_city"]   = plan.get("needs_city", False)
    state["active_tasks"] = list({t["agent"] for t in plan.get("tasks", [])})
    state["budget_min"]   = plan.get("budget_min")
    state["budget_max"]   = plan.get("budget_max")
    return state


# ══════════════════════════════════════════════════════════════════════════════
#  NODE 3 – WEATHER
# ══════════════════════════════════════════════════════════════════════════════
def weather_agent(state: AgentState) -> AgentState:
    if "weather" not in state.get("active_tasks", []):
        return state
    city = state.get("city", "New York")
    data = get_weather(city)
    if "error" in data:
        state["weather_result"]    = f"Could not fetch weather for {city}: {data['error']}"
        state["weather_condition"] = "unknown"
    else:
        status = data["status"].lower()
        if any(w in status for w in ["rain", "drizzle", "storm", "shower"]):
            condition = "rainy"
        elif any(w in status for w in ["cloud", "overcast", "mist", "fog"]):
            condition = "cloudy"
        elif any(w in status for w in ["clear", "sun"]):
            condition = "sunny"
        else:
            condition = "mild"
        state["weather_condition"] = condition
        state["weather_result"] = (
            f"🌤 Weather in {city}:\n"
            f"  Condition : {data['status'].title()}\n"
            f"  Temp      : {data['temp_c']}°C (feels like {data['feels_like']}°C)\n"
            f"  Humidity  : {data['humidity']}%\n"
            f"  Wind      : {data['wind_kmh']} km/h\n"
            f"  Summary   : It is {condition} right now."
        )
    return state


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER – conditional task gate
# ══════════════════════════════════════════════════════════════════════════════
def _task_is_active(state: AgentState, agent_name: str) -> bool:
    plan         = state.get("intent_plan", {})
    weather_cond = state.get("weather_condition", "unknown")
    for task in plan.get("tasks", []):
        if task["agent"] != agent_name:
            continue
        cond = (task.get("condition") or "").lower()
        if not cond:
            return True
        if "sunny" in cond or "clear" in cond:
            return weather_cond in ("sunny", "mild")
        if "rain" in cond:
            return weather_cond == "rainy"
        if "cloud" in cond:
            return weather_cond == "cloudy"
        return True
    return False


# ══════════════════════════════════════════════════════════════════════════════
#  LINK INSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════
LINK_INSTRUCTION = """
IMPORTANT — Formatting rules:
- For EVERY specific venue, show, event, or restaurant you mention, include a clickable
  markdown booking/info link if the source URL is available.
- Format links as: [Book / Details here](<url>)
- If no direct booking URL is found, link to the search source instead.
- Never omit the URL if it was provided in the search results.
"""


# ══════════════════════════════════════════════════════════════════════════════
#  NODE 4 – OUTDOOR
# ══════════════════════════════════════════════════════════════════════════════
def outdoor_agent(state: AgentState) -> AgentState:
    if "outdoor" not in state.get("active_tasks", []):
        return state
    if not _task_is_active(state, "outdoor"):
        state["outdoor_result"] = "⛔ Skipped outdoor suggestions — weather not suitable."
        return state
    city       = state.get("city", "")
    query_desc = next(
        (t["description"] for t in state.get("intent_plan", {}).get("tasks", []) if t["agent"] == "outdoor"),
        state["user_query"]
    )
    raw_text, _ = tavily_search(f"{query_desc} in {city} today" if city else query_desc)
    llm      = get_llm(temperature=0.4)
    messages = [
        SystemMessage(content=(
            "You are an Outdoor Activities Agent. Summarize into 3-5 actionable recommendations. "
            "Be specific — include location names, timings, tips.\n" + LINK_INSTRUCTION
        )),
        HumanMessage(content=f"Search results:\n{raw_text}\n\nCity: {city}\nRequest: {query_desc}")
    ]
    resp = llm.invoke(messages)
    state["outdoor_result"] = f"🏕 Outdoor Options:\n{resp.content}"
    return state


# ══════════════════════════════════════════════════════════════════════════════
#  NODE 5 – ENTERTAINMENT
# ══════════════════════════════════════════════════════════════════════════════
def entertainment_agent(state: AgentState) -> AgentState:
    if "entertainment" not in state.get("active_tasks", []):
        return state
    if not _task_is_active(state, "entertainment"):
        state["entertainment_result"] = "⛔ Skipped entertainment suggestions — conditions not met."
        return state
    city       = state.get("city", "")
    query_desc = next(
        (t["description"] for t in state.get("intent_plan", {}).get("tasks", []) if t["agent"] == "entertainment"),
        state["user_query"]
    )
    raw_text, _ = tavily_search(f"{query_desc} in {city}" if city else query_desc)
    llm      = get_llm(temperature=0.4)
    messages = [
        SystemMessage(content=(
            "You are an Entertainment Agent. Summarize into 3-5 specific options. "
            "Include venue names, showtimes, and addresses where available.\n" + LINK_INSTRUCTION
        )),
        HumanMessage(content=f"Search results:\n{raw_text}\n\nCity: {city}\nRequest: {query_desc}")
    ]
    resp = llm.invoke(messages)
    state["entertainment_result"] = f"🎬 Entertainment Options:\n{resp.content}"
    return state


# ══════════════════════════════════════════════════════════════════════════════
#  NODE 6 – FOOD
# ══════════════════════════════════════════════════════════════════════════════
def food_agent(state: AgentState) -> AgentState:
    if "food" not in state.get("active_tasks", []):
        return state
    city       = state.get("city", "")
    query_desc = next(
        (t["description"] for t in state.get("intent_plan", {}).get("tasks", []) if t["agent"] == "food"),
        "restaurants"
    )
    budget_hint = ""
    if state.get("budget_max"):
        budget_hint = f" under ₹{state['budget_max']}"
    elif state.get("budget_min"):
        budget_hint = f" above ₹{state['budget_min']}"
    raw_text, _ = tavily_search(
        f"best {query_desc}{budget_hint} in {city}" if city else f"best {query_desc}{budget_hint}"
    )
    llm      = get_llm(temperature=0.4)
    messages = [
        SystemMessage(content=(
            "You are a Food & Restaurant Agent. Provide 3-5 specific recommendations. "
            "Include cuisine type, price range, and highlights.\n" + LINK_INSTRUCTION
        )),
        HumanMessage(content=f"Search results:\n{raw_text}\n\nCity: {city}\nRequest: {query_desc}")
    ]
    resp = llm.invoke(messages)
    state["food_result"] = f"🍽 Food Recommendations:\n{resp.content}"
    return state


# ══════════════════════════════════════════════════════════════════════════════
#  NODE 7 – PRICE FILTER AGENT
# ══════════════════════════════════════════════════════════════════════════════
def price_agent(state: AgentState) -> AgentState:
    if "price" not in state.get("active_tasks", []):
        return state
    city       = state.get("city", "")
    bmin       = state.get("budget_min")
    bmax       = state.get("budget_max")
    user_query = state.get("user_query", "")
    if bmax and bmin:
        price_clause = f"between ₹{bmin} and ₹{bmax}"
    elif bmax:
        price_clause = f"under ₹{bmax}"
    elif bmin:
        price_clause = f"above ₹{bmin}"
    else:
        price_clause = "affordable cheap budget"
    raw_text, _ = tavily_search(
        f"{user_query} {price_clause} in {city}" if city else f"{user_query} {price_clause}"
    )
    if bmax and bmin:
        budget_context = f"The user's budget is between ₹{bmin} and ₹{bmax}."
    elif bmax:
        budget_context = f"The user's budget is under ₹{bmax}. Only include options at or below this price."
    elif bmin:
        budget_context = f"The user wants options above ₹{bmin} (premium/special)."
    else:
        budget_context = "The user wants affordable/budget-friendly options."
    llm      = get_llm(temperature=0.3)
    messages = [
        SystemMessage(content=(
            f"You are a Price & Budget Agent. {budget_context}\n"
            "From the search results, extract and list ONLY options that fit the budget. "
            "For each option clearly state the price/cost. Discard anything outside the range.\n"
            + LINK_INSTRUCTION
        )),
        HumanMessage(content=f"Search results:\n{raw_text}\n\nQuery: {user_query}\nCity: {city}")
    ]
    resp = llm.invoke(messages)
    budget_label = (
        f"₹{bmin}–₹{bmax}" if (bmin and bmax) else
        (f"under ₹{bmax}" if bmax else (f"above ₹{bmin}" if bmin else "budget-friendly"))
    )
    state["price_result"] = f"💰 Budget Options ({budget_label}):\n{resp.content}"
    return state


# ══════════════════════════════════════════════════════════════════════════════
#  NODE 8 – GENERAL
# ══════════════════════════════════════════════════════════════════════════════
def general_agent(state: AgentState) -> AgentState:
    if "general" not in state.get("active_tasks", []):
        return state
    city        = state.get("city", "")
    query       = f"{state['user_query']} in {city}" if city else state["user_query"]
    raw_text, _ = tavily_search(query)
    llm      = get_llm(temperature=0.5)
    messages = [
        SystemMessage(content="You are a helpful general-purpose assistant. Answer concisely.\n" + LINK_INSTRUCTION),
        HumanMessage(content=f"Search results:\n{raw_text}\n\nQuestion: {state['user_query']}")
    ]
    resp = llm.invoke(messages)
    state["general_result"] = f"🔍 General Info:\n{resp.content}"
    return state


# ══════════════════════════════════════════════════════════════════════════════
#  NODE 9 – ANALYSIS / SYNTHESIS
#  streaming=True  →  LangGraph "messages" mode automatically intercepts and
#  emits each token as a (AIMessageChunk, metadata) tuple without any extra
#  code here. The node itself uses plain .invoke() as before.
# ══════════════════════════════════════════════════════════════════════════════
def analysis_agent(state: AgentState) -> AgentState:
    parts = []
    if state.get("weather_result"):       parts.append(state["weather_result"])
    if state.get("outdoor_result"):       parts.append(state["outdoor_result"])
    if state.get("entertainment_result"): parts.append(state["entertainment_result"])
    if state.get("food_result"):          parts.append(state["food_result"])
    if state.get("price_result"):         parts.append(state["price_result"])
    if state.get("general_result"):       parts.append(state["general_result"])
    combined = "\n\n".join(parts)

    # streaming=True enables token-level events in LangGraph "messages" mode
    llm = get_llm(temperature=0.6, streaming=True)

    messages = [
        SystemMessage(content=(
            "You are the Analysis Agent — the final synthesizer. Create ONE cohesive, friendly, "
            "actionable response. Start with a brief summary, then present each category's highlights. "
            "Add outfit suggestions or timing tips based on weather. "
            "End with a warm encouraging line. Use emojis sparingly.\n"
            "IMPORTANT: Preserve ALL markdown links ([text](url)) from the agent outputs."
        )),
        HumanMessage(content=(
            f"Original query: {state['user_query']}\n"
            f"City: {state.get('city', 'Unknown')}\n"
            f"Weather: {state.get('weather_condition', 'unknown')}\n"
            f"Budget: min={state.get('budget_min')}, max={state.get('budget_max')}\n\n"
            f"Agent outputs:\n{combined}"
        ))
    ]
    resp = llm.invoke(messages)
    state["final_answer"] = resp.content
    save_to_memory(
        query=state["user_query"],
        result=state["final_answer"],
        city=state.get("city", "")
    )
    return state


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTING
# ══════════════════════════════════════════════════════════════════════════════
def route_after_weather(state: AgentState) -> str:
    tasks = state.get("active_tasks", [])
    for t in ["outdoor", "entertainment", "food", "price", "general"]:
        if t in tasks:
            return t
    return "analysis"

def route_after_outdoor(state: AgentState) -> str:
    tasks = state.get("active_tasks", [])
    for t in ["entertainment", "food", "price", "general"]:
        if t in tasks:
            return t
    return "analysis"

def route_after_entertainment(state: AgentState) -> str:
    tasks = state.get("active_tasks", [])
    for t in ["food", "price", "general"]:
        if t in tasks:
            return t
    return "analysis"

def route_after_food(state: AgentState) -> str:
    tasks = state.get("active_tasks", [])
    for t in ["price", "general"]:
        if t in tasks:
            return t
    return "analysis"

def route_after_price(state: AgentState) -> str:
    if "general" in state.get("active_tasks", []):
        return "general"
    return "analysis"


# ══════════════════════════════════════════════════════════════════════════════
#  BUILD LANGGRAPH
# ══════════════════════════════════════════════════════════════════════════════
def build_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("memory",        memory_node)
    workflow.add_node("intent",        intent_agent)
    workflow.add_node("weather",       weather_agent)
    workflow.add_node("outdoor",       outdoor_agent)
    workflow.add_node("entertainment", entertainment_agent)
    workflow.add_node("food",          food_agent)
    workflow.add_node("price",         price_agent)
    workflow.add_node("general",       general_agent)
    workflow.add_node("analysis",      analysis_agent)

    workflow.set_entry_point("memory")
    workflow.add_edge("memory", "intent")

    all_nodes = {
        "weather": "weather", "outdoor": "outdoor",
        "entertainment": "entertainment", "food": "food",
        "price": "price", "general": "general", "analysis": "analysis"
    }

    workflow.add_conditional_edges(
        "intent",
        lambda s: "weather" if "weather" in s.get("active_tasks", []) else route_after_weather(s),
        all_nodes
    )
    workflow.add_conditional_edges("weather",       route_after_weather,       all_nodes)
    workflow.add_conditional_edges("outdoor",       route_after_outdoor,       all_nodes)
    workflow.add_conditional_edges("entertainment", route_after_entertainment, all_nodes)
    workflow.add_conditional_edges("food",          route_after_food,          all_nodes)
    workflow.add_conditional_edges("price",         route_after_price,
                                   {"general": "general", "analysis": "analysis"})
    workflow.add_edge("general",  "analysis")
    workflow.add_edge("analysis", END)

    return workflow.compile(checkpointer=MemorySaver())


# ── Singleton graph ──────────────────────────────────────────────────────────
_graph = None

def _get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def _make_initial_state(user_query: str, city_override: str = "") -> AgentState:
    return {
        "user_query":           user_query,
        "city":                 city_override.strip() if city_override else None,
        "needs_city":           False,
        "intent_plan":          None,
        "weather_result":       None,
        "outdoor_result":       None,
        "entertainment_result": None,
        "food_result":          None,
        "general_result":       None,
        "price_result":         None,
        "final_answer":         None,
        "memory_context":       None,
        "active_tasks":         [],
        "weather_condition":    None,
        "budget_min":           None,
        "budget_max":           None,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC ENTRY POINT 1 — blocking (original API, unchanged)
# ══════════════════════════════════════════════════════════════════════════════
def run_query(user_query: str, city_override: str = "") -> dict:
    config = {"configurable": {"thread_id": "session_blocking"}}
    return _get_graph().invoke(
        _make_initial_state(user_query, city_override), config=config
    )


# ══════════════════════════════════════════════════════════════════════════════
#  STATUS DISPLAY HELPERS
# ══════════════════════════════════════════════════════════════════════════════
_NODE_LABELS = {
    "memory":        "🧠  Recalling memory",
    "intent":        "🎯  Classifying intent",
    "weather":       "🌤  Fetching weather",
    "outdoor":       "🏕  Searching outdoor activities",
    "entertainment": "🎬  Searching entertainment",
    "food":          "🍽  Searching food options",
    "price":         "💰  Filtering by budget",
    "general":       "🔍  Searching general info",
    "analysis":      "✍️  Writing your plan",
}

# Execution order for status display
_NODE_ORDER = ["memory", "intent", "weather", "outdoor",
               "entertainment", "food", "price", "general", "analysis"]


def _build_status_md(done: list[str], active: str | None) -> str:
    """
    Renders all nodes in pipeline order.
      ✅  completed nodes
      ⏳  currently running node
    Only nodes that have been seen (done or active) are shown.
    """
    seen = set(done) | ({active} if active else set())
    lines = []
    for node in _NODE_ORDER:
        if node not in seen:
            continue
        label = _NODE_LABELS.get(node, node)
        if node in done:
            lines.append(f"✅  {label}")
        elif node == active:
            lines.append(f"⏳  {label}…")
    return "\n\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC ENTRY POINT 2 — streaming generator
#
#  Uses LangGraph stream_mode=["updates", "messages"]:
#
#    "updates"  events → chunk is {node_name: state_delta}
#                        Fired once a node FINISHES. Use it to advance the
#                        status display.
#
#    "messages" events → chunk is (AIMessageChunk, metadata)
#                        Fired for every LLM token in any node that has
#                        streaming=True. We filter on metadata["langgraph_node"]
#                        == ANALYSIS_NODE so only synthesis tokens appear.
#
#  Gradio ChatInterface streaming: the generator must yield the FULL
#  cumulative string each time (Gradio replaces the bubble, not appends).
# ══════════════════════════════════════════════════════════════════════════════
def stream_query(user_query: str, city_override: str = "") -> Generator[str, None, None]:
    graph  = _get_graph()
    config = {"configurable": {"thread_id": "session_stream"}}
    state  = _make_initial_state(user_query, city_override)

    done_nodes:      list[str] = []
    active_node:     str | None = None
    answer_tokens:   str = ""
    answer_started:  bool = False

    for mode, chunk in graph.stream(
        state,
        config=config,
        stream_mode=["updates", "messages"],
    ):

        # ── Node finished ──────────────────────────────────────────────────
        if mode == "updates":
            # chunk = {node_name: state_delta}  (may contain __end__ key)
            for node_name in chunk:
                if node_name in ("__end__", "__start__"):
                    continue

                # The previous active node is now done
                if active_node and active_node not in done_nodes:
                    done_nodes.append(active_node)

                active_node = node_name
                status = _build_status_md(done_nodes, active=active_node)

                if answer_started:
                    # Analysis already producing tokens — append below separator
                    yield status + "\n\n---\n\n" + answer_tokens
                else:
                    yield status

        # ── LLM token arrived ─────────────────────────────────────────────
        elif mode == "messages":
            message_chunk, metadata = chunk

            # Only handle tokens from the analysis (synthesis) node
            if metadata.get("langgraph_node") != ANALYSIS_NODE:
                continue

            token = message_chunk.content
            if not token:
                continue

            answer_started = True
            answer_tokens += token

            # Show analysis as still running while tokens stream in
            status = _build_status_md(done_nodes, active=ANALYSIS_NODE)
            yield status + "\n\n---\n\n" + answer_tokens

    # ── All done: mark final node as complete ─────────────────────────────
    if active_node and active_node not in done_nodes:
        done_nodes.append(active_node)

    status = _build_status_md(done_nodes, active=None)

    if answer_started:
        yield status + "\n\n---\n\n" + answer_tokens
    else:
        # Fallback: graph ran but no tokens streamed (shouldn't happen normally)
        yield status