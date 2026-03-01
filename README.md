# 🤖 Multi-Agent Intent-Based AI Assistant

A production-grade multi-agent system that understands *any* natural language query,
routes it intelligently through specialized agents, and synthesizes a single cohesive answer.

---

## Architecture

```
User Query
    │
    ▼
[Memory Node]          ← ChromaDB recalls past searches
    │
    ▼
[Intent Agent]         ← Groq LLaMA 3.3 — the "Sorting Hat"
    │                     Outputs structured JSON plan
    │                     Identifies city, breaks tasks,
    │                     flags conditional logic
    │
    ├──(if weather needed)──► [Weather Agent]  ← OpenWeatherMap API
    │                              │
    │                    (condition resolved)
    │                              │
    ├──────────────────────────────► [Outdoor Agent]       ← Tavily search
    ├──────────────────────────────► [Entertainment Agent] ← Tavily search
    ├──────────────────────────────► [Food Agent]          ← Tavily search
    └──────────────────────────────► [General Agent]       ← Tavily search
                                         │
                                         ▼
                                  [Analysis Agent]   ← Synthesizes everything
                                         │
                                         ▼
                                   Final Answer
                                  + Save to Memory
```

### Key Design Patterns

| Feature | Implementation |
|---|---|
| Intent Routing | Pydantic-validated JSON plan from LLaMA 3.3 |
| Conditional Logic | Weather condition gates outdoor/indoor tasks |
| Memory | ChromaDB (local, free, persistent) |
| Web Search | Tavily AI (clean, LLM-optimized results) |
| Weather | OpenWeatherMap API |
| LLM | Groq + LLaMA 3.3-70b-versatile |
| UI | Gradio with custom dark CSS |

---

## Setup

### 1. Clone / download this folder

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set API keys
```bash
cp .env.example .env
# Edit .env with your keys
```

Get your keys here:
- **Groq**: https://console.groq.com  (free tier available)
- **Tavily**: https://tavily.com       (free tier: 1000 searches/month)
- **OpenWeatherMap**: https://openweathermap.org/api  (free tier available)

### 4. Run
```bash
python app.ipynb
```

Open http://localhost:7860 in your browser.

---

## Example Queries

| Query | City | What happens |
|---|---|---|
| "I want to go hiking today, but if it rains find me a movie theater" | Mumbai | Weather → Outdoor (sunny) OR Entertainment (rainy) |
| "I want to find a place to paint today" | Pune | General/Outdoor → Paint & Sip studios or parks |
| "Plan my Sunday with outdoor and indoor backup" | Bangalore | Weather + Outdoor + Entertainment + Food |
| "Find me sushi and a concert tonight" | Delhi | Food + Entertainment |

---

## File Structure

```
multi_agent_system/
├── agents.py          ← All agents, tools, LangGraph graph
├── app.ipynb             ← Gradio UI
├── requirements.txt
├── .env.example       ← Copy to .env and fill keys
└── README.md
```

---

## Extending the System

**Add a new agent** (e.g. Shopping Agent):
1. Add `"shopping"` to the agent list in the Intent Agent's system prompt
2. Write a `shopping_agent(state)` function in `agents.py`
3. Add the node: `workflow.add_node("shopping", shopping_agent)`
4. Add routing edges to include it in the flow

**Swap LLM**: Change `model_name` in `get_llm()` — any Groq-supported model works.

**Persistent ChromaDB**: Replace the in-memory client:
```python
chroma_client = chromadb.PersistentClient(path="./chroma_data")
```
