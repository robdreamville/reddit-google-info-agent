# Reddit Topic Agent

A robust AI agent for searching and analyzing information using Reddit, Google, and other tools. Built with LangGraph, Gemini, and PRAW, with local logging and extensible tool support.

## Features
- Multi-source search: Reddit, Google, and more
- Step-by-step reasoning and synthesis
- Tool-based architecture (easy to add new tools)
- Local logging of all agent runs and tool calls
- Token usage and latency tracking
- Configurable system prompt for robust behavior

## Setup
1. Clone the repository:
   ```sh
   git clone <your-repo-url>
   cd reddit_topic
   ```
2. Create and activate a Python virtual environment:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
4. Add your API keys and secrets to `.env` (see below).

## .env Example
```
GEMINI_API_KEY=your_gemini_api_key
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
GOOGLE_API_KEY=your_google_api_key
```

## Usage
Run the agent interactively:
```sh
python agent.py
```

## Logging
- All agent runs and tool calls are logged in `logs/agent_runs.jsonl`.
- Logs are ignored by git via `.gitignore`.

## Extending
- Add new tools in `tools.py` and import them in `agent.py`.
- Update the system prompt in `agent.py` to guide agent behavior.

