"""
SportsResearchAgent - a separate agent that leverages your existing stack (Google + Reddit tools)
Does not modify your existing code; references existing `tools` module where available.

Simple usage:
from sports_agent.sports_agent import SportsResearchAgent
agent = SportsResearchAgent()
report = agent.research_and_analyze("Manchester United season 2025")
print(report)
"""
import os
import time
from typing import List, Optional

try:
    from langchain_core.messages import SystemMessage, HumanMessage
    from langchain_google_genai import ChatGoogleGenerativeAI
except Exception:
    # Fallback lightweight message/LLM stubs for environments without langchain
    class _SimpleMessage:
        def __init__(self, content: str):
            self.content = content

    SystemMessage = _SimpleMessage
    HumanMessage = _SimpleMessage

    class ChatGoogleGenerativeAI:
        """Very small fallback LLM used for dry runs when the real client isn't available."""
        def __init__(self, model: str = None, temperature: float = 0.0, api_key: str = None):
            self.model = model
            self.temperature = temperature
            self.api_key = api_key

        def invoke(self, messages):
            # Return a simple object mimicking the real response shape
            class _Resp:
                def __init__(self, content):
                    self.content = content
                    self.usage_metadata = {"total_tokens": 0}

            combined = "\n\n".join(getattr(m, "content", str(m)) for m in messages)
            return _Resp("[DUMMY LLM] Synthesis unavailable in this environment.\n" + combined[:2000])

# import helper tools from our new sports_tools module
from sports_agent.sports_tools import combined_research, extract_youtube_ids_from_urls, fetch_youtube_transcript, get_live_events
from sports_agent.content_logger_sports import SportsLogger
from sports_agent.sports_config import get_sports_config

# Attempt to import some existing tools if present
try:
    from tools import google_grounding_search, search_subreddit_content, search_subreddits
except Exception:
    google_grounding_search = None
    search_subreddit_content = None
    search_subreddits = None


class SportsResearchAgent:
    def __init__(self, config_preset: Optional[str] = None):
        self.config = get_sports_config()
        if config_preset:
            # simple preset hook (not implemented fully)
            self.config["model"]["temperature"] = 0.5

        self.llm = ChatGoogleGenerativeAI(
            model=self.config["model"]["name"],
            temperature=self.config["model"].get("temperature", 0.3),
            api_key=os.getenv("GEMINI_API_KEY"),
        )

    def research_and_analyze(self, topic: str) -> dict:
        """Run combined research and synthesize a short analysis."""
        start = time.time()
        run = {
            "topic": topic,
            "google": None,
            "subreddits": None,
            "youtube_links": None,
            "transcripts": {},
            "synthesis": None,
            "token_usage": None,
            "latency": None,
        }
        # --- 0) Get current date (use tool if available) ---
        try:
            from tools import get_current_date as _get_current_date_tool
            # Many tool wrappers expect a single 'tool_input' arg — try sending an empty string first.
            try:
                current_date = _get_current_date_tool("")
            except TypeError:
                # fallback: some wrappers expose .run
                if hasattr(_get_current_date_tool, "run"):
                    current_date = _get_current_date_tool.run("")
                else:
                    # Unable to call the tool wrapper with expected signature; surface to outer except
                    raise
        except Exception:
            from datetime import datetime
            current_date = datetime.utcnow().strftime("%Y-%m-%d")

        # Normalize to a local date string (YYYY-MM-DD) to avoid UTC offset issues
        from datetime import datetime, timezone
        current_date_only = None
        try:
            parsed = datetime.fromisoformat(current_date)
            # If no tzinfo, assume UTC
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            local_dt = parsed.astimezone()  # convert to local timezone
            current_date_only = local_dt.date().isoformat()
        except Exception:
            # Fallback to local system date
            current_date_only = datetime.now().date().isoformat()

        run["current_date"] = current_date_only

        # 1) Combined research using helper (wraps google + reddit tools if available)
        # Fetch google grounding results once and reuse to reduce quota usage
        google_results = None
        if google_grounding_search is not None:
            try:
                try:
                    google_results = google_grounding_search(topic, limit=4)
                except TypeError:
                    google_results = google_grounding_search(topic)
            except Exception:
                google_results = None

        research = combined_research(
            topic,
            subreddit_limit=self.config["search"]["subreddit_search_limit"],
            content_limit=self.config["search"]["content_search_limit"],
            google_results=google_results,
        )
        if research is None:
            research = {"google_results": None, "subreddits": [], "youtube_links": []}
        run["google"] = research.get("google_results")
        run["subreddits"] = research.get("subreddits")

        # 2) collect youtube links and transcripts
        yt_links = research.get("youtube_links", []) or []
        run["youtube_links"] = yt_links
        ids = extract_youtube_ids_from_urls(yt_links)
        for vid in ids:
            t = fetch_youtube_transcript(vid)
            if t:
                run["transcripts"][vid] = t

        # 2.5) Get live events for today (if any) — reuse google results to avoid extra queries
        try:
            from sports_agent.sports_tools import get_live_events
            live = get_live_events(current_date_only, precomputed_results=google_results)
        except Exception:
            live = []
        run["live_events"] = live

        # 3) Prepare prompt for LLM to synthesize findings (use local date-only string)
        prompt = self._build_synthesis_prompt(topic, research, run["transcripts"], current_date_only, run.get("live_events", []))
        # prime the model as a rigorous sports analyst
        system_msg = (
            "You are a world-class sports analyst and researcher.\n"
            "Your job is to synthesize the latest, verifiable information into actionable insights for sports research and betting.\n"
            "Prioritize authoritative news sources first, then summarize public sentiment from social platforms (Reddit, YouTube comments).\n"
            f"Current date: {current_date_only}. Use this date when judging recency.\n"
            "Be explicit about uncertainty and risk factors. Do not hallucinate. Cite sources when possible.\n"
            "Provide clear betting-relevant signals, confidence levels, and concise summaries."
        )
        messages = [SystemMessage(content=system_msg), HumanMessage(content=prompt)]
        try:
            resp = self.llm.invoke(messages)
            # many LLM wrappers put the reply in resp['content'] or resp itself may be a message object
            if isinstance(resp, dict) and "messages" in resp:
                # mirror your main agent format if present
                candidate = resp["messages"][-1]
                text = getattr(candidate, "content", str(candidate))
                usage = getattr(candidate, "usage_metadata", None)
            else:
                # assume resp is a message-like object
                text = getattr(resp, "content", str(resp))
                usage = getattr(resp, "usage_metadata", None)

            run["synthesis"] = text
            if usage and "total_tokens" in usage:
                run["token_usage"] = usage["total_tokens"]
        except Exception as e:
            run["synthesis"] = f"[ERROR] {e}"

        run["latency"] = round(time.time() - start, 3)

        # 4) log the run
        SportsLogger.log_run(run)

        return run

    def _build_synthesis_prompt(self, topic: str, research: dict, transcripts: dict, current_date: str, live_events: list) -> str:
        parts = []
        parts.append(f"Topic: {topic}")
        parts.append(f"\nCurrent date: {current_date}\n")
        parts.append("\nGoogle findings:\n")
        parts.append(str(research.get("google_results"))[:2000])
        parts.append("\nReddit subs:\n")
        parts.append(str(research.get("subreddits"))[:2000])
        if live_events:
            parts.append("\nLive events detected for today:\n")
            for e in live_events[:8]:
                parts.append(f"- {e.get('title')} ({e.get('start_time')}) [{e.get('source')}] -> {e.get('link')}")
        if transcripts:
            parts.append("\nYouTube transcripts summary attached. Use them where relevant.\n")
        parts.append("\nTask: Synthesize a concise research-backed analysis useful for sports betting research. Include key facts, potential betting-relevant signals, and risk factors. Do not hallucinate.")
        return "\n".join(parts)

    def interactive_content_creator(self):
        print("SportsResearchAgent interactive mode. Type 'quit' to exit.")
        while True:
            q = input("Topic> ").strip()
            if q.lower() in ("quit", "exit", "q"):
                break
            r = self.research_and_analyze(q)
            print("--- SYNTHESIS ---")
            print(r.get("synthesis"))
            print("--- END ---")
