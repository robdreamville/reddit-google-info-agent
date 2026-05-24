# Sports Research & Betting Agent

This is a standalone project that uses the same stack patterns as the main Reddit/Google agent in the parent project. It is intentionally isolated and does not modify any existing files.

Files created:
- `sports_tools.py` - helpers for YouTube transcript fetching and combined research wrappers
- `sports_config.py` - lightweight configuration for the sports agent
- `content_logger_sports.py` - separate logger for sports runs
- `sports_agent.py` - a simple SportsResearchAgent that orchestrates research and synthesis

Usage:
```python
from sports_agent.sports_agent import SportsResearchAgent
agent = SportsResearchAgent()
report = agent.research_and_analyze("Manchester United 2025 season")
print(report["synthesis"])
```

Notes:
- The YouTube transcript fetcher requires `youtube_transcript_api` to be installed.
- The agent will use existing `tools` functions (`google_grounding_search`, `search_subreddits`, `search_subreddit_content`) if present; otherwise it falls back gracefully.
- This scaffold is intended as a starting point. Add more tools (odds APIs, player stats, line movement) to expand betting-specific capabilities.
