"""
Simple config for the Sports Research & Analysis Agent.
This is intentionally lightweight and separate from the main config.py.
"""

SPORTS_CONFIG = {
    "model": {
        "name": "gemini-2.5-flash",
        "temperature": 0.3,
        "max_tokens": 1024,
    },
    "search": {
        "top_k_google": 6,
        "subreddit_search_limit": 6,
        "content_search_limit": 6,
    },
    "logging": {
        "enabled": True,
        "log_file": "sports_agent_runs.jsonl"
    }
}


def get_sports_config():
    return SPORTS_CONFIG
