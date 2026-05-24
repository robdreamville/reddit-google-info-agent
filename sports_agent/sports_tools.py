"""
Sports-specific tools: YouTube transcript fetcher, helper wrappers around existing google/reddit tools.
This module intentionally uses your existing tools where possible so nothing in the main repo is modified.
"""
from typing import List, Optional
import re

# Try to import existing tools from the main workspace
try:
    from tools import google_grounding_search, search_subreddit_content, search_subreddits
except Exception:
    google_grounding_search = None
    search_subreddit_content = None
    search_subreddits = None


def extract_youtube_ids_from_urls(urls: List[str]) -> List[str]:
    """Extract YouTube video IDs from a list of URLs."""
    ids = []
    for u in urls:
        # common patterns
        m = re.search(r"v=([A-Za-z0-9_-]{11})", u)
        if m:
            ids.append(m.group(1))
            continue
        m = re.search(r"youtu\.be/([A-Za-z0-9_-]{11})", u)
        if m:
            ids.append(m.group(1))
            continue
    return ids


def fetch_youtube_transcript(video_id: str) -> Optional[str]:
    """
    Try to fetch a YouTube transcript using `youtube_transcript_api` if available.
    Returns transcript string or None if the library isn't installed or transcript not available.
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except Exception:
        return None

    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        # transcript_list is list of {text, start, duration}
        full = "\n".join([t["text"] for t in transcript_list])
        return full
    except Exception:
        return None


def google_search_youtube_links(query: str, max_results: int = 5, precomputed_results=None) -> List[str]:
    """
    Use `google_grounding_search` (if present) to find YouTube links for the query.
    Returns a list of urls.
    """
    # If caller provided precomputed google results, prefer extracting youtube links from them.
    if precomputed_results is not None:
        results = precomputed_results
    elif google_grounding_search is None:
        return []

    try:
        # some grounding tools accept a "limit" kwarg; callers may vary. Try both styles.
        if precomputed_results is None:
            try:
                results = google_grounding_search(query + " site:youtube.com/watch", limit=max_results)
            except TypeError:
                results = google_grounding_search(query + " site:youtube.com/watch")

        # Expect results as list of dicts or raw text; attempt to extract urls
        urls = []
        if isinstance(results, list):
            for r in results:
                # If result is dict and has 'link' or 'url'
                if isinstance(r, dict):
                    if "link" in r:
                        urls.append(r["link"])
                    elif "url" in r:
                        urls.append(r["url"])
                    else:
                        # fallback to text search
                        text = " ".join(str(v) for v in r.values())
                        found = re.findall(r"https?://[\w\./\-?=&%]+", text)
                        urls.extend(found)
                else:
                    found = re.findall(r"https?://[\w\./\-?=&%]+", str(r))
                    urls.extend(found)
        else:
            found = re.findall(r"https?://[\w\./\-?=&%]+", str(results))
            urls.extend(found)

        # Filter youtube links
        yt = [u for u in urls if "youtube.com/watch" in u or "youtu.be/" in u]
        # dedupe
        seen = set()
        out = []
        for u in yt:
            if u not in seen:
                seen.add(u)
                out.append(u)
        return out[:max_results]
    except Exception:
        return []


def combined_research(topic: str, subreddit_limit: int = 5, content_limit: int = 5, google_results=None):
    """
    Lightweight combined research helper that uses google + reddit tools (if available).
    Returns a dict with keys: google_results, subreddits, subreddit_content, youtube_links
    """
    out = {
        "google_results": None,
        "subreddits": [],
        "subreddit_content": {},
        "youtube_links": [],
    }

    # Allow callers to pass precomputed google_results via kwargs in the future.
    # Use provided google_results when available to avoid duplicate queries
    if google_results is not None:
        out["google_results"] = google_results
    elif google_grounding_search is not None:
        try:
            try:
                out["google_results"] = google_grounding_search(topic, limit=8)
            except TypeError:
                out["google_results"] = google_grounding_search(topic)
        except Exception:
            out["google_results"] = None

    if search_subreddits is not None:
        try:
            try:
                subs_raw = search_subreddits(topic, limit=subreddit_limit)
            except TypeError:
                subs_raw = search_subreddits(topic)

            out["subreddits"] = subs_raw
            # If function returns list of dicts with name field
            names = []
            if isinstance(subs_raw, list):
                for s in subs_raw:
                    if isinstance(s, dict) and "name" in s:
                        names.append(s["name"])
                    elif isinstance(s, str):
                        names.append(s)
            # fetch content for top N subreddits
            for name in names[:3]:
                try:
                    try:
                        out["subreddit_content"][name] = search_subreddit_content(name, topic, limit=content_limit)
                    except TypeError:
                        out["subreddit_content"][name] = search_subreddit_content(name, topic)
                except Exception:
                    out["subreddit_content"][name] = []
        except Exception:
            out["subreddits"] = []

    # Try to find youtube links (passing google results to avoid extra queries when possible)
    try:
        yt_links = []
        if out.get("google_results"):
            yt_links = google_search_youtube_links(topic, max_results=5, precomputed_results=out.get("google_results"))
        else:
            yt_links = google_search_youtube_links(topic, max_results=5)
        out["youtube_links"] = yt_links
    except Exception:
        out["youtube_links"] = []

    return out


def get_live_events(date_str: str, sport: Optional[str] = None, max_results: int = 8, precomputed_results=None):
    """
    Heuristic live-event discovery.
    - Prefer google_grounding_search results when available.
    - Fall back to subreddit search for keywords like 'live thread', 'match thread', 'fight thread'.

    Returns a list of event dicts: {title, start_time, source, link}
    """
    events = []
    sport_part = f" {sport}" if sport else ""
    if google_grounding_search is not None or precomputed_results is not None:
        q = f"{date_str}{sport_part} live event schedule"
        try:
            if precomputed_results is not None:
                results = precomputed_results
            else:
                try:
                    results = google_grounding_search(q, limit=max_results)
                except TypeError:
                    results = google_grounding_search(q)

            if isinstance(results, list):
                for r in results:
                    title = None
                    link = None
                    if isinstance(r, dict):
                        title = r.get("title") or r.get("snippet") or str(r)
                        link = r.get("link") or r.get("url")
                    else:
                        title = str(r)
                        found = re.findall(r"https?://[\w\./\-?=&%]+", str(r))
                        link = found[0] if found else None

                    if title:
                        events.append({
                            "title": title,
                            "start_time": date_str,
                            "source": "google",
                            "link": link,
                        })
            # dedupe by link/title
            seen = set()
            out = []
            for e in events:
                key = (e.get("link"), e.get("title"))
                if key not in seen:
                    seen.add(key)
                    out.append(e)
            return out[:max_results]
        except Exception:
            pass

    # Fallback: search subreddits for live/match/fight threads
    if search_subreddits is not None:
        try:
            try:
                subs = search_subreddits(date_str + sport_part, limit=6)
            except TypeError:
                subs = search_subreddits(date_str + sport_part)

            for s in subs or []:
                name = s.get("name") if isinstance(s, dict) else str(s)
                try:
                    try:
                        posts = search_subreddit_content(name, date_str + " live")
                    except TypeError:
                        posts = search_subreddit_content(name, date_str + " live")
                    for p in posts or []:
                        title = p.get("title") if isinstance(p, dict) else str(p)
                        link = p.get("url") or p.get("permalink") if isinstance(p, dict) else None
                        events.append({
                            "title": title,
                            "start_time": date_str,
                            "source": f"reddit:{name}",
                            "link": link,
                        })
                except Exception:
                    continue
        except Exception:
            pass

    # final dedupe and return
    seen = set()
    out = []
    for e in events:
        key = (e.get("link"), e.get("title"))
        if key not in seen:
            seen.add(key)
            out.append(e)
    return out[:max_results]
