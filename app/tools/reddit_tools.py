from langchain_core.tools import tool
import os

_reddit_client = None

def get_reddit_client():
    """Lazily load and cache PRAW Reddit client to avoid re-instantiation overhead."""
    global _reddit_client
    if _reddit_client is None:
        import praw
        client_id = os.getenv("REDDIT_CLIENT_ID")
        client_secret = os.getenv("REDDIT_CLIENT_SECRET")
        if not client_id or not client_secret:
            raise ValueError("REDDIT_CLIENT_ID or REDDIT_CLIENT_SECRET not set in environment.")
        _reddit_client = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent="reddit_agent_platform_v2"
        )
    return _reddit_client

@tool
def search_subreddit_content(subreddit: str, query: str, limit: int = 5, sort: str = "relevance") -> list:
    """
    Search for relevant posts and comments in a subreddit using a query string.
    Returns a list of matching posts/comments with title, author, score, and snippet.
    The 'sort' parameter can be 'new', 'top', or 'relevance'.
    """
    try:
        reddit = get_reddit_client()
        results = []
        
        # Search posts with sort
        for submission in reddit.subreddit(subreddit).search(query, sort=sort, limit=limit):
            results.append({
                "type": "post",
                "title": submission.title,
                "author": str(submission.author),
                "score": submission.score,
                "url": submission.url,
                "snippet": submission.selftext[:200] if submission.selftext else ""
            })
            
        # Search comments
        for comment in reddit.subreddit(subreddit).comments(limit=limit):
            if query.lower() in comment.body.lower():
                results.append({
                    "type": "comment",
                    "author": str(comment.author),
                    "score": comment.score,
                    "snippet": comment.body[:200],
                    "link": f"https://reddit.com{comment.permalink}"
                })
                
        # Sort comments by score (top first)
        comment_results = [r for r in results if r["type"] == "comment"]
        post_results = [r for r in results if r["type"] == "post"]
        comment_results.sort(key=lambda x: x["score"], reverse=True)
        
        return post_results + comment_results
    except Exception as e:
        return [{"error": f"Failed to search subreddit content: {str(e)}"}]

@tool
def search_subreddits(query: str, limit: int = 5) -> list:
    """
    Search for relevant subreddits using a query string.
    Returns a list of subreddit names, titles, and descriptions.
    """
    try:
        reddit = get_reddit_client()
        results = []
        for sub in reddit.subreddits.search(query, limit=limit):
            results.append({
                "name": sub.display_name,
                "title": sub.title,
                "description": sub.public_description
            })
        return results
    except Exception as e:
        return [{"error": f"Failed to search subreddits: {str(e)}"}]
