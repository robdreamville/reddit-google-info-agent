from langchain_core.tools import tool
import praw
import os
from datetime import datetime

@tool
def search_subreddit_content(subreddit: str, query: str, limit: int = 5, sort: str = "relevance") -> list:
    """
    Search for relevant posts and comments in a subreddit using a query string.
    Returns a list of matching posts/comments with title, author, score, and snippet.
    The 'sort' parameter can be 'new', 'top', or 'relevance'.
    """
    import praw
    import os
    reddit = praw.Reddit(
        client_id=os.getenv("REDDIT_CLIENT_ID"),
        client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
        user_agent="subreddit_content_search_agent"
    )
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
    # Combine posts and sorted comments
    return post_results + comment_results


@tool
def search_subreddits(query: str, limit: int = 5) -> list:
    """
    Search for relevant subreddits using a query string.
    Returns a list of subreddit names and their descriptions.
    """
    reddit = praw.Reddit(
        client_id=os.getenv("REDDIT_CLIENT_ID"),
        client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
        user_agent="subreddit_search_agent"
    )
    results = []
    for subreddit in reddit.subreddits.search(query, limit=limit):
        results.append({
            "name": subreddit.display_name,
            "title": subreddit.title,
            "description": subreddit.public_description
        })
    return results

@tool
def get_current_date() -> str:
    """Returns the current date in ISO format."""
    return datetime.utcnow().isoformat()

@tool
def google_grounding_search(query: str) -> str:
    """
    Search for current information using Google's grounded search.
    
    Use this tool when you need:
    - Latest/current information (news, events, prices, etc.)
    - Real-time data that might not be in your training
    - Recent developments or updates
    - Current facts to supplement your knowledge
    
    Args:
        query: Search query (be specific and focused)
        
    Returns:
        Current information from Google search with citations
        
    Example usage:
    - google_grounding_search("latest AI news January 2025")
    - google_grounding_search("current Tesla stock price")
    - google_grounding_search("Manchester United new signings 2025")
    """
    try:
        # Import the newer Google genai library
        from google import genai
        from google.genai import types
        import os
        
        # Get API key from environment
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return "Error: GEMINI_API_KEY not found in environment variables"
        
        # Initialize client and grounding tool
        client = genai.Client(api_key=api_key)
        grounding_tool = types.Tool(google_search=types.GoogleSearch())
        
        # Configure for grounding
        grounding_config = types.GenerateContentConfig(
            tools=[grounding_tool]
        )
        
        #print(f"🔎 Performing grounded search for: {query}")
        
        # Make grounded search request
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=f"Search for and provide current information about: {query}",
            config=grounding_config
        )
        
        result = response.text.strip()
        
        if not result:
            return "No results found from grounded search"
            
        return f"Current Information (via Google Search):\n{result}"
        
    except ImportError as e:
        return f"Error: google-genai library not available. Import error: {str(e)}"
    except Exception as e:
        return f"Error performing grounded search: {str(e)}"