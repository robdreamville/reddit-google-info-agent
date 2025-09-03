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

@tool
def research_topic_for_content(topic: str, platform_focus: str = "both") -> str:
    """
    Research a topic using Google search and Reddit to gather current information,
    trends, discussions, and public sentiment for content creation.
    
    Args:
        topic: The topic to research (e.g., "AI news", "crypto trends", "tech reviews")
        platform_focus: Target platform - "youtube", "tiktok", or "both"
    
    Returns:
        Comprehensive research findings including news, trends, and community discussions
    """
    try:
        # Import here to avoid circular imports
        from reddit_agent import RedditAgent
        from config import get_tool_prompt
        from datetime import datetime
        
        # Get research prompt from config
        research_prompt = get_tool_prompt(
            "research_prompt", 
            topic=topic, 
            platform_focus=platform_focus,
            current_date=datetime.utcnow().isoformat()
        )
        
        # Create research agent instance
        research_agent = RedditAgent()
        
        # Use the Reddit agent to research the topic with config prompt
        research_results = research_agent.chat(research_prompt)
        
        return f"Research Results for '{topic}' (Platform focus: {platform_focus}):\n{research_results}"
        
    except Exception as e:
        return f"Error during research: {str(e)}"

@tool
def research_trending_topics(category: str = "general") -> str:
    """
    Research trending topics using Google search and Reddit to find what is currently popular.
    
    Args:
        category: The category to research trends in (e.g., "AI", "gaming", "finance"). Defaults to "general".
    
    Returns:
        A report of trending topics, sentiment, and content angles.
    """
    try:
        from reddit_agent import RedditAgent
        from config import get_tool_prompt
        from datetime import datetime
        
        trending_prompt = get_tool_prompt(
            "trending_research_prompt", 
            category=category,
            current_date=datetime.utcnow().isoformat()
        )
        
        research_agent = RedditAgent()
        
        trending_results = research_agent.chat(trending_prompt)
        
        return f"Trending Topics Report (Category: {category}):\n{trending_results}"
        
    except Exception as e:
        return f"Error during trending research: {str(e)}"

@tool
def generate_platform_content(
    topic: str,
    platform: str,
    research_summary: str,
    content_type: str = "educational",
    tone: str = "engaging"
) -> str:
    """
    Generate ready-to-use script content for YouTube or TikTok videos.
    
    Args:
        topic: Main topic/subject for the content.
        platform: "youtube" or "tiktok".
        research_summary: A summary of research findings for context.
        content_type: "educational", "how-to", "storytelling", "news", "review", "comparison".
        tone: "conversational", "authoritative", "energetic", "inspirational", "humorous", "intriguing", "suspenseful".
    
    Returns:
        Formatted script with timing cues, emphasis points, and platform optimization.
    """
    
    try:
        from config import get_content_creator_config, get_tool_prompt
        from langchain_google_genai import ChatGoogleGenerativeAI
        import os

        config = get_content_creator_config()
        platform_specs = config["platform_specs"]
        content_types = config["content_types"]
        tone_settings = config["tone_settings"]
        
        specs = platform_specs.get(platform.lower(), platform_specs["youtube"])
        
        content_type_details = content_types.get(content_type, {})
        content_description = content_type_details.get("description", "")
        content_structure = content_type_details.get("structure", "")

        tone_description = tone_settings.get(tone, "")

        duration = specs.get("optimal_duration", "30-60s")

        content_prompt = get_tool_prompt(
            "content_generation_prompt",
            topic=topic,
            platform=platform.upper(),
            research_summary=research_summary,
            content_description=content_description,
            content_structure=content_structure,
            tone_description=tone_description,
            duration=duration,
            hook_time=specs["hook_time"],
            pace=specs["pace"],
            style=specs["style"]
        )
        
        model_config = config["model"]
        llm = ChatGoogleGenerativeAI(
            model=model_config["name"],
            temperature=model_config["temperature"],
            api_key=os.getenv("GEMINI_API_KEY")
        )

        response = llm.invoke(content_prompt)
        return response.content
        
    except Exception as e:
        return f"Error generating content: {str(e)}"

@tool
def analyze_content_performance(content_text: str, platform: str) -> str:
    """
    Analyze content for potential performance metrics and optimization suggestions.
    
    Args:
        content_text: The script/content to analyze
        platform: Target platform ("youtube" or "tiktok")
    
    Returns:
        Analysis with engagement predictions and optimization tips
    """
    
    try:
        from config import get_tool_prompt, get_content_creator_config
        from langchain_google_genai import ChatGoogleGenerativeAI
        import os

        current_date = datetime.utcnow().isoformat()

        analysis_prompt = get_tool_prompt(
            "content_analysis_prompt",
            content_text=content_text,
            platform=platform,
            current_date=current_date
        )
        
        config = get_content_creator_config()
        model_config = config["model"]
        
        llm = ChatGoogleGenerativeAI(
            model=model_config["name"],
            temperature=model_config["temperature"],
            api_key=os.getenv("GEMINI_API_KEY")
        )
        
        response = llm.invoke(analysis_prompt)
        return response.content
        
    except Exception as e:
        return f"Error during content analysis: {str(e)}"

@tool
def generate_article(topic: str, research_summary: str, tone: str, style: str, optimal_length: str) -> str:
    """
    Generates a full article based on a topic.
    
    Args:
        topic: The main subject of the article.
        research_summary: A summary of research findings for context.
        tone: "conversational", "authoritative", "energetic", "inspirational", "humorous", "intriguing", "suspenseful".
        style: The desired writing style.
        optimal_length: The target length for the article.
        
    Returns:
        The generated article as a string.
    """
    try:
        from config import get_content_creator_config, get_tool_prompt
        from langchain_google_genai import ChatGoogleGenerativeAI
        import os

        config = get_content_creator_config()
        tone_settings = config["tone_settings"]
        
        tone_description = tone_settings.get(tone, "")

        prompt = get_tool_prompt(
            "article_generation_prompt",
            topic=topic,
            research_summary=research_summary,
            tone_description=tone_description,
            style=style,
            optimal_length=optimal_length
        )
        
        model_config = config["model"]
        llm = ChatGoogleGenerativeAI(
            model=model_config["name"],
            temperature=model_config["temperature"],
            api_key=os.getenv("GEMINI_API_KEY")
        )

        response = llm.invoke(prompt)
        return response.content
        
    except Exception as e:
        return f"Error generating article: {str(e)}"

@tool
def generate_x_thread(topic: str, research_summary: str, tone: str, style: str, thread_length: str) -> str:
    """
    Generates an X (Twitter) thread based on a topic.
    
    Args:
        topic: The main subject of the thread.
        research_summary: A summary of research findings for context.
        tone: "conversational", "authoritative", "energetic", "inspirational", "humorous", "intriguing", "suspenseful".
        style: The desired writing style.
        thread_length: The target number of posts in the thread.
        
    Returns:
        The generated thread as a single string, with posts separated by '---'.
    """
    try:
        from config import get_content_creator_config, get_tool_prompt
        from langchain_google_genai import ChatGoogleGenerativeAI
        import os

        config = get_content_creator_config()
        tone_settings = config["tone_settings"]
        
        tone_description = tone_settings.get(tone, "")

        prompt = get_tool_prompt(
            "x_thread_generation_prompt",
            topic=topic,
            research_summary=research_summary,
            tone_description=tone_description,
            style=style,
            thread_length=thread_length
        )
        
        model_config = config["model"]
        llm = ChatGoogleGenerativeAI(
            model=model_config["name"],
            temperature=model_config["temperature"],
            api_key=os.getenv("GEMINI_API_KEY")
        )

        response = llm.invoke(prompt)
        return response.content
        
    except Exception as e:
        return f"Error generating X thread: {str(e)}"

@tool
def save_content_to_file(content: str, folder: str, topic: str, platform: str) -> str:
    """
    Saves the given content to a file in the specified folder.
    
    Args:
        content: The text content to save.
        folder: The subfolder to save the file in (e.g., 'articles', 'x_threads').
        topic: The topic of the content, used for the filename.
        platform: The platform the content was generated for (e.g., 'article', 'x').
        
    Returns:
        The path to the saved file or an error message.
    """
    try:
        import os
        import re
        from datetime import datetime

        # Create the folder if it doesn't exist
        os.makedirs(folder, exist_ok=True)
        
        # Sanitize the topic to create a valid filename
        # Remove special characters, replace spaces with underscores
        sanitized_topic = re.sub(r'[^\w\s-]', '', topic).strip().replace(' ', '_')
        sanitized_topic = re.sub(r'[-\s]+', '_', sanitized_topic).lower()

        # Create a timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Determine file extension
        ext = ".txt"
        if platform == "article":
            ext = ".md"
        elif platform == "x":
            ext = ".txt"

        # Construct filename
        filename = f"{timestamp}_{sanitized_topic}{ext}"
        
        # Construct the full file path
        file_path = os.path.join(folder, filename)
        
        # Write the content to the file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
        return f"Successfully saved content to {file_path}"

    except Exception as e:
        return f"Error saving file: {str(e)}"