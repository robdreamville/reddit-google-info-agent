from langchain_core.tools import tool
from app.schemas.agent_schemas import (
    ResearchBrief, TrendReport, VideoScript, Article, XThread, ContentAnalysis
)
from app.core.config import get_content_creator_config
from app.core.logger import AppLogger
from datetime import datetime
import os
import json

_reddit_agent_instance = None
_llm_cache = {}

def get_reddit_agent():
    """Lazily load and cache the RedditAgent instance to avoid circular imports and re-instantiation."""
    global _reddit_agent_instance
    if _reddit_agent_instance is None:
        from app.agents.reddit_agent import RedditAgent
        _reddit_agent_instance = RedditAgent()
    return _reddit_agent_instance

def get_llm(temperature: float = 0.7):
    """Lazily load and cache ChatGoogleGenerativeAI instances by temperature."""
    global _llm_cache
    if temperature not in _llm_cache:
        from langchain_google_genai import ChatGoogleGenerativeAI
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        
        # Use content creator model name from config
        config = get_content_creator_config()
        model_name = config.model.name or "gemini-2.5-flash"
        
        _llm_cache[temperature] = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            api_key=api_key
        )
    return _llm_cache[temperature]

@tool
def research_topic_for_content(topic: str, platform_focus: str = "both") -> str:
    """
    Research a topic using Google search and Reddit to gather current information,
    trends, discussions, and public sentiment for content creation.
    
    Args:
        topic: The topic to research (e.g., "AI news", "crypto trends", "tech reviews")
        platform_focus: Target platform - "youtube", "tiktok", or "both"
    
    Returns:
        Comprehensive research findings as a serialized ResearchBrief JSON
    """
    try:
        from app.core.config import get_tool_prompt
        
        research_prompt = get_tool_prompt(
            "research_prompt", 
            topic=topic, 
            platform_focus=platform_focus,
            current_date=datetime.utcnow().isoformat()
        )
        
        # Get cached reddit agent and chat
        research_agent = get_reddit_agent()
        
        # Execute research. We expect a structured ResearchBrief JSON string back
        # because the RedditAgent will be updated to output structured data.
        brief_data = research_agent.research(research_prompt)
        
        # Return as pretty formatted JSON string so the Content Creator Agent can easily parse/read it
        return json.dumps(brief_data, indent=2)
        
    except Exception as e:
        return json.dumps({"error": f"Error during research: {str(e)}", "topic": topic})

@tool
def research_trending_topics(category: str = "general") -> str:
    """
    Research trending topics using Google search and Reddit to find what is currently popular.
    
    Args:
        category: The category to research trends in (e.g., "AI", "gaming", "finance"). Defaults to "general".
    
    Returns:
        A serialized TrendReport JSON containing trending topics, sentiment, and content angles.
    """
    try:
        from app.core.config import get_tool_prompt
        
        trending_prompt = get_tool_prompt(
            "trending_research_prompt", 
            category=category,
            current_date=datetime.utcnow().isoformat()
        )
        
        research_agent = get_reddit_agent()
        trend_data = research_agent.research_trends(trending_prompt)
        
        return json.dumps(trend_data, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Error during trending research: {str(e)}", "category": category})

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
        research_summary: A summary of research findings (ResearchBrief JSON).
        content_type: "educational", "how-to", "storytelling", "news", "review", "comparison".
        tone: "conversational", "authoritative", "energetic", "inspirational", "humorous", "intriguing", "suspenseful".
    
    Returns:
        A serialized VideoScript JSON string containing hook, segments, and call to action.
    """
    try:
        from app.core.config import get_tool_prompt
        config = get_content_creator_config()
        
        platform_lower = platform.lower()
        if platform_lower not in config.platform_specs:
            platform_lower = "youtube"
            
        specs = config.platform_specs[platform_lower]
        
        ctype_details = config.content_types.get(content_type)
        content_description = ctype_details.description if ctype_details else ""
        content_structure = ctype_details.structure if ctype_details else ""
        
        tone_description = config.tone_settings.get(tone, "")
        duration = specs.optimal_duration or "30-60s"
        
        content_prompt = get_tool_prompt(
            "content_generation_prompt",
            topic=topic,
            platform=platform.upper(),
            research_summary=research_summary,
            content_description=content_description,
            content_structure=content_structure,
            tone_description=tone_description,
            duration=duration,
            hook_time=specs.hook_time or "0-5s",
            pace=specs.pace or "moderate",
            style=specs.style
        )
        
        # Use LLM with structured output
        llm = get_llm(temperature=config.model.temperature)
        structured_llm = llm.with_structured_output(VideoScript)
        
        script: VideoScript = structured_llm.invoke(content_prompt)
        return script.model_dump_json(indent=2)
        
    except Exception as e:
        return json.dumps({"error": f"Error generating script: {str(e)}", "topic": topic, "platform": platform})

@tool
def generate_article(
    topic: str,
    research_summary: str,
    tone: str,
    style: str,
    optimal_length: str
) -> str:
    """
    Generates a full article based on a topic and research brief.
    
    Args:
        topic: The main subject of the article.
        research_summary: A summary of research findings (ResearchBrief JSON).
        tone: "conversational", "authoritative", "energetic", "inspirational", "humorous", "intriguing", "suspenseful".
        style: The desired writing style.
        optimal_length: The target length for the article.
        
    Returns:
        A serialized Article JSON string.
    """
    try:
        from app.core.config import get_tool_prompt
        config = get_content_creator_config()
        tone_description = config.tone_settings.get(tone, "")
        
        prompt = get_tool_prompt(
            "article_generation_prompt",
            topic=topic,
            research_summary=research_summary,
            tone_description=tone_description,
            style=style,
            optimal_length=optimal_length
        )
        
        llm = get_llm(temperature=config.model.temperature)
        structured_llm = llm.with_structured_output(Article)
        
        article: Article = structured_llm.invoke(prompt)
        return article.model_dump_json(indent=2)
        
    except Exception as e:
        return json.dumps({"error": f"Error generating article: {str(e)}", "topic": topic})

@tool
def generate_x_thread(
    topic: str,
    research_summary: str,
    tone: str,
    style: str,
    thread_length: str
) -> str:
    """
    Generates an X (Twitter) thread based on a topic and research.
    
    Args:
        topic: The main subject of the thread.
        research_summary: A summary of research findings (ResearchBrief JSON).
        tone: "conversational", "authoritative", "energetic", "inspirational", "humorous", "intriguing", "suspenseful".
        style: The desired writing style.
        thread_length: The target number of posts in the thread.
        
    Returns:
        A serialized XThread JSON string.
    """
    try:
        from app.core.config import get_tool_prompt
        config = get_content_creator_config()
        tone_description = config.tone_settings.get(tone, "")
        
        prompt = get_tool_prompt(
            "x_thread_generation_prompt",
            topic=topic,
            research_summary=research_summary,
            tone_description=tone_description,
            style=style,
            thread_length=thread_length
        )
        
        llm = get_llm(temperature=config.model.temperature)
        structured_llm = llm.with_structured_output(XThread)
        
        thread: XThread = structured_llm.invoke(prompt)
        return thread.model_dump_json(indent=2)
        
    except Exception as e:
        return json.dumps({"error": f"Error generating X thread: {str(e)}", "topic": topic})

@tool
def analyze_content_performance(content_text: str, platform: str) -> str:
    """
    Analyze content for potential performance metrics and optimization suggestions.
    
    Args:
        content_text: The script/content text to analyze
        platform: Target platform ("youtube", "tiktok", "article", or "x")
    
    Returns:
        A serialized ContentAnalysis JSON string.
    """
    try:
        from app.core.config import get_tool_prompt
        config = get_content_creator_config()
        
        analysis_prompt = get_tool_prompt(
            "content_analysis_prompt",
            content_text=content_text,
            platform=platform,
            current_date=datetime.utcnow().isoformat()
        )
        
        llm = get_llm(temperature=0.3)  # Lower temperature for critical evaluation
        structured_llm = llm.with_structured_output(ContentAnalysis)
        
        analysis: ContentAnalysis = structured_llm.invoke(analysis_prompt)
        return analysis.model_dump_json(indent=2)
        
    except Exception as e:
        return json.dumps({"error": f"Error analyzing content: {str(e)}", "platform": platform})
