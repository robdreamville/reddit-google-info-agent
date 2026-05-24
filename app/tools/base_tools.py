from langchain_core.tools import tool
from datetime import datetime
import os
import re

@tool
def get_current_date() -> str:
    """Returns the current date in ISO format."""
    return datetime.utcnow().isoformat()

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
        os.makedirs(folder, exist_ok=True)
        
        # Sanitize the topic
        sanitized = re.sub(r'[^\w\s-]', '', topic).strip().replace(' ', '_')
        sanitized = re.sub(r'[-\s]+', '_', sanitized).lower()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        ext = ".txt"
        if platform == "article":
            ext = ".md"
        elif platform == "x":
            ext = ".txt"

        filename = f"{timestamp}_{sanitized}{ext}"
        file_path = os.path.join(folder, filename)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
        return f"Successfully saved content to {file_path}"
    except Exception as e:
        return f"Error saving file: {str(e)}"

@tool
def google_grounding_search(query: str) -> str:
    """
    Search for current information using Google's grounded search.
    
    Use this tool when you need:
    - Latest/current information (news, events, prices, etc.)
    - Real-time data that might not be in your training
    - Recent developments or updates
    
    Args:
        query: Search query (be specific and focused)
        
    Returns:
        Current information from Google search with citations
    """
    try:
        from google import genai
        from google.genai import types
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return "Error: GEMINI_API_KEY not found in environment variables"
        
        client = genai.Client(api_key=api_key)
        grounding_tool = types.Tool(google_search=types.GoogleSearch())
        
        grounding_config = types.GenerateContentConfig(
            tools=[grounding_tool]
        )
        
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=f"Search for and provide current information about: {query}",
            config=grounding_config
        )
        
        result = response.text.strip() if response.text else ""
        if not result:
            return "No results found from grounded search"
            
        return f"Current Information (via Google Search):\n{result}"
        
    except ImportError as e:
        return f"Error: google-genai library not available. Import error: {str(e)}"
    except Exception as e:
        return f"Error performing grounded search: {str(e)}"
