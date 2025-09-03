import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated, List, Dict, Optional
import time
import traceback
import json

from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from tools import (
    research_topic_for_content, 
    research_trending_topics, 
    generate_platform_content, 
    analyze_content_performance,
    generate_article,
    generate_x_thread,
    save_content_to_file
)
from config import get_content_creator_config, get_shared_config
from content_logger import ContentCreatorLogger
from datetime import datetime

# Load environment variables
load_dotenv()

class ContentCreatorAgent:
    def __init__(self, config_preset: Optional[str] = None):
        """
        Initialize content creator agent with configurable settings
        
        Args:
            config_preset: Optional preset configuration ("viral_focused", "educational_focused", etc.)
        """
        
        # Load configuration
        self.config = get_content_creator_config()
        self.shared_config = get_shared_config()
        
        # Apply preset if specified
        if config_preset:
            from config import apply_preset
            apply_preset(config_preset)
            # Reload config after preset
            self.config = get_content_creator_config()
        
        # Validate environment
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        
        # Check required environment variables
        for var in self.shared_config["environment"]["required_env_vars"]:
            if not os.getenv(var):
                raise ValueError(f"{var} environment variable not set")
        
        # Initialize memory with system message
        self.memory: List[AnyMessage] = [SystemMessage(content=self.config["system_prompt"])]
        
        # Initialize LLM with config settings
        model_config = self.config["model"]
        llm_kwargs = {
            "model": model_config["name"],
            "temperature": model_config["temperature"],
            "api_key": self.api_key,
        }
        
        # Add optional parameters if specified
        if model_config.get("max_tokens"):
            llm_kwargs["max_tokens"] = model_config["max_tokens"]
        if model_config.get("top_p"):
            llm_kwargs["top_p"] = model_config["top_p"]
        
        self.llm = ChatGoogleGenerativeAI(**llm_kwargs)
        
        # Define tools
        self.tools = [
            research_topic_for_content, 
            research_trending_topics,
            generate_platform_content, 
            analyze_content_performance,
            generate_article,
            generate_x_thread,
            save_content_to_file
        ]
        self.chat_with_tools = self.llm.bind_tools(self.tools)
        
        # Build agent
        self.agent = self._build_agent()
        
        # Initialize logger
        if self.config["logging"]["enabled"]:
            self.logger = ContentCreatorLogger(self.config["logging"]["log_file"])
    
    def _build_agent(self):
        """Build the LangGraph agent workflow"""
        
        class AgentState(TypedDict):
            messages: Annotated[list[AnyMessage], add_messages]
        
        def assistant(state: AgentState):
            """Main assistant node"""
            messages = state["messages"]
            
            if not messages or not isinstance(messages[0], SystemMessage):
                messages = [SystemMessage(content=self.config["system_prompt"])] + messages
            
            return {
                "messages": [self.chat_with_tools.invoke(messages)],
            }
        
        builder = StateGraph(AgentState)
        builder.add_node("assistant", assistant)
        builder.add_node("tools", ToolNode(self.tools))
        builder.add_edge(START, "assistant")
        builder.add_conditional_edges(
            "assistant",
            tools_condition,
        )
        builder.add_edge("tools", "assistant")
        
        return builder.compile()
    
    def _log_error_separately(self, error_data: Dict[str, any]):
        """Log errors to separate file if configured"""
        if (self.config["logging"]["enabled"] and 
            self.config["logging"].get("separate_error_log", False)):
            
            error_log_file = self.config["logging"].get("error_log_file", "content_creator_errors.json")
            error_logger = ContentCreatorLogger(error_log_file)
            ContentCreatorLogger.log_error(error_data)
    
    def create_content(self, topic: str, platforms: List[str], content_type: str = "educational", 
                      duration: str = None, tone: str = "engaging") -> Dict[str, str]:
        """
        Main method to create content for a given topic for multiple platforms.
        
        Args:
            topic: What to create content about
            platforms: List of platforms to generate for (e.g., ["youtube", "article"])
            content_type: Type of content to create
            duration: Target video duration (uses config defaults if None)
            tone: Tone/style of the content
            
        Returns:
            Dictionary with platform-specific scripts, metadata, and file paths.
        """
        
        start_time = time.time()
        current_date = datetime.utcnow().isoformat()
        
        self.memory.append(SystemMessage(content=f"Today's date is {current_date}"))
        
        run_log = {
            "user_message": f"Create {', '.join(platforms)} content about {topic}",
            "topic": topic,
            "platforms": platforms,
            "content_type": content_type,
            "duration": duration,
            "tone": tone,
            "tool_calls": [],
            "files_saved": [],
            "token_usage": 0,
            "latency": None,
            "success": False
        }
        
        final_result = {
            "topic": topic,
            "content_type": content_type,
            "tone": tone,
            "generated_at": current_date,
            "content": {},
            "files": {}
        }

        try:
            # Initial research for the topic
            research_results = self.research_topic(topic, platform_focus=', '.join(platforms))
            self.memory.append(SystemMessage(content=f"Research summary for {topic}:\n{research_results}"))

            for platform in platforms:
                platform_config = self.config["platform_specs"].get(platform, {})
                content = ""

                if platform in ["youtube", "tiktok"]:
                    content = generate_platform_content.invoke({
                        "topic": topic, "platform": platform, "content_type": content_type,
                        "tone": tone, "research_summary": research_results
                    })
                elif platform == "article":
                    content = generate_article.invoke({
                        "topic": topic, "tone": tone, "style": platform_config.get("style"),
                        "optimal_length": platform_config.get("optimal_length"), "research_summary": research_results
                    })
                elif platform == "x":
                    content = generate_x_thread.invoke({
                        "topic": topic, "tone": tone, "style": platform_config.get("style"),
                        "thread_length": platform_config.get("thread_length"), "research_summary": research_results
                    })
                
                if content:
                    final_result["content"][platform] = content
                    # Save the content
                    folder = self.config["output_paths"].get(f"{platform}s", "output")
                    if platform == "article":
                        folder = self.config["output_paths"].get("articles", "articles")
                    elif platform == "x":
                         folder = self.config["output_paths"].get("x_threads", "x_threads")

                    save_path = save_content_to_file.invoke({
                        "content": content, "folder": folder, "topic": topic, "platform": platform
                    })
                    final_result["files"][platform] = save_path
                    run_log["files_saved"].append(save_path)

            run_log["agent_response"] = f"Generated content for {', '.join(platforms)}. See files for details."
            run_log["generated_content"] = final_result
            run_log["success"] = True

        except Exception as e:
            error_data = {
                "error_type": "content_creation_error",
                "topic": topic, "platforms": platforms, "content_type": content_type,
                "error": str(e), "traceback": traceback.format_exc(),
                "latency": round(time.time() - start_time, 3)
            }
            self._log_error_separately(error_data)
            if self.config["logging"].get("log_errors", True):
                run_log["error"] = str(e)
                run_log["agent_response"] = f"[ERROR] {str(e)}"
            final_result["error"] = str(e)
        
        finally:
            run_log["latency"] = round(time.time() - start_time, 3)
            if self.config["logging"]["enabled"] and (run_log["success"] or self.config["logging"].get("log_errors", True)):
                ContentCreatorLogger.log_content_creation(run_log)
        
        return final_result

    def research_topic(self, topic: str, platform_focus: str = "all") -> str:
        """Research a topic with logging"""
        start_time = time.time()
        try:
            results = research_topic_for_content.invoke({"topic": topic, "platform_focus": platform_focus})
            if self.config["logging"]["enabled"]:
                ContentCreatorLogger.log_research_call({
                    "topic": topic, "platform_focus": platform_focus,
                    "results_length": len(results) if results else 0,
                    "results_preview": results[:200] if results else "",
                    "latency": round(time.time() - start_time, 3), "success": True
                })
            return results
        except Exception as e:
            error_msg = f"Research error: {str(e)}"
            self._log_error_separately({
                "error_type": "research_error", "topic": topic, "platform_focus": platform_focus,
                "error": error_msg, "latency": round(time.time() - start_time, 3)
            })
            return error_msg
    
    def analyze_content(self, content: str, platform: str) -> str:
        """Analyze content performance potential"""
        try:
            return analyze_content_performance.invoke({"content_text": content, "platform": platform})
        except Exception as e:
            return f"Analysis error: {str(e)}"
    
    def research_trending(self, category: str = "general") -> str:
        """Research trending topics with logging"""
        start_time = time.time()
        try:
            results = research_trending_topics.invoke({"category": category})
            if self.config["logging"]["enabled"]:
                ContentCreatorLogger.log_research_call({
                    "topic": f"trending_{category}",
                    "platform_focus": category,
                    "results_length": len(results) if results else 0,
                    "results_preview": results[:200] if results else "",
                    "latency": round(time.time() - start_time, 3), "success": True
                })
            return results
        except Exception as e:
            error_msg = f"Trending research error: {str(e)}"
            self._log_error_separately({
                "error_type": "trending_research_error", "category": category,
                "error": error_msg, "latency": round(time.time() - start_time, 3)
            })
            return error_msg
    
    def get_analytics(self) -> Dict[str, any]:
        """Get analytics for this agent's usage"""
        if self.config["logging"]["enabled"]:
            return ContentCreatorLogger.get_analytics()
        else:
            return {"message": "Logging is disabled"}
    
    def interactive_content_creator(self):
        """Interactive mode for content creation with enhanced options"""
        print("🎬 Content Creator Agent Ready!")
        print("Generate content for YouTube, TikTok, Articles, and X.")
        # ... (rest of the interactive mode setup print statements)
        print("Type 'quit' to exit")
        print("-" * 60)
        
        while True:
            try:
                print("\n📝 Content Creation Options:")
                print("1. Quick Create - Just enter a topic for all platforms")
                print("2. Custom Create - Full customization")
                print("3. Trending Research - Research what's trending")
                print("4. Analyze Content - Analyze existing content")
                print("5. View Analytics - See usage statistics")
                print("6. Configuration - View current settings")
                
                choice = input("\nChoose option (1-6) or 'quit': ").strip()
                
                if choice.lower() in ['quit', 'exit', 'q']:
                    print("Happy creating! 🎬")
                    break
                
                elif choice == "1":
                    topic = input("Enter topic: ").strip()
                    if topic:
                        print("\n🔄 Researching and creating content for all platforms...")
                        result = self.create_content(topic, platforms=["youtube", "tiktok", "article", "x"])
                        self._display_content(result)
                
                elif choice == "2":
                    topic = input("Topic: ").strip()
                    platform_input = input("Platforms (youtube,tiktok,article,x - comma separated): ").strip() or "youtube,tiktok"
                    platforms = [p.strip() for p in platform_input.split(',')]
                    
                    duration = None
                    # Only ask for duration if a video platform is selected
                    if any(p in ['youtube', 'tiktok'] for p in platforms):
                        duration = input("Video Duration (e.g., 15-30s, 1-3min): ").strip()

                    content_type = input(f"Type {list(self.config['content_types'].keys())}: ").strip() or "educational"
                    tone = input(f"Tone {list(self.config['tone_settings'].keys())}: ").strip() or "engaging"
                    
                    if topic:
                        print(f"\n🔄 Creating content for {', '.join(platforms)}...")
                        result = self.create_content(topic, platforms, content_type, duration, tone)
                        self._display_content(result)
                
                elif choice == "3":
                    category = input("Enter category to research (e.g., AI, gaming, or leave blank for general): ").strip() or "general"
                    print(f"\n🔍 Researching trends for {category}...")
                    trends = self.research_trending(category)
                    print(f"\n📈 Current Trends:\n{trends}")
                
                elif choice == "4":
                    # Analyze content by selecting from a list (remains largely the same)
                    print("\n🔍 Select content to analyze:")
                    logs = self.logger.get_logs(log_type="content_creation", limit=15)
                    successful_runs = [log for log in logs if log.get("data", {}).get("success") and "generated_content" in log.get("data", {})]
                    if not successful_runs:
                        print("No recently generated content found to analyze.")
                        continue
                    for i, log in enumerate(successful_runs):
                        topic = log.get("data", {}).get("topic", "Unknown Topic")
                        platforms = log.get("data", {}).get("platforms", [])
                        timestamp = log.get("timestamp", "Unknown time").split("T")[0]
                        print(f"{i + 1}. [{timestamp}] {topic} ({', '.join(platforms)})")
                    try:
                        selection = int(input("\nEnter number to analyze (or 0 to cancel): ").strip())
                        if selection == 0: continue
                        if not (1 <= selection <= len(successful_runs)): print("Invalid selection."); continue
                        selected_log = successful_runs[selection - 1]["data"]["generated_content"]["content"]
                        platforms_in_log = list(selected_log.keys())
                        if len(platforms_in_log) == 1:
                            platform_to_analyze = platforms_in_log[0]
                        else:
                            p_choice = input(f"Analyze for which platform? ({'/'.join(platforms_in_log)}): ").strip().lower()
                            if p_choice in platforms_in_log:
                                platform_to_analyze = p_choice
                            else:
                                print("Invalid platform choice."); continue
                        content_to_analyze = selected_log[platform_to_analyze]
                        print(f"\n🔍 Analyzing content for {platform_to_analyze}...")
                        analysis = self.analyze_content(content_to_analyze, platform_to_analyze)
                        print(f"\n📊 Analysis Results:\n{analysis}")
                    except (ValueError, IndexError): print("Invalid input.")
                
                elif choice == "5":
                    print("\n📊 Usage Analytics:")
                    analytics = self.get_analytics()
                    print(json.dumps(analytics, indent=2))
                
                elif choice == "6":
                    print("\n⚙️ Current Configuration:")
                    config_display = {"model": self.config["model"], "logging": self.config["logging"], "platform_specs": self.config["platform_specs"]}
                    print(json.dumps(config_display, indent=2))
                
                else:
                    print("Invalid choice. Please try again.")
                    
            except KeyboardInterrupt:
                print("\nHappy creating! 🎬")
                break
            except Exception as e:
                print(f"Error: {str(e)}")
                self._log_error_separately({"error_type": "interactive_error", "error": str(e), "traceback": traceback.format_exc()})
    
    def _display_content(self, result: Dict[str, any]):
        """Display generated content in a formatted way"""
        print("\n" + "="*60)
        print(f"🎯 CONTENT GENERATED FOR: {result.get('topic', 'Unknown Topic').upper()}")
        print("="*60)
        
        for platform, script in result.get("content", {}).items():
            print(f"\n📱 {platform.upper()} SCRIPT:")
            print("-" * 30)
            print(script)
            print("-" * 30)

        for platform, file_path in result.get("files", {}).items():
            print(f"\n💾 Saved {platform.upper()} content to: {file_path}")

        if 'error' in result:
            print(f"\n❌ ERROR: {result['error']}")
        
        print(f"\n⏰ Generated at: {result.get('generated_at', 'Unknown')}")
        print("="*60)


# Example usage and testing
if __name__ == "__main__":
    try:
        print("🚀 Initializing Content Creator Agent...")
        creator = ContentCreatorAgent(config_preset=None)
        creator.interactive_content_creator()
    except Exception as e:
        print(f"❌ Error initializing Content Creator Agent: {str(e)}")
        print("Make sure your .env file has GEMINI_API_KEY, REDDIT_CLIENT_ID, and REDDIT_CLIENT_SECRET")
        print("Also ensure you have the config.py and content_logger.py files in the same directory")
