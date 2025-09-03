import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated, List
import time
import traceback

from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from tools import search_subreddits, search_subreddit_content, google_grounding_search, get_current_date
from config import get_reddit_agent_config, get_shared_config
from content_logger import ContentCreatorLogger
from datetime import datetime

# Load environment variables
load_dotenv()

class RedditAgent:
    def __init__(self):
        """Initialize agent with Gemini and tools from config"""
        
        # Load configuration
        self.config = get_reddit_agent_config()
        self.shared_config = get_shared_config()
        
        # Validate environment
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
            
        # Check required environment variables
        for var in self.shared_config["environment"]["required_env_vars"]:
            if not os.getenv(var):
                raise ValueError(f"{var} environment variable not set")

        # Initialize memory with system message from config
        self.system_message = self.config["system_prompt"]
        self.memory: List[AnyMessage] = [SystemMessage(content=self.system_message)]

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

        self.tools = [search_subreddits, search_subreddit_content, google_grounding_search, get_current_date]
        self.chat_with_tools = self.llm.bind_tools(self.tools)
        self.agent = self._build_agent()
        
        # Initialize logger
        if self.config["logging"]["enabled"]:
            # Note: We instantiate the logger but use the static method for logging
            # This could be refactored to use the instance logger if desired
            self.logger = ContentCreatorLogger(self.config["logging"]["log_file"])

    def _build_agent(self):
        """Build the LangGraph agent workflow"""

        class AgentState(TypedDict):
            messages: Annotated[list[AnyMessage], add_messages]

        def assistant(state: AgentState):
            """Main assistant node"""
            messages = state["messages"]
            
            # Add system message if not already present
            if not messages or not isinstance(messages[0], SystemMessage):
                messages = [SystemMessage(content=self.system_message)] + messages
            
            return {
                "messages": [self.chat_with_tools.invoke(messages)],
            }
    
        # Build the graph
        builder = StateGraph(AgentState)
        
        # Define nodes
        builder.add_node("assistant", assistant)
        builder.add_node("tools", ToolNode(self.tools))
        
        # Define edges
        builder.add_edge(START, "assistant")
        builder.add_conditional_edges(
            "assistant",
            tools_condition,
        )
        builder.add_edge("tools", "assistant")
        
        return builder.compile()

    def chat(self, message):
        """Chat with the agent and log each run with metrics and tool calls."""
        current_date = datetime.utcnow().isoformat()
        
        self.memory.append(SystemMessage(content=f"Today's date is {current_date}"))
        self.memory.append(HumanMessage(content=message))
        start_time = time.time()
        
        run_log = {
            "user_message": message,
            "tool_calls": [],
            "error": None,
            "token_usage": None,
            "latency": None,
            "success": False
        }
        
        try:
            result = self.agent.invoke({"messages": self.memory})
            messages = result["messages"]
            agent_response = messages[-1]
            self.memory.append(agent_response)

            # Extract ToolMessage objects and log them
            for msg in messages:
                if isinstance(msg, ToolMessage):
                    run_log["tool_calls"].append({
                        "tool_name": msg.name,
                        "tool_call_id": getattr(msg, "tool_call_id", None),
                        "content": msg.content,
                        "id": getattr(msg, "id", None)
                    })

            # Extract token usage from Gemini usage_metadata if available
            usage_metadata = getattr(agent_response, "usage_metadata", None)
            if usage_metadata and "total_tokens" in usage_metadata:
                run_log["token_usage"] = usage_metadata["total_tokens"]
            
            run_log["agent_response"] = agent_response.content
            run_log["success"] = True
            
        except Exception as e:
            run_log["error"] = traceback.format_exc()
            run_log["agent_response"] = f"[ERROR] {str(e)}"
            
        finally:
            run_log["latency"] = round(time.time() - start_time, 3)
            if self.config["logging"]["enabled"]:
                ContentCreatorLogger.log_reddit_run(run_log)

        return run_log["agent_response"]

    def interactive_chat(self):
        """Start interactive chat session with memory"""
        print("🤖 Reddit Agent Ready!")
        print(f"📊 Logging: {'Enabled' if self.config['logging']['enabled'] else 'Disabled'}")
        if self.config["logging"]["enabled"]:
            print(f"📁 Log File: {self.config['logging']['log_file']}")
        print(f"🤖 Model: {self.config['model']['name']} (temp: {self.config['model']['temperature']})")
        print("Type 'quit' to exit.")
        print("-" * 60)
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                if not user_input:
                    continue
                response = self.chat(user_input)
                print(f"Agent: {response}")
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"An error occurred: {str(e)}")

# Simple test function
if __name__ == "__main__":
    try:
        agent = RedditAgent()
        agent.interactive_chat()
    except Exception as e:
        print(f"❌ Error initializing Reddit Agent: {str(e)}")
        print("Make sure your .env file and config.py are set up correctly.")
