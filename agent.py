import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated, List
import time
import traceback

from langchain_core.messages import AnyMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage

from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from tools import search_subreddits, search_subreddit_content, google_grounding_search, get_current_date  # Import the tool
from logger import AgentLogger

from datetime import datetime


# ADDED: Load environment variables
load_dotenv()

class Agent:
    def __init__(self):
        """Initialize agent with Gemini and tools"""
        
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        # Default system message - you can change this easily

        self.system_message = """You are a helpful AI assistant specialized in searching and analyzing information using all available tools, including Google and Reddit.
For every query, first search for the latest news or authoritative sources using Google or other news tools.
Then, search Reddit to understand the general public's thoughts, discussions, and community sentiment about the topic.
Always combine and synthesize information from both news sources and Reddit to provide a comprehensive answer.
Your internal knowledge is outdated, so you must use these tools to find new, relevant, and up-to-date information for every user query.
Always think step by step, explain your reasoning, and use all available tools to verify, supplement, and combine your answers.
If you do not know the answer, do not give up—try again by rephrasing your query, searching for new or top content, or using different tools and sources.
Never rely solely on your own knowledge; always seek the latest information from multiple sources.
If your first attempt is incomplete, continue searching, combining, or synthesizing information until you have a thorough answer.
If you cannot find direct results, provide related information, explain why data may be missing, and suggest helpful next steps or alternative queries."""

        # ADDED: Memory to store conversation history
        # Starts with system message so agent knows its role from the beginning
        self.memory: List[AnyMessage] = [SystemMessage(content=self.system_message)]

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.4,
            api_key=self.api_key,
        )

        self.tools = [search_subreddits, search_subreddit_content, google_grounding_search, get_current_date]

        self.chat_with_tools = self.llm.bind_tools(self.tools)

        self.agent = self._build_agent()

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
            tools_condition,  # If tools needed, go to tools; otherwise end
        )
        builder.add_edge("tools", "assistant")
        
        return builder.compile()

    def chat(self, message):
        """Chat with the agent and log each run with metrics and tool calls."""
        import time, traceback
        from langchain_core.messages import ToolMessage

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
            else:
                run_log["token_usage"] = None

            run_log["agent_response"] = agent_response.content
        except Exception as e:
            run_log["error"] = traceback.format_exc()
            run_log["agent_response"] = f"[ERROR] {str(e)}"
        finally:
            run_log["latency"] = round(time.time() - start_time, 3)
            AgentLogger.log_run(run_log)

        return run_log["agent_response"]

    # ADDED: Interactive chat method for testing
    def interactive_chat(self):
        """Start interactive chat session with memory"""
        print("🤖 Agent ready! Type 'quit' to exit.")
        print("-" * 40)
        while True:
            try:
                user_input = input("\nYou: ").strip()
                # Exit commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                # Skip empty input
                if not user_input:
                    continue
                # Chat with memory
                response = self.chat(user_input)
                print(f"Agent: {response}")
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break

# Simple test function
if __name__ == "__main__":
    # Create agent
    agent = Agent()
    # ADDED: Start interactive mode instead of single test
    agent.interactive_chat()