import os
import time
import traceback
from typing import List, Dict, Any, Optional, Annotated
from pydantic import BaseModel, Field

from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.graph.message import add_messages

from app.tools.reddit_tools import search_subreddits, search_subreddit_content
from app.tools.base_tools import google_grounding_search, get_current_date
from app.core.config import get_reddit_agent_config, get_shared_config
from app.core.logger import AppLogger
from app.schemas.agent_schemas import ResearchBrief, TrendReport

class AgentState(BaseModel):
    messages: Annotated[list, add_messages] = Field(default_factory=list)

class RedditAgent:
    def __init__(self):
        """Initialize Reddit research agent with Gemini, tools, and logger."""
        self.config = get_reddit_agent_config()
        self.shared_config = get_shared_config()
        
        # Verify API Keys
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
            
        for var in self.shared_config.required_env_vars:
            if not os.getenv(var):
                raise ValueError(f"{var} environment variable not set")

        self.system_message = self.config.system_prompt
        
        # Initialize LLM
        model_cfg = self.config.model
        llm_kwargs = {
            "model": model_cfg.name or "gemini-2.5-flash",
            "temperature": model_cfg.temperature,
            "api_key": self.api_key
        }
        if model_cfg.max_tokens:
            llm_kwargs["max_tokens"] = model_cfg.max_tokens
        if model_cfg.top_p:
            llm_kwargs["top_p"] = model_cfg.top_p
            
        self.llm = ChatGoogleGenerativeAI(**llm_kwargs)
        self.tools = [search_subreddits, search_subreddit_content, google_grounding_search, get_current_date]
        self.chat_with_tools = self.llm.bind_tools(self.tools)
        
        # Build LangGraph workflow
        self.agent = self._build_agent()
        
        # Initialize Logger
        self.logger = AppLogger("reddit_agent_logs.json")

    def _build_agent(self):
        """Compile the search reasoning loop in LangGraph."""
        
        def assistant(state: AgentState):
            messages = state.messages
            # Ensure system prompt is present
            if not messages or not isinstance(messages[0], SystemMessage):
                messages = [SystemMessage(content=self.system_message)] + messages
            
            response = self.chat_with_tools.invoke(messages)
            return {"messages": [response]}
            
        builder = StateGraph(AgentState)
        builder.add_node("assistant", assistant)
        builder.add_node("tools", ToolNode(self.tools))
        
        builder.add_edge(START, "assistant")
        builder.add_conditional_edges("assistant", tools_condition)
        builder.add_edge("tools", "assistant")
        
        return builder.compile()

    def run_search_loop(self, query: str) -> List[AnyMessage]:
        """Execute the tool-search loop to gather data on a query."""
        initial_state = {
            "messages": [
                SystemMessage(content=self.system_message),
                HumanMessage(content=f"Research request details: {query}")
            ]
        }
        result = self.agent.invoke(initial_state)
        return result["messages"]

    def research(self, research_prompt: str) -> Dict[str, Any]:
        """Collect search facts and return a structured Pydantic ResearchBrief."""
        start_time = time.time()
        run_log = {
            "prompt": research_prompt,
            "tool_calls": [],
            "error": None,
            "success": False,
            "latency": None
        }
        
        try:
            # Run the search reasoning loop to collect sources/opinions
            messages = self.run_search_loop(research_prompt)
            
            # Log any tool calls executed during the run
            for msg in messages:
                if isinstance(msg, ToolMessage):
                    run_log["tool_calls"].append({
                        "tool_name": msg.name,
                        "content": msg.content[:300] + "..." if len(msg.content) > 300 else msg.content
                    })

            # Synthesis step: Take message logs and project onto structured output
            synthesis_prompt = f"""Review the following research logs containing search results and Reddit discussions.
Synthesize the information into the requested schema. Ensure all facts are sourced from the research logs, sentiment is accurate, and proposed content angles are highly engaging.

RESEARCH LOGS:
{self._format_messages_for_synthesis(messages)}
"""
            # Use structured output
            structured_llm = self.llm.with_structured_output(ResearchBrief)
            brief: ResearchBrief = structured_llm.invoke([
                SystemMessage(content="You are a Senior Research Analyst. Synthesize the findings into a structured report."),
                HumanMessage(content=synthesis_prompt)
            ])
            
            run_log["success"] = True
            run_log["result"] = brief.model_dump()
            return brief.model_dump()
            
        except Exception as e:
            run_log["error"] = traceback.format_exc()
            raise e
        finally:
            run_log["latency"] = round(time.time() - start_time, 3)
            if self.config.logging.enabled:
                self.logger.log_reddit_run(run_log)

    def research_trends(self, trending_prompt: str) -> Dict[str, Any]:
        """Collect search trends and return a structured Pydantic TrendReport."""
        start_time = time.time()
        run_log = {
            "prompt": trending_prompt,
            "tool_calls": [],
            "error": None,
            "success": False,
            "latency": None
        }
        
        try:
            messages = self.run_search_loop(trending_prompt)
            
            for msg in messages:
                if isinstance(msg, ToolMessage):
                    run_log["tool_calls"].append({
                        "tool_name": msg.name,
                        "content": msg.content[:300] + "..." if len(msg.content) > 300 else msg.content
                    })

            synthesis_prompt = f"""Review the trending topics gathered in these logs.
Synthesize the findings into the requested TrendReport schema.

TREND RESEARCH LOGS:
{self._format_messages_for_synthesis(messages)}
"""
            structured_llm = self.llm.with_structured_output(TrendReport)
            report: TrendReport = structured_llm.invoke([
                SystemMessage(content="You are a Senior Research Analyst specializing in digital trends. Synthesize the findings."),
                HumanMessage(content=synthesis_prompt)
            ])
            
            run_log["success"] = True
            run_log["result"] = report.model_dump()
            return report.model_dump()
            
        except Exception as e:
            run_log["error"] = traceback.format_exc()
            raise e
        finally:
            run_log["latency"] = round(time.time() - start_time, 3)
            if self.config.logging.enabled:
                self.logger.log_reddit_run(run_log)

    def chat(self, message: str) -> str:
        """Standard conversational chat interface returning raw text."""
        try:
            messages = self.run_search_loop(message)
            # Find last assistant message
            for msg in reversed(messages):
                if isinstance(msg, AIMessage) and not msg.tool_calls:
                    return msg.content
            return "No text response could be synthesized."
        except Exception as e:
            return f"[ERROR] Failed to chat: {str(e)}"

    def _format_messages_for_synthesis(self, messages: List[AnyMessage]) -> str:
        """Helper to print history for synthesis prompt."""
        formatted = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                continue
            elif isinstance(msg, HumanMessage):
                formatted.append(f"User: {msg.content}")
            elif isinstance(msg, AIMessage):
                if msg.content:
                    formatted.append(f"Agent: {msg.content}")
                if msg.tool_calls:
                    formatted.append(f"Agent decided to run tools: {[tc['name'] for tc in msg.tool_calls]}")
            elif isinstance(msg, ToolMessage):
                formatted.append(f"Tool Result ({msg.name}): {msg.content}")
        return "\n".join(formatted)
