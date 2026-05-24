"""
Compatibility wrapper for Content Creator Logger.
Redirects logs and analytic reports to the new validated AppLogger system.
"""
from typing import Dict, Any, List, Optional
from app.core.logger import AppLogger

class ContentCreatorLogger:
    """Wrapper class pointing to the refactored AppLogger implementation."""
    
    def __init__(self, log_file: str = "content_creator_logs.json"):
        self.logger = AppLogger(log_file)
        self.log_file = log_file
        self.session_id = self.logger.session_id
        self.log_path = self.logger.log_path

    @staticmethod
    def log_content_creation(run_data: Dict[str, Any]) -> None:
        AppLogger().log_content_creation(run_data)

    @staticmethod
    def log_research_call(research_data: Dict[str, Any]) -> None:
        AppLogger().log_research_call(research_data)

    @staticmethod
    def log_tool_usage(tool_data: Dict[str, Any]) -> None:
        AppLogger().log_tool_usage(tool_data)

    @staticmethod
    def log_error(error_data: Dict[str, Any]) -> None:
        AppLogger().log_error(error_data)

    @staticmethod
    def log_performance_metrics(metrics_data: Dict[str, Any]) -> None:
        AppLogger().log_performance_metrics(metrics_data)

    @staticmethod
    def log_reddit_run(run_data: Dict[str, Any]) -> None:
        AppLogger().log_reddit_run(run_data)

    def get_logs(
        self,
        log_type: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        return self.logger.get_logs(
            log_type=log_type,
            session_id=session_id,
            limit=limit,
            start_date=start_date,
            end_date=end_date
        )

    @staticmethod
    def get_analytics() -> Dict[str, Any]:
        return AppLogger().get_analytics()

    @staticmethod
    def clear_logs(confirm: bool = False) -> bool:
        return AppLogger().clear_logs(confirm)

# Helper functions
def log_content_run(
    user_message: str,
    topic: str,
    platform: str,
    content_type: str,
    duration: str,
    tone: str,
    agent_response: str,
    tool_calls: List[Dict[str, Any]] = None,
    latency: float = None,
    token_usage: int = None,
    error: str = None
):
    from app.core.logger import AppLogger
    run_data = {
        "user_message": user_message,
        "topic": topic,
        "platform": platform,
        "content_type": content_type,
        "duration": duration,
        "tone": tone,
        "agent_response": agent_response,
        "tool_calls": tool_calls or [],
        "latency": latency,
        "token_usage": token_usage,
        "error": error,
        "success": error is None
    }
    AppLogger().log_content_creation(run_data)

def log_research_request(topic: str, platform_focus: str, results: str, latency: float = None):
    from app.core.logger import AppLogger
    research_data = {
        "topic": topic,
        "platform_focus": platform_focus,
        "results_length": len(results) if results else 0,
        "results_preview": results[:200] if results else "",
        "latency": latency,
        "success": results is not None
    }
    AppLogger().log_research_call(research_data)