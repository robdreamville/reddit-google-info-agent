"""
Logging system for Content Creator Agent
Separate from Reddit Agent logging for better organization and tracking
"""

import json
import os
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
import uuid

class ContentCreatorLogger:
    """Logger specifically for Content Creator Agent operations"""
    
    def __init__(self, log_file: str = "content_creator_logs.json"):
        self.log_file = log_file
        self.session_id = str(uuid.uuid4())[:8]  # Unique session identifier
        
        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)
        self.log_path = os.path.join("logs", log_file)
        
        # Initialize log file if it doesn't exist
        if not os.path.exists(self.log_path):
            with open(self.log_path, 'w') as f:
                json.dump([], f)
    
    @staticmethod
    def log_content_creation(run_data: Dict[str, Any]) -> None:
        """
        Log content creation runs with detailed metrics
        
        Args:
            run_data: Dictionary containing run information
        """
        logger = ContentCreatorLogger()
        logger._write_log(run_data, log_type="content_creation")
    
    @staticmethod 
    def log_research_call(research_data: Dict[str, Any]) -> None:
        """
        Log research tool calls
        
        Args:
            research_data: Dictionary containing research information
        """
        logger = ContentCreatorLogger()
        logger._write_log(research_data, log_type="research")
    
    @staticmethod
    def log_tool_usage(tool_data: Dict[str, Any]) -> None:
        """
        Log individual tool usage
        
        Args:
            tool_data: Dictionary containing tool usage information
        """
        logger = ContentCreatorLogger()
        logger._write_log(tool_data, log_type="tool_usage")
    
    @staticmethod
    def log_error(error_data: Dict[str, Any]) -> None:
        """
        Log errors and exceptions
        
        Args:
            error_data: Dictionary containing error information
        """
        logger = ContentCreatorLogger()
        logger._write_log(error_data, log_type="error")
    
    @staticmethod
    def log_performance_metrics(metrics_data: Dict[str, Any]) -> None:
        """
        Log performance metrics and analytics
        
        Args:
            metrics_data: Dictionary containing performance metrics
        """
        logger = ContentCreatorLogger()
        logger._write_log(metrics_data, log_type="performance")

    @staticmethod
    def log_reddit_run(run_data: Dict[str, Any]) -> None:
        """
        Log Reddit Agent runs
        
        Args:
            run_data: Dictionary containing run information
        """
        logger = ContentCreatorLogger("reddit_agent_logs.json")
        logger._write_log(run_data, log_type="reddit_agent_run")
    
    def _write_log(self, data: Dict[str, Any], log_type: str) -> None:
        """
        Write log entry to file
        
        Args:
            data: Data to log
            log_type: Type of log entry
        """
        try:
            # Load existing logs
            with open(self.log_path, 'r') as f:
                logs = json.load(f)
            
            # Create log entry
            log_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "session_id": self.session_id,
                "log_type": log_type,
                "data": data
            }
            
            # Add log entry
            logs.append(log_entry)
            
            # Write back to file
            with open(self.log_path, 'w') as f:
                json.dump(logs, f, indent=2, default=str)
                
        except Exception as e:
            # Fallback logging to console if file logging fails
            print(f"Logging error: {str(e)}")
            print(f"Failed to log: {json.dumps(data, default=str)}")
    
    @staticmethod
    def get_logs(
        log_type: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve logs with filtering options
        
        Args:
            log_type: Filter by log type
            session_id: Filter by session ID
            limit: Limit number of results
            start_date: Filter by start date (ISO format)
            end_date: Filter by end date (ISO format)
            
        Returns:
            List of log entries matching filters
        """
        logger = ContentCreatorLogger()
        
        try:
            with open(logger.log_path, 'r') as f:
                logs = json.load(f)
            
            # Apply filters
            filtered_logs = logs
            
            if log_type:
                filtered_logs = [log for log in filtered_logs if log.get("log_type") == log_type]
            
            if session_id:
                filtered_logs = [log for log in filtered_logs if log.get("session_id") == session_id]
            
            if start_date:
                filtered_logs = [log for log in filtered_logs if log.get("timestamp", "") >= start_date]
            
            if end_date:
                filtered_logs = [log for log in filtered_logs if log.get("timestamp", "") <= end_date]
            
            # Sort by timestamp (newest first)
            filtered_logs.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            
            # Apply limit
            if limit:
                filtered_logs = filtered_logs[:limit]
            
            return filtered_logs
            
        except Exception as e:
            print(f"Error retrieving logs: {str(e)}")
            return []
    
    @staticmethod
    def get_analytics() -> Dict[str, Any]:
        """
        Get analytics and summary statistics from logs
        
        Returns:
            Dictionary containing analytics data
        """
        logger = ContentCreatorLogger()
        
        try:
            with open(logger.log_path, 'r') as f:
                logs = json.load(f)
            
            if not logs:
                return {"message": "No logs found"}
            
            # Basic stats
            total_runs = len([log for log in logs if log.get("log_type") == "content_creation"])
            total_research_calls = len([log for log in logs if log.get("log_type") == "research"])
            total_errors = len([log for log in logs if log.get("log_type") == "error"])
            
            # Platform distribution
            platforms = {}
            content_types = {}
            
            for log in logs:
                if log.get("log_type") == "content_creation":
                    data = log.get("data", {})
                    
                    # Platform stats
                    platform = data.get("platform", "unknown")
                    platforms[platform] = platforms.get(platform, 0) + 1
                    
                    # Content type stats
                    content_type = data.get("content_type", "unknown")
                    content_types[content_type] = content_types.get(content_type, 0) + 1
            
            # Performance stats
            latencies = []
            token_usage = []
            
            for log in logs:
                if log.get("log_type") == "content_creation":
                    data = log.get("data", {})
                    if "latency" in data:
                        latencies.append(data["latency"])
                    if "token_usage" in data:
                        token_usage.append(data["token_usage"])
            
            avg_latency = sum(latencies) / len(latencies) if latencies else 0
            avg_tokens = sum(token_usage) / len(token_usage) if token_usage else 0
            
            return {
                "summary": {
                    "total_content_created": total_runs,
                    "total_research_calls": total_research_calls,
                    "total_errors": total_errors,
                    "error_rate": total_errors / max(total_runs, 1) * 100
                },
                "platform_distribution": platforms,
                "content_type_distribution": content_types,
                "performance": {
                    "average_latency_seconds": round(avg_latency, 3),
                    "average_token_usage": round(avg_tokens, 0),
                    "total_runs": total_runs
                },
                "date_range": {
                    "first_log": logs[0].get("timestamp") if logs else None,
                    "last_log": logs[-1].get("timestamp") if logs else None
                }
            }
            
        except Exception as e:
            return {"error": f"Error generating analytics: {str(e)}"}
    
    @staticmethod
    def clear_logs(confirm: bool = False) -> bool:
        """
        Clear all logs (use with caution)
        
        Args:
            confirm: Must be True to actually clear logs
            
        Returns:
            True if logs were cleared, False otherwise
        """
        if not confirm:
            return False
        
        logger = ContentCreatorLogger()
        
        try:
            with open(logger.log_path, 'w') as f:
                json.dump([], f)
            return True
        except Exception as e:
            print(f"Error clearing logs: {str(e)}")
            return False


# Helper functions for easier logging
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
    """Convenient function to log content creation runs"""
    
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
    
    ContentCreatorLogger.log_content_creation(run_data)

def log_research_request(topic: str, platform_focus: str, results: str, latency: float = None):
    """Convenient function to log research requests"""
    
    research_data = {
        "topic": topic,
        "platform_focus": platform_focus,
        "results_length": len(results) if results else 0,
        "results_preview": results[:200] if results else "",
        "latency": latency,
        "success": results is not None
    }
    
    ContentCreatorLogger.log_research_call(research_data)