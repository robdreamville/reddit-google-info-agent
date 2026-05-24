import json
import os
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from app.schemas.log_schemas import LogEntry

class AppLogger:
    """Central Pydantic-powered logging system for agent runs and tool metrics."""
    
    def __init__(self, log_file: str = "content_creator_logs.json"):
        self.log_file = log_file
        self.session_id = str(uuid.uuid4())[:8]
        
        os.makedirs("logs", exist_ok=True)
        self.log_path = os.path.join("logs", log_file)
        
        # Initialize file
        if not os.path.exists(self.log_path):
            with open(self.log_path, 'w') as f:
                json.dump([], f)
                
    def _write_log(self, data: Dict[str, Any], log_type: str) -> None:
        """Validate with Pydantic and write to file."""
        try:
            # Read current logs
            logs = []
            if os.path.exists(self.log_path):
                try:
                    with open(self.log_path, 'r', encoding='utf-8') as f:
                        logs = json.load(f)
                except Exception:
                    logs = []

            # Create Pydantic log entry
            log_entry = LogEntry(
                timestamp=datetime.now(timezone.utc).isoformat(),
                session_id=self.session_id,
                log_type=log_type,
                data=data
            )
            
            # Serialize model and append
            logs.append(log_entry.model_dump())
            
            # Write back
            with open(self.log_path, 'w', encoding='utf-8') as f:
                json.dump(logs, f, indent=2, default=str)
                
        except Exception as e:
            print(f"Logging error: {str(e)}")
            print(f"Fallback log: type={log_type}, data={data}")

    def log_content_creation(self, run_data: Dict[str, Any]) -> None:
        self._write_log(run_data, log_type="content_creation")
        
    def log_research_call(self, research_data: Dict[str, Any]) -> None:
        self._write_log(research_data, log_type="research")
        
    def log_tool_usage(self, tool_data: Dict[str, Any]) -> None:
        self._write_log(tool_data, log_type="tool_usage")
        
    def log_error(self, error_data: Dict[str, Any]) -> None:
        self._write_log(error_data, log_type="error")
        
    def log_performance_metrics(self, metrics_data: Dict[str, Any]) -> None:
        self._write_log(metrics_data, log_type="performance")
        
    def log_reddit_run(self, run_data: Dict[str, Any]) -> None:
        # Save Reddit runs to its own file or keep them in the main file
        # Using a dedicated Reddit log file for cleanliness
        reddit_logger = AppLogger("reddit_agent_logs.json")
        reddit_logger._write_log(run_data, log_type="reddit_agent_run")
        
    def get_logs(
        self,
        log_type: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve and filter logs."""
        try:
            if not os.path.exists(self.log_path):
                return []
                
            with open(self.log_path, 'r', encoding='utf-8') as f:
                logs = json.load(f)
            
            filtered = logs
            if log_type:
                filtered = [l for l in filtered if l.get("log_type") == log_type]
            if session_id:
                filtered = [l for l in filtered if l.get("session_id") == session_id]
            if start_date:
                filtered = [l for l in filtered if l.get("timestamp", "") >= start_date]
            if end_date:
                filtered = [l for l in filtered if l.get("timestamp", "") <= end_date]
                
            # Sort newest first
            filtered.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            
            if limit:
                filtered = filtered[:limit]
                
            return filtered
        except Exception as e:
            print(f"Error fetching logs: {e}")
            return []

    def get_analytics(self) -> Dict[str, Any]:
        """Aggregate stats from logs for dashboard consumption."""
        try:
            if not os.path.exists(self.log_path):
                return {"message": "No logs found"}
                
            with open(self.log_path, 'r', encoding='utf-8') as f:
                logs = json.load(f)
                
            creation_logs = [l for l in logs if l.get("log_type") == "content_creation"]
            research_logs = [l for l in logs if l.get("log_type") == "research"]
            error_logs = [l for l in logs if l.get("log_type") == "error"]
            
            total_runs = len(creation_logs)
            total_research = len(research_logs)
            total_errors = len(error_logs)
            
            platforms = {}
            content_types = {}
            latencies = []
            token_usage = []
            success_count = 0
            
            for log in creation_logs:
                data = log.get("data", {})
                
                # Check success
                if data.get("success", False):
                    success_count += 1
                
                # Platforms counts
                requested_platforms = data.get("platforms", [])
                for plat in requested_platforms:
                    platforms[plat] = platforms.get(plat, 0) + 1
                    
                # Content type count
                ctype = data.get("content_type", "unknown")
                content_types[ctype] = content_types.get(ctype, 0) + 1
                
                # Performance metrics
                latency = data.get("latency")
                if latency is not None:
                    latencies.append(latency)
                    
                tokens = data.get("token_usage")
                if tokens is not None:
                    token_usage.append(tokens)
                    
            avg_latency = sum(latencies) / len(latencies) if latencies else 0
            avg_tokens = sum(token_usage) / len(token_usage) if token_usage else 0
            success_rate = (success_count / total_runs * 100) if total_runs else 0
            
            return {
                "summary": {
                    "total_content_created": total_runs,
                    "total_research_calls": total_research,
                    "total_errors": total_errors,
                    "success_rate": round(success_rate, 2),
                },
                "platform_distribution": platforms,
                "content_type_distribution": content_types,
                "performance": {
                    "average_latency_seconds": round(avg_latency, 3),
                    "average_token_usage": round(avg_tokens, 0),
                },
                "date_range": {
                    "first_log": logs[0].get("timestamp") if logs else None,
                    "last_log": logs[-1].get("timestamp") if logs else None
                }
            }
        except Exception as e:
            return {"error": f"Error compiling analytics: {str(e)}"}
            
    def clear_logs(self, confirm: bool = False) -> bool:
        if not confirm:
            return False
        try:
            with open(self.log_path, 'w', encoding='utf-8') as f:
                json.dump([], f)
            return True
        except Exception as e:
            print(f"Error clearing logs: {str(e)}")
            return False
