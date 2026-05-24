from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from datetime import datetime

class LogEntry(BaseModel):
    timestamp: str = Field(description="ISO format UTC timestamp of the log entry")
    session_id: str = Field(description="Unique session identifier for group tracking")
    log_type: str = Field(description="Type of the log entry (e.g. content_creation, research, error, etc.)")
    data: Dict[str, Any] = Field(description="Log payload data containing run metrics or error details")
