from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from datetime import datetime

class ContentCreationData(BaseModel):
    topic: str
    platforms: List[str]
    content_type: str
    duration: Optional[str] = None
    tone: str
    custom_instructions: Optional[str] = None
    temperature: Optional[float] = None
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    files_saved: List[str] = Field(default_factory=list)
    generated_content: Optional[Dict[str, str]] = Field(default_factory=dict)
    token_usage: int = 0
    latency: Optional[float] = None
    success: bool = False
    error: Optional[str] = None
    agent_response: Optional[str] = None
    analyses: Optional[Dict[str, Any]] = None

class LogEntry(BaseModel):
    timestamp: str = Field(description="ISO format UTC timestamp of the log entry")
    session_id: str = Field(description="Unique session identifier for group tracking")
    log_type: str = Field(description="Type of the log entry (e.g. content_creation, research, error, etc.)")
    data: Dict[str, Any] = Field(description="Log payload data containing run metrics or error details")
