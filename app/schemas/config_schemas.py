from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any

class ModelConfig(BaseModel):
    name: str = Field(default="gemini-2.5-flash")
    temperature: float = Field(default=0.7)
    max_tokens: Optional[int] = Field(default=None)
    top_p: Optional[float] = Field(default=None)

class LoggingConfig(BaseModel):
    enabled: bool = Field(default=True)
    log_file: str = Field(default="content_creator_logs.json")
    log_level: str = Field(default="INFO")
    log_errors: bool = Field(default=False)
    separate_error_log: bool = Field(default=True)
    error_log_file: str = Field(default="content_creator_errors.json")

class SearchLimits(BaseModel):
    subreddit_search_limit: int = Field(default=8)
    content_search_limit: int = Field(default=8)

class PlatformSpec(BaseModel):
    hook_time: Optional[str] = Field(default=None)
    pace: Optional[str] = Field(default=None)
    style: str
    optimal_duration: Optional[str] = Field(default=None)
    optimal_length: Optional[str] = Field(default=None)
    thread_length: Optional[str] = Field(default=None)

class ContentTypeSpec(BaseModel):
    description: str
    structure: str

class RedditAgentConfig(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    system_prompt: str
    search_limits: SearchLimits = Field(default_factory=SearchLimits)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

class ContentCreatorConfig(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    system_prompt: str
    tool_prompts: Dict[str, str] = Field(default_factory=dict)
    platform_specs: Dict[str, PlatformSpec] = Field(default_factory=dict)
    output_paths: Dict[str, str] = Field(default_factory=dict)
    content_types: Dict[str, ContentTypeSpec] = Field(default_factory=dict)
    tone_settings: Dict[str, str] = Field(default_factory=dict)
    audience_settings: Dict[str, str] = Field(default_factory=dict)
    output_formats: Dict[str, str] = Field(default_factory=dict)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

class SharedConfig(BaseModel):
    required_env_vars: List[str] = Field(default_factory=list)
    api_timeout: int = Field(default=30)
    tool_timeout: int = Field(default=45)
    reddit_requests_per_minute: int = Field(default=60)
    google_requests_per_minute: int = Field(default=100)

class AppConfig(BaseModel):
    reddit_agent: RedditAgentConfig
    content_creator: ContentCreatorConfig
    shared: SharedConfig
