"""
Compatibility wrapper for configuration settings.
Routes config retrieval and preset applications to the new Pydantic setup in app/core/config.py.
"""
from app.core.config import (
    get_reddit_agent_config as _get_reddit_agent_config,
    get_content_creator_config as _get_content_creator_config,
    get_shared_config as _get_shared_config,
    get_tool_prompt as _get_tool_prompt,
    update_agent_config as _update_agent_config,
    apply_preset as _apply_preset
)

def get_reddit_agent_config():
    """Wrapper mapping to Pydantic config model dump."""
    return _get_reddit_agent_config().model_dump()

def get_content_creator_config():
    """Wrapper mapping to Pydantic config model dump."""
    return _get_content_creator_config().model_dump()

def get_shared_config():
    """Wrapper mapping to Pydantic config model dump."""
    return _get_shared_config().model_dump()

def get_tool_prompt(prompt_name: str, **kwargs) -> str:
    return _get_tool_prompt(prompt_name, **kwargs)

def update_config(agent_type: str, section: str, key: str, value):
    updates = {section: {key: value}}
    _update_agent_config(agent_type, updates)

def apply_preset(preset_name: str):
    _apply_preset(preset_name)
