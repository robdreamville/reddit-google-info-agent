from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import asyncio
import json
import threading
from pathlib import Path

from app.core.config import (
    load_app_config, save_app_config, update_agent_config, apply_preset
)
from app.core.logger import AppLogger
from app.agents.content_creator import ContentCreatorAgent

router = APIRouter()
logger = AppLogger("content_creator_logs.json")

class ConfigUpdateModel(BaseModel):
    agent_type: str  # "reddit", "content_creator", "shared"
    settings: Dict[str, Any]

class ContentRequest(BaseModel):
    topic: str
    platforms: List[str]
    content_type: str = "educational"
    duration: Optional[str] = None
    tone: str = "engaging"
    custom_instructions: Optional[str] = None
    temperature_override: Optional[float] = None
    system_prompt_override: Optional[str] = None

@router.get("/health")
def health_check():
    """Verify system health, env variables, and API status."""
    import os
    env_vars = ["GEMINI_API_KEY", "REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET"]
    missing = [var for var in env_vars if not os.getenv(var)]
    
    return {
        "status": "healthy" if not missing else "unconfigured",
        "missing_env_vars": missing,
        "active_model": load_app_config().content_creator.model.name,
        "timestamp": json_now_iso()
    }

@router.get("/config")
def get_config():
    """Get active application config."""
    return load_app_config().model_dump()

@router.post("/config")
def update_config(update_data: ConfigUpdateModel):
    """Update active configuration."""
    try:
        updated = update_agent_config(update_data.agent_type, update_data.settings)
        return updated.model_dump()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/config/preset/{preset_name}")
def apply_config_preset(preset_name: str):
    """Apply a preset config."""
    try:
        updated = apply_preset(preset_name)
        return updated.model_dump()
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.get("/logs")
def get_logs(
    log_type: Optional[str] = None, 
    limit: Optional[int] = 200,
    topic: Optional[str] = None,
    platform: Optional[str] = None,
    status: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """Fetch history logs with optional filters."""
    logs = logger.get_logs(
        log_type=log_type, 
        limit=None, # We'll limit after filtering
        start_date=start_date,
        end_date=end_date
    )
    
    # Filter content creation logs based on query parameters
    if log_type == "content_creation":
        if topic:
            logs = [l for l in logs if topic.lower() in l.get("data", {}).get("topic", "").lower()]
        
        if platform and platform != "all":
            logs = [l for l in logs if platform in l.get("data", {}).get("platforms", [])]

        if status and status != "all":
            is_success = (status == "success")
            logs = [l for l in logs if l.get("data", {}).get("success") == is_success]
    
    # Process files and apply limit *after* filtering
    processed_logs = []
    base_dir = Path(__file__).resolve().parents[2]
    
    for log in logs:
        if log.get("log_type") == "content_creation":
            data = log.get("data", {})
            files_saved = data.get("files_saved", [])
            platforms_list = data.get("platforms", [])
            content_payload = {}

            for i, item in enumerate(files_saved):
                try:
                    file_path_str = None
                    if isinstance(item, str):
                        prefix = "Successfully saved content to "
                        if item.startswith(prefix):
                            file_path_str = item[len(prefix):].strip()
                        else:
                            file_path_str = item.strip()
                    
                    if not file_path_str:
                        continue

                    file_path_obj = Path(file_path_str)
                    if not file_path_obj.is_absolute():
                        file_path_obj = base_dir / file_path_obj
                    
                    if file_path_obj.exists() and i < len(platforms_list):
                        platform = platforms_list[i]
                        content_payload[platform] = file_path_obj.read_text(encoding='utf-8', errors='ignore')
                
                except Exception as e:
                    print(f"Could not read or process log file entry: {item}, Error: {e}")
                    continue
            
            if content_payload:
                data["generated_content"] = content_payload

        processed_logs.append(log)

    return processed_logs[:limit]

@router.delete("/logs")
def clear_logs():
    """Clear history logs."""
    success = logger.clear_logs(confirm=True)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to clear logs")
    return {"message": "Logs cleared successfully"}

@router.get("/analytics")
def get_analytics():
    """Fetch usage analytics metrics."""
    return logger.get_analytics()

@router.post("/generate")
def generate_content_sync(req: ContentRequest):
    """Generate content synchronously (fallback)."""
    try:
        creator = ContentCreatorAgent()
        result = creator.create_content(
            topic=req.topic,
            platforms=req.platforms,
            content_type=req.content_type,
            duration=req.duration,
            tone=req.tone,
            custom_instructions=req.custom_instructions,
            temperature_override=req.temperature_override,
            system_prompt_override=req.system_prompt_override
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/generate/stream")
async def generate_content_stream(
    topic: str,
    platforms: str,  # Comma separated
    content_type: str = "educational",
    duration: Optional[str] = None,
    tone: str = "engaging",
    target_audience: str = "general",
    output_format: str = "platform-native",
    custom_instructions: Optional[str] = None,
    temperature: Optional[float] = None,
    system_prompt: Optional[str] = None
):
    """
    SSE stream for content generation.
    Pushes step progress in real-time, finishing with the full JSON results payload.
    """
    platform_list = [p.strip() for p in platforms.split(",") if p.strip()]
    if not platform_list:
        raise HTTPException(status_code=400, detail="At least one platform is required")
        
    main_loop = asyncio.get_running_loop()
    queue = asyncio.Queue()

    # Step callback helper that works across threads
    def step_callback(step_name: str):
        main_loop.call_soon_threadsafe(queue.put_nowait, {"type": "step", "message": step_name})

    def run_pipeline(loop):
        # Set the event loop for this new thread
        asyncio.set_event_loop(asyncio.new_event_loop())
        
        try:
            creator = ContentCreatorAgent()
            result = creator.create_content(
                topic=topic,
                platforms=platform_list,
                content_type=content_type,
                duration=duration,
                tone=tone,
                target_audience=target_audience,
                output_format=output_format,
                custom_instructions=custom_instructions,
                temperature_override=temperature,
                system_prompt_override=system_prompt,
                active_step_callback=step_callback
            )
            loop.call_soon_threadsafe(queue.put_nowait, {"type": "result", "data": result})
        except Exception as e:
            loop.call_soon_threadsafe(queue.put_nowait, {"type": "error", "message": str(e)})

    # Pass the main loop to the thread
    threading.Thread(target=run_pipeline, args=(main_loop,), daemon=True).start()

    async def event_generator():
        while True:
            item = await queue.get()
            yield f"data: {json.dumps(item)}\n\n"
            
            # Close stream on final actions
            if item["type"] in ["result", "error"]:
                break

    return StreamingResponse(event_generator(), media_type="text/event-stream")

def json_now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
