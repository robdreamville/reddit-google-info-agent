import os
import time
import traceback
import json
from typing import List, Dict, Any, Optional

from langchain_core.messages import SystemMessage, HumanMessage
from app.core.config import get_content_creator_config, get_shared_config
from app.core.logger import AppLogger
from app.tools.content_tools import (
    research_topic_for_content,
    research_trending_topics,
    generate_platform_content,
    generate_article,
    generate_x_thread,
    analyze_content_performance,
    run_quality_checks,
    get_llm
)
from app.tools.base_tools import save_content_to_file
from app.schemas.agent_schemas import VideoScript, Article, XThread, ContentAnalysis, ResearchBrief

class ContentCreatorAgent:
    def __init__(self):
        """Initialize Content Creator Agent with configuration and logger."""
        self.config = get_content_creator_config()
        self.shared_config = get_shared_config()
        
        # Verify keys
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
            
        for var in self.shared_config.required_env_vars:
            if not os.getenv(var):
                raise ValueError(f"{var} environment variable not set")

        self.logger = AppLogger("content_creator_logs.json")

    def create_content(
        self,
        topic: str,
        platforms: List[str],
        content_type: str = "educational",
        duration: Optional[str] = None,
        tone: str = "engaging",
        target_audience: str = "general",
        output_format: str = "platform-native",
        custom_instructions: Optional[str] = None,
        temperature_override: Optional[float] = None,
        system_prompt_override: Optional[str] = None,
        active_step_callback: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Orchestrate the structured content generation pipeline.
        
        Args:
            topic: The content topic.
            platforms: List of target platforms, e.g. ["youtube", "article"].
            content_type: E.g., educational, how-to.
            duration: Video length.
            tone: Creative tone.
            custom_instructions: Custom tweaking instructions per-run.
            temperature_override: Override model temperature.
            system_prompt_override: Override model system prompt.
            active_step_callback: Async/sync callback to notify dashboard of current task.
        """
        start_time = time.time()
        
        # Determine parameters
        temp = temperature_override if temperature_override is not None else self.config.model.temperature
        sys_prompt = system_prompt_override if system_prompt_override is not None else self.config.system_prompt
        
        # Setup logging metadata
        run_log = {
            "topic": topic,
            "platforms": platforms,
            "content_type": content_type,
            "duration": duration,
            "tone": tone,
            "target_audience": target_audience,
            "output_format": output_format,
            "custom_instructions": custom_instructions,
            "temperature": temp,
            "tool_calls": [],
            "files_saved": [],
            "generated_content": {},
            "quality_checks": {},
            "token_usage": 0,
            "latency": None,
            "success": False,
            "error": None
        }
        
        final_result = {
            "topic": topic,
            "content_type": content_type,
            "tone": tone,
            "target_audience": target_audience,
            "output_format": output_format,
            "generated_at": datetime_now_iso(),
            "content": {},
            "analyses": {},
            "quality_checks": {},
            "files": {},
            "success": False
        }

        def notify_step(step_name: str):
            if active_step_callback:
                try:
                    active_step_callback(step_name)
                except Exception:
                    pass

        try:
            # Step 1: Research Topic
            notify_step("🔍 Researching topic with Reddit and Google scan...")
            research_focus = "both"
            if len(platforms) == 1:
                research_focus = platforms[0]
            
            research_json = research_topic_for_content.invoke({
                "topic": topic,
                "platform_focus": research_focus
            })
            
            # Sum tokens if logged by research agent
            # For simplicity, we also count LLM synthesis tokens inside the generator tools
            run_log["tool_calls"].append({
                "tool": "research_topic_for_content",
                "status": "completed"
            })
            
            # Optional: Append custom instructions to the research summary so generator LLMs see it
            research_summary_to_use = research_json
            if custom_instructions:
                research_summary_to_use = (
                    f"RESEARCH SUMMARY:\n{research_json}\n\n"
                    f"USER CUSTOM INSTRUCTIONS (TWEAKS):\n{custom_instructions}"
                )

            # Step 2: Loop Platforms and Generate Content
            for plat in platforms:
                plat_lower = plat.lower()
                notify_step(f"🎬 Generating structured content for {plat_lower.upper()}...")
                
                content_json = ""
                pretty_text = ""
                
                if plat_lower in ["youtube", "tiktok"]:
                    # Run generator tool which returns VideoScript JSON string
                    content_json = generate_platform_content.invoke({
                        "topic": topic,
                        "platform": plat_lower,
                        "research_summary": research_summary_to_use,
                        "content_type": content_type,
                        "tone": tone,
                        "target_audience": target_audience,
                        "output_format": output_format
                    })
                    
                    # Validate via Pydantic & format a nice display string
                    try:
                        parsed = VideoScript.model_validate_json(content_json)
                        # Build pretty string for files / display
                        segs = []
                        for s in parsed.segments:
                            emp = f" (Emphasis: {', '.join(s.vocal_emphasis)})" if s.vocal_emphasis else ""
                            pause = " [PAUSE]" if s.is_pause_after else ""
                            segs.append(f"[{s.time_cue}] {s.narration}{emp}{pause}\nVisuals: {s.visual_cue}")
                        
                        pretty_text = (
                            f"=== {plat_lower.upper()} SCRIPT: {parsed.topic.upper()} ===\n\n"
                            f"HOOK:\n{parsed.hook}\n\n"
                            f"SEGMENTS:\n" + "\n\n".join(segs) + f"\n\n"
                            f"CALL TO ACTION:\n{parsed.call_to_action}"
                        )
                        final_result["content"][plat_lower] = parsed.model_dump()
                    except Exception:
                        pretty_text = content_json  # Fallback to raw json if parse fails
                        final_result["content"][plat_lower] = {"raw_output": content_json}

                elif plat_lower == "article":
                    plat_spec = self.config.platform_specs.get("article")
                    style = plat_spec.style if plat_spec else ""
                    opt_len = plat_spec.optimal_length if plat_spec else "500-800 words"
                    
                    content_json = generate_article.invoke({
                        "topic": topic,
                        "research_summary": research_summary_to_use,
                        "tone": tone,
                        "style": style,
                        "optimal_length": opt_len,
                        "target_audience": target_audience,
                        "output_format": output_format
                    })
                    
                    try:
                        parsed = Article.model_validate_json(content_json)
                        sec_texts = []
                        for s in parsed.sections:
                            sec_texts.append(f"## {s.heading}\n\n{s.content}")
                            
                        pretty_text = (
                            f"# {parsed.title}\n\n"
                            f"{parsed.introduction}\n\n" + 
                            "\n\n".join(sec_texts) + f"\n\n"
                            f"## Conclusion\n\n{parsed.conclusion}\n\n"
                            f"--- \n"
                            f"*SEO Keywords: {', '.join(parsed.seo_keywords)}*\n"
                            f"*Meta Description: {parsed.meta_description}*"
                        )
                        final_result["content"][plat_lower] = parsed.model_dump()
                    except Exception:
                        pretty_text = content_json
                        final_result["content"][plat_lower] = {"raw_output": content_json}

                elif plat_lower == "x":
                    plat_spec = self.config.platform_specs.get("x")
                    style = plat_spec.style if plat_spec else ""
                    thread_len = plat_spec.thread_length if plat_spec else "3-5"
                    
                    content_json = generate_x_thread.invoke({
                        "topic": topic,
                        "research_summary": research_summary_to_use,
                        "tone": tone,
                        "style": style,
                        "thread_length": thread_len,
                        "target_audience": target_audience,
                        "output_format": output_format
                    })
                    
                    try:
                        parsed = XThread.model_validate_json(content_json)
                        tweets = []
                        for t in parsed.thread:
                            tweets.append(f"({t.index}/{len(parsed.thread)})\n{t.text}")
                        
                        pretty_text = (
                            "=== X THREAD ===\n\n" + 
                            "\n\n---\n\n".join(tweets) + f"\n\n"
                            f"{' '.join(parsed.hashtags)}"
                        )
                        final_result["content"][plat_lower] = parsed.model_dump()
                    except Exception:
                        pretty_text = content_json
                        final_result["content"][plat_lower] = {"raw_output": content_json}

                # Save Pretty content
                notify_step(f"💾 Saving generated {plat_lower.upper()} file...")
                folder = self.config.output_paths.get(f"{plat_lower}s", "output")
                if plat_lower == "article":
                    folder = self.config.output_paths.get("articles", "articles")
                elif plat_lower == "x":
                    folder = self.config.output_paths.get("x_threads", "x_threads")
                
                save_path = save_content_to_file.invoke({
                    "content": pretty_text,
                    "folder": folder,
                    "topic": topic,
                    "platform": plat_lower
                })
                
                final_result["files"][plat_lower] = save_path
                run_log["files_saved"].append(save_path)
                run_log["generated_content"][plat_lower] = pretty_text
                
                # Step 3: Analyze content performance
                notify_step(f"🧐 Evaluating content quality and performance potential...")
                analysis_json = analyze_content_performance.invoke({
                    "content_text": pretty_text,
                    "platform": plat_lower
                })
                
                try:
                    parsed_analysis = ContentAnalysis.model_validate_json(analysis_json)
                    final_result["analyses"][plat_lower] = parsed_analysis.model_dump()
                except Exception:
                    final_result["analyses"][plat_lower] = {"raw_output": analysis_json}

                # Step 4: Run local quality checks and readability scoring
                quality_report_json = run_quality_checks.invoke({"content_text": pretty_text})
                quality_report = json.loads(quality_report_json)
                run_log["quality_checks"][plat_lower] = quality_report
                final_result["quality_checks"][plat_lower] = quality_report

            final_result["success"] = True
            run_log["success"] = True
            notify_step("✨ Content generation completed successfully!")
            
        except Exception as e:
            err_msg = str(e)
            run_log["error"] = traceback.format_exc()
            final_result["error"] = err_msg
            notify_step(f"❌ Error during generation: {err_msg}")
            
        finally:
            run_log["latency"] = round(time.time() - start_time, 3)
            # Estimate token usage if needed (can be fetched in a real model context, 
            # here we mock or add placeholder tokens for runs since LangChain core returns it in response metadata)
            run_log["token_usage"] = 4500 * len(platforms)  # Average run usage estimate for flash
            
            if self.config.logging.enabled:
                self.logger.log_content_creation(run_log)
                
        return final_result

    def research_trending(self, category: str = "general") -> Dict[str, Any]:
        """Research trending topics."""
        try:
            results_json = research_trending_topics.invoke({"category": category})
            return json.loads(results_json)
        except Exception as e:
            return {"error": f"Failed to get trends: {str(e)}", "category": category}

def datetime_now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
