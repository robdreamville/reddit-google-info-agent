import os
import json
from pathlib import Path
from dotenv import load_dotenv
from app.schemas.config_schemas import (
    AppConfig, RedditAgentConfig, ContentCreatorConfig, SharedConfig,
    ModelConfig, SearchLimits, LoggingConfig, PlatformSpec, ContentTypeSpec
)
from app.core.prompts import (
    REDDIT_AGENT_SYSTEM_PROMPT, CONTENT_CREATOR_SYSTEM_PROMPT,
    RESEARCH_PROMPT, TRENDING_RESEARCH_PROMPT, CONTENT_GENERATION_PROMPT,
    ARTICLE_GENERATION_PROMPT, X_THREAD_GENERATION_PROMPT, CONTENT_ANALYSIS_PROMPT
)

load_dotenv()

# Define default configs
DEFAULT_REDDIT_AGENT_CONFIG = RedditAgentConfig(
    model=ModelConfig(name="gemini-2.5-flash", temperature=0.4),
    system_prompt=REDDIT_AGENT_SYSTEM_PROMPT,
    search_limits=SearchLimits(subreddit_search_limit=8, content_search_limit=8),
    logging=LoggingConfig(enabled=True, log_file="reddit_agent_logs.json")
)

DEFAULT_CONTENT_CREATOR_CONFIG = ContentCreatorConfig(
    model=ModelConfig(name="gemini-2.5-flash", temperature=0.7),
    system_prompt=CONTENT_CREATOR_SYSTEM_PROMPT,
    tool_prompts={
        "research_prompt": RESEARCH_PROMPT,
        "trending_research_prompt": TRENDING_RESEARCH_PROMPT,
        "content_generation_prompt": CONTENT_GENERATION_PROMPT,
        "article_generation_prompt": ARTICLE_GENERATION_PROMPT,
        "x_thread_generation_prompt": X_THREAD_GENERATION_PROMPT,
        "content_analysis_prompt": CONTENT_ANALYSIS_PROMPT
    },
    platform_specs={
        "tiktok": PlatformSpec(
            hook_time="0-3s", pace="fast", optimal_duration="15-30s",
            style="Fast-paced and high-energy editing style. Content must align with current trends, use popular sounds, and deliver a punchy message immediately. The tone should be informal and exciting."
        ),
        "youtube": PlatformSpec(
            hook_time="0-5s", pace="moderate", optimal_duration="30-90s",
            style="Create a narrative that is informative and holds viewer attention. Use a moderate pace with clear, high-quality voiceover. The style should be detailed and well-researched, aiming to be a definitive resource on the topic."
        ),
        "article": PlatformSpec(
            optimal_length="500-800 words",
            style="Write a comprehensive but easy-to-read article. Use clear headings, subheadings, bullet points, and bold text to make the content highly scannable. The tone should be authoritative yet accessible to a general audience."
        ),
        "x": PlatformSpec(
            thread_length="3-5",
            style="Each post in the thread must deliver a high-value, standalone piece of information. The overall thread must tell a coherent and compelling story. The tone is professional, direct, and confident."
        )
    },
    output_paths={
        "articles": "articles",
        "x_threads": "x_threads"
    },
    content_types={
        "educational": ContentTypeSpec(
            description="Clearly explain a topic or concept to inform the audience. Assume they have little prior knowledge.",
            structure="Hook -> Core Concept -> Key Examples -> Summary/Conclusion"
        ),
        "how-to": ContentTypeSpec(
            description="Provide clear, step-by-step instructions to help the audience accomplish a specific task.",
            structure="Hook -> Required Tools/Setup -> Step 1, Step 2, ... -> Final Result -> Troubleshooting/Tips"
        ),
        "storytelling": ContentTypeSpec(
            description="Tell a compelling narrative with a clear beginning, middle, and end. Focus on emotional engagement.",
            structure="Hook -> Introduce Characters/Setting -> Rising Action/Conflict -> Climax -> Resolution/Moral"
        ),
        "news": ContentTypeSpec(
            description="Report on a current event or recent development in a factual, objective manner.",
            structure="Hook (Headline) -> Key Facts (5 Ws) -> Context/Background -> Implications/Future Outlook"
        ),
        "review": ContentTypeSpec(
            description="Provide a balanced and honest assessment of a product, service, or experience.",
            structure="Hook -> Overview/Specs -> Pros -> Cons -> Final Verdict/Recommendation"
        ),
        "comparison": ContentTypeSpec(
            description="Compare two or more items head-to-head on key criteria to help the audience make a decision.",
            structure="Hook -> Introduce Contenders -> Criterion 1 Comparison -> Criterion 2 Comparison -> Overall Recommendation"
        )
    },
    tone_settings={
        "conversational": "Write as if you're talking directly to a friend. Use simple language, ask questions, and adopt a warm, informal, and friendly approach.",
        "authoritative": "Project confidence and expertise. Use clear, direct statements and well-reasoned arguments. The language should be formal, credible, and objective.",
        "inspirational": "Aim to motivate and uplift the audience. Use positive language, powerful stories, and a hopeful, encouraging, and empowering perspective.",
        "humorous": "Use wit, jokes, and clever wordplay to entertain. The style should be lighthearted and funny, but still on-topic.",
        "intriguing": "Build curiosity and suspense. Use questions, teasers, and foreshadowing to make the audience eager to know more. The style is suspenseful and thought-provoking.",
        "suspenseful": "Create a sense of dread and anticipation. Use evocative, atmospheric language, short, tense sentences, and reveal information slowly to build suspense and unease.",
        "horror": "Evoke fear, dread, and unease. Use vivid, eerie descriptions, unsettling imagery, and a darker, slower pacing. The language should be chilling, immersive, and provoke a visceral reaction, leaving the audience disturbed or spooked."
    },
    audience_settings={
        "general": "Write for a broad general audience with clear, accessible explanations.",
        "beginners": "Write for people who are new to the topic and need step-by-step clarity.",
        "enthusiasts": "Write for interested readers who already know some basics and want a richer take.",
        "professionals": "Write for experienced readers or professionals, with precision and confidence."
    },
    output_formats={
        "article": "a long-form article",
        "newsletter": "a concise newsletter-style update",
        "thread": "an engaging X thread",
        "social": "a short social media post"
    },
    logging=LoggingConfig(
        enabled=True, log_file="content_creator_logs.json",
        log_errors=False, separate_error_log=True, error_log_file="content_creator_errors.json"
    )
)

DEFAULT_SHARED_CONFIG = SharedConfig(
    required_env_vars=["GEMINI_API_KEY", "REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET"],
    api_timeout=30,
    tool_timeout=45,
    reddit_requests_per_minute=60,
    google_requests_per_minute=100
)

# Active configuration path
CONFIG_DIR = Path("logs")
CONFIG_FILE = CONFIG_DIR / "active_config.json"

def load_app_config() -> AppConfig:
    """Load configuration from active config file or initialize from defaults."""
    os.makedirs(CONFIG_DIR, exist_ok=True)
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r") as f:
                data = json.load(f)
                return AppConfig.model_validate(data)
        except Exception as e:
            print(f"Error loading active config, resetting to defaults: {e}")
    
    # Initialize default config
    config = AppConfig(
        reddit_agent=DEFAULT_REDDIT_AGENT_CONFIG,
        content_creator=DEFAULT_CONTENT_CREATOR_CONFIG,
        shared=DEFAULT_SHARED_CONFIG
    )
    save_app_config(config)
    return config

def save_app_config(config: AppConfig) -> None:
    """Save active configuration to file."""
    os.makedirs(CONFIG_DIR, exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        f.write(config.model_dump_json(indent=2))

def get_reddit_agent_config() -> RedditAgentConfig:
    return load_app_config().reddit_agent

def get_content_creator_config() -> ContentCreatorConfig:
    return load_app_config().content_creator

def get_shared_config() -> SharedConfig:
    return load_app_config().shared

def get_tool_prompt(prompt_name: str, **kwargs) -> str:
    """Get and format a prompt from config."""
    config = get_content_creator_config()
    if prompt_name not in config.tool_prompts:
        raise ValueError(f"Prompt '{prompt_name}' not found in tool_prompts")
    prompt_template = config.tool_prompts[prompt_name]
    try:
        return prompt_template.format(**kwargs)
    except KeyError as e:
        raise ValueError(f"Missing required variable {e} for prompt '{prompt_name}'")

def update_agent_config(agent_type: str, updates: dict) -> AppConfig:
    """Update configuration for a specific agent type."""
    config = load_app_config()
    if agent_type == "reddit":
        # Merge dict
        current_data = config.reddit_agent.model_dump()
        _deep_update(current_data, updates)
        config.reddit_agent = RedditAgentConfig.model_validate(current_data)
    elif agent_type == "content_creator":
        current_data = config.content_creator.model_dump()
        _deep_update(current_data, updates)
        config.content_creator = ContentCreatorConfig.model_validate(current_data)
    elif agent_type == "shared":
        current_data = config.shared.model_dump()
        _deep_update(current_data, updates)
        config.shared = SharedConfig.model_validate(current_data)
    else:
        raise ValueError(f"Invalid agent type: {agent_type}")
    
    save_app_config(config)
    return config

def _deep_update(d: dict, u: dict) -> dict:
    """Recursively update a nested dictionary."""
    for k, v in u.items():
        if isinstance(v, dict) and k in d and isinstance(d[k], dict):
            _deep_update(d[k], v)
        else:
            d[k] = v
    return d

PRESETS = {
    "viral_focused": {
        "content_creator": {
            "model": {"temperature": 0.8},
            "system_prompt": CONTENT_CREATOR_SYSTEM_PROMPT + "\n\nFOCUS: Prioritize viral potential and shareability above all else. Use trending language, memes, and current references."
        }
    },
    "educational_focused": {
        "content_creator": {
            "model": {"temperature": 0.5},
            "system_prompt": CONTENT_CREATOR_SYSTEM_PROMPT + "\n\nFOCUS: Prioritize accuracy and educational value. Ensure content is informative and well-researched."
        }
    },
    "conservative": {
        "reddit_agent": {"model": {"temperature": 0.2}},
        "content_creator": {"model": {"temperature": 0.4}}
    },
    "creative": {
        "reddit_agent": {"model": {"temperature": 0.6}},
        "content_creator": {"model": {"temperature": 0.9}}
    }
}

def apply_preset(preset_name: str) -> AppConfig:
    """Apply a preset configuration."""
    if preset_name not in PRESETS:
        raise ValueError(f"Preset '{preset_name}' not found")
    
    preset = PRESETS[preset_name]
    config = load_app_config()
    
    if "reddit_agent" in preset:
        current = config.reddit_agent.model_dump()
        _deep_update(current, preset["reddit_agent"])
        config.reddit_agent = RedditAgentConfig.model_validate(current)
        
    if "content_creator" in preset:
        current = config.content_creator.model_dump()
        _deep_update(current, preset["content_creator"])
        config.content_creator = ContentCreatorConfig.model_validate(current)
        
    save_app_config(config)
    return config
