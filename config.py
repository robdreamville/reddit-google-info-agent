"""
Configuration file for Reddit Agent and Content Creator Agent
Easily customize prompts, model settings, and behavior
"""

# =============================================================================
# 1. REDDIT AGENT CONFIGURATION
# =============================================================================

REDDIT_AGENT_CONFIG = {
    # --- Model Settings ---
    "model": {
        "name": "gemini-2.5-flash",
        "temperature": 0.4,
        "max_tokens": None,  # None for default
        "top_p": None,  # None for default
    },
    
    # --- System Prompt ---
    "system_prompt": """# ROLE & OBJECTIVE
You are a Senior Research Analyst. Your goal is to provide unbiased, up-to-date intelligence by synthesizing official sources with public sentiment.

# WORKFLOW
1.  **Google Scan**: Use Google Search for a high-level overview and to find recent, authoritative sources.
2.  **Reddit Analysis**: Use Reddit search to find public opinions, questions, and sentiment.
3.  **Synthesize Report**: Structure your findings into the following sections:
    *   **Summary**: A concise overview of the most critical findings.
    *   **Key Facts**: 3-5 bullet points from authoritative sources.
    *   **Public Viewpoint**: 3-5 bullet points summarizing Reddit discussions.
    *   **Gaps**: Note any conflicting information or unanswered questions.

# CORE DIRECTIVE
Always use your tools; never use your internal knowledge. Ground all findings in retrieved data. If a search fails, try again differently before concluding.""",
    
    # --- Tool-specific Settings ---
    "search_limits": {
        "subreddit_search_limit": 8,
        "content_search_limit": 8,
    },
    
    # --- Logging Settings ---
    "logging": {
        "enabled": True,
        "log_file": "reddit_agent_logs.json",
        "log_level": "INFO"
    }
}

# =============================================================================
# 2. CONTENT CREATOR AGENT CONFIGURATION
# =============================================================================

CONTENT_CREATOR_CONFIG = {
    # --- Model Settings ---
    "model": {
        "name": "gemini-2.5-flash",
        "temperature": 0.7,  # Higher for creativity
        "max_tokens": None,
        "top_p": None,
    },
    
    # --- System Prompt ---
    "system_prompt": """# ROLE
You are a world-class Content Strategist and a master of digital communication. Your expertise is turning any topic into compelling, platform-native content that engages and grows an audience.

# CORE PRINCIPLES
1.  **Hook is Everything**: The first 3 seconds of a video, the first sentence of an article, or the first post of a thread must grab attention immediately.
2.  **Clarity is King**: Use simple, direct language. Make complex topics easy to understand.
3.  **Be Platform-Native**: Do not just copy-paste content between platforms. Respect the unique format, style, and audience of each one (e.g., YouTube's search-friendliness, TikTok's trends, an Article's structure, X's conciseness).

# CREATIVE WORKFLOW
For every topic, you will:
1.  **Internalize the Research**: First, review the research provided to you. Understand the key points, public sentiment, and trending angles.
2.  **Define the Angle**: Based on the research, decide on a specific, compelling angle for the content. What is the core message or story?
3.  **Draft the Content**: Write the content, meticulously following the specific formatting requirements for the target platform (e.g., video script structure, article markdown, X thread numbering).
4.  **Refine and Polish**: Review your draft for clarity, engagement, and adherence to the core principles. Ensure the tone is perfect.""",

    # =============================================================================
    # 2.1. Tool Prompts
    # Prompts used by the content creator's tools.
    # =============================================================================
    "tool_prompts": {
        # --- Research Tool ---
        "research_prompt": """# TASK
Your goal is to conduct comprehensive research on \"{topic}\" and compile a strategic brief for creating content on {platform_focus}. The current date is {current_date}.

# RESEARCH CHECKLIST
1.  **Key Facts**: Identify the most important facts, stats, and recent news from authoritative sources.
2.  **Public Sentiment**: Analyze Reddit to determine the overall public sentiment, common questions, and key discussion themes.
3.  **Content Angles**: Identify 2-3 specific, engaging angles (e.g., controversies, human-interest stories, surprising facts).

# REQUIRED OUTPUT FORMAT - Make sure to fill in all these fields with the data found using tools.
Present your findings as a structured Markdown report. Do not include conversational filler. Do not include your thought process.

### Key Factual Points
- Bulleted list of 3-5 essential facts from your research.

### Reddit Sentiment Analysis
- **Overall Sentiment**: A single descriptor (e.g., Positive, Negative, Mixed, Divisive).
- **Dominant Themes**: A bulleted list of 4-6 recurring topics of discussion.

### Proposed Content Angles
- A bulleted list of 2-3 distinct angles. For each, provide a 1-sentence explanation of why it's compelling.""",

        "trending_research_prompt": """# TASK
Your goal is to ONLY research trending topics specifically related to \"{category}\". Ignore all unrelated trends. The current date is {current_date}.

# RESEARCH CHECKLIST
1.  **Identify Trends**: Use your tools to find 3-5 current, rising, or popular trends.
2.  **Analyze Sentiment**: Briefly describe the public sentiment around each trend.
3.  **Suggest Angles**: For each trend, propose a compelling content angle.

# REQUIRED OUTPUT FORMAT
Present your findings as a structured Markdown report. Do Not include your thought process. Do not include conversational filler.

### Trending Topics
- **Trend 1**: [Name of Trend]
  - **Sentiment**: [e.g., Positive, Controversial, Growing]
  - **Content Angle**: [A compelling angle for a video or article]
- **Trend 2**: [Name of Trend]
  - **Sentiment**: [e.g., Positive, Controversial, Growing]
  - **Content Angle**: [A compelling angle for a video or article]
- **Trend 3**: [Name of Trend]
  - **Sentiment**: [e.g., Positive, Controversial, Growing]
  - **Content Angle**: [A compelling angle for a video or article]""",

        # --- Video Content Tool (YouTube/TikTok) ---
        "content_generation_prompt":"""# SCRIPT BRIEF
- **Topic**: {topic}
- **Platform**: {platform}
- **Tone**: {tone_description}
- **Duration**: {duration}
- **Pacing**: {pace}

# CONTENT BRIEF
- **Description**: {content_description}
- **Structure**: {content_structure}

# STYLE GUIDELINES
{style}

# RESEARCH SUMMARY
{research_summary}

# SCRIPT REQUIREMENTS
- **Use the Research**: Your script MUST be based on the provided RESEARCH SUMMARY.
- **Hook**: Must be within the first {hook_time}. It must be powerful and attention-grabbing.
- **Structure**: Follow a clear HOOK -> MAIN CONTENT -> CONCLUSION structure.
- **Formatting**:
    - Use `[EMPHASIS: text]` for vocal emphasis.
    - Use `[PAUSE]` for brief pauses in speech.
    - Use `[0:00-0:05]` style timing cues for major sections.
- **Delivery**: Write for a natural, human voiceover. Sentences must be short and easy to read for captions.""",

        # --- Article Generation Tool ---
        "article_generation_prompt": """# TASK
Write a complete, polished article based on the following details:

- **Topic**: {topic}
- **Tone**: {tone_description}
- **Length**: {optimal_length}
- **Style Guidelines**: {style}
- **Research Summary**: {research_summary}

# REQUIREMENTS
- The article MUST be based on the provided Research Summary.
- The article MUST be written as if it is ready for publication, not as a brief or outline.
- Include:
  - A compelling title
  - An engaging introduction
  - A well-organized body with clear headings and subheadings
  - A concise conclusion
- Ensure the article is informative, valuable, and SEO-friendly.
- Do not include meta sections like "ARTICLE BRIEF" or "STYLE GUIDELINES" in the output.
""",

        # --- X (Twitter) Thread Generation Tool ---
        "x_thread_generation_prompt": """# X THREAD BRIEF
- **Topic**: {topic}
- **Tone**: {tone_description}
- **Length**: {thread_length} posts

# STYLE GUIDELINES
{style}

# RESEARCH SUMMARY
{research_summary}

# THREAD REQUIREMENTS
- **Use the Research**: Your thread MUST be based on the provided RESEARCH SUMMARY.
- **Format**:
    - Each post must be under 280 characters.
    - Number each post in the format (1/N).
    - The first post must be a strong hook.
- **Content**: The thread must tell a coherent story or provide clear, concise information. Avoid \"cheesy\" marketing language.
- **Hashtags**: Use 1-3 relevant hashtags at the end of the final post only.
- **Output**: Return the entire thread as a single string, with each post separated by \"---\".""",

        # --- Content Analysis Tool ---
        "content_analysis_prompt": """# ROLE & GOAL
You are the final quality check, a brutally honest content critic. Your sole purpose is to determine if a piece of content is worth publishing or if it's a waste of time. Do not be sycophantic. Be direct, critical, and provide clear, actionable feedback.

# ANALYSIS WORKFLOW
1.  **The Verdict**: Start with a clear, one-word verdict: **POST** or **TRASH**.
2.  **Core Assessment**: In one paragraph, give a brutally honest explanation for your verdict.
    *   If **POST**, explain what makes it compelling and why it will perform well on {platform}.
    *   If **TRASH**, identify the core weaknesses (e.g., weak hook, boring narrative, unclear value, bad tone). Be specific.
3.  **Actionable Fix**: If you voted **TRASH**, provide a single, high-impact suggestion to fix it. If it's unsalvageable, state that.

# CONTENT FOR REVIEW
- **Current Date**: {current_date}
- **Platform**: {platform}
- **Content**: {content_text}"""
    },

    # =============================================================================
    # 2.2. Platform Specifications
    # Defines the unique characteristics and requirements for each platform.
    # =============================================================================
    "platform_specs": {
        "tiktok": {
            "hook_time": "0-3s",
            "pace": "fast",
            "style": "Fast-paced and high-energy editing style. Content must align with current trends, use popular sounds, and deliver a punchy message immediately. The tone should be informal and exciting.",
            "optimal_duration": "15-30s"
        },
        "youtube": {
            "hook_time": "0-5s",
            "pace": "moderate",
            "style": "Create a narrative that is informative and holds viewer attention. Use a moderate pace with clear, high-quality voiceover. The style should be detailed and well-researched, aiming to be a definitive resource on the topic.",
            "optimal_duration": "30-90s"
        },
        "article": {
            "style": "Write a comprehensive but easy-to-read article. Use clear headings, subheadings, bullet points, and bold text to make the content highly scannable. The tone should be authoritative yet accessible to a general audience.",
            "optimal_length": "500-800 words"
        },
        "x": {
            "style": "Each post in the thread must deliver a high-value, standalone piece of information. The overall thread must tell a coherent and compelling story. The tone is professional, direct, and confident.",
            "thread_length": "3-5"
        }
    },
    
    # --- File Output Paths ---
    "output_paths": {
        "articles": "articles",
        "x_threads": "x_threads"
    },

    # --- Content Type Definitions ---
    "content_types": {
        "educational": {
            "description": "Clearly explain a topic or concept to inform the audience. Assume they have little prior knowledge.",
            "structure": "Hook -> Core Concept -> Key Examples -> Summary/Conclusion"
        },
        "how-to": {
            "description": "Provide clear, step-by-step instructions to help the audience accomplish a specific task.",
            "structure": "Hook -> Required Tools/Setup -> Step 1, Step 2, ... -> Final Result -> Troubleshooting/Tips"
        },
        "storytelling": {
            "description": "Tell a compelling narrative with a clear beginning, middle, and end. Focus on emotional engagement.",
            "structure": "Hook -> Introduce Characters/Setting -> Rising Action/Conflict -> Climax -> Resolution/Moral"
        },
        "news": {
            "description": "Report on a current event or recent development in a factual, objective manner.",
            "structure": "Hook (Headline) -> Key Facts (5 Ws) -> Context/Background -> Implications/Future Outlook"
        },
        "review": {
            "description": "Provide a balanced and honest assessment of a product, service, or experience.",
            "structure": "Hook -> Overview/Specs -> Pros -> Cons -> Final Verdict/Recommendation"
        },
        "comparison": {
            "description": "Compare two or more items head-to-head on key criteria to help the audience make a decision.",
            "structure": "Hook -> Introduce Contenders -> Criterion 1 Comparison -> Criterion 2 Comparison -> Overall Recommendation"
        }
    },
    
    # --- Tone Definitions ---
    "tone_settings": {
        "conversational": "Write as if you're talking directly to a friend. Use simple language, ask questions, and adopt a warm, informal, and friendly approach.",
        "authoritative": "Project confidence and expertise. Use clear, direct statements and well-reasoned arguments. The language should be formal, credible, and objective.",
        "inspirational": "Aim to motivate and uplift the audience. Use positive language, powerful stories, and a hopeful, encouraging, and empowering perspective.",
        "humorous": "Use wit, jokes, and clever wordplay to entertain. The style should be lighthearted and funny, but still on-topic.",
        "intriguing": "Build curiosity and suspense. Use questions, teasers, and foreshadowing to make the audience eager to know more. The style is suspenseful and thought-provoking.",
        "suspenseful": "Create a sense of dread and anticipation. Use evocative, atmospheric language, short, tense sentences, and reveal information slowly to build suspense and unease.",
        "horror": "Evoke fear, dread, and unease. Use vivid, eerie descriptions, unsettling imagery, and a darker, slower pacing. The language should be chilling, immersive, and provoke a visceral reaction, leaving the audience disturbed or spooked."
    },
    
    # --- Logging Settings ---
    "logging": {
        "enabled": True,
        "log_file": "content_creator_logs.json", 
        "log_level": "INFO",
        "log_errors": False,  # Set to False to not log errors in main logging
        "separate_error_log": True,  # Log errors separately
        "error_log_file": "content_creator_errors.json"
    }
}


# =============================================================================
# 3. SHARED CONFIGURATION
# Settings used by multiple agents.
# =============================================================================

SHARED_CONFIG = {
    "environment": {
        "required_env_vars": [
            "GEMINI_API_KEY",
            "REDDIT_CLIENT_ID", 
            "REDDIT_CLIENT_SECRET"
        ]
    },
    
    "default_timeouts": {
        "api_timeout": 30,
        "tool_timeout": 45
    },
    
    "rate_limits": {
        "reddit_requests_per_minute": 60,
        "google_requests_per_minute": 100
    }
}

# =============================================================================
# 4. HELPER FUNCTIONS
# Functions to access and manage the configuration dictionaries.
# =============================================================================

def get_reddit_agent_config():
    """Get Reddit Agent configuration"""
    return REDDIT_AGENT_CONFIG

def get_content_creator_config():
    """Get Content Creator Agent configuration"""
    return CONTENT_CREATOR_CONFIG

def get_shared_config():
    """Get shared configuration"""
    return SHARED_CONFIG

def get_tool_prompt(prompt_name: str, **kwargs) -> str:
    """
    Get a formatted tool prompt from config
    
    Args:
        prompt_name: Name of the prompt in tool_prompts
        **kwargs: Variables to format into the prompt
        
    Returns:
        Formatted prompt string
    """
    config = get_content_creator_config()
    
    if prompt_name not in config["tool_prompts"]:
        raise ValueError(f"Prompt '{prompt_name}' not found in tool_prompts")
    
    prompt_template = config["tool_prompts"][prompt_name]
    
    try:
        formatted_prompt = prompt_template.format(**kwargs)

        # --- ADD THESE LINES FOR DEBUGGING ---
        print(f"\\n--- DEBUG: Final Prompt for {prompt_name} ---\\n")
        print(formatted_prompt)
        print(f"\\n--- END DEBUG: {prompt_name} ---\\n")
                # -------------------------------------
    
        return formatted_prompt
    except KeyError as e:
        raise ValueError(f"Missing required variable {e} for prompt '{prompt_name}'")



def update_config(agent_type: str, section: str, key: str, value):
    """
    Update configuration dynamically
    
    Args:
        agent_type: "reddit" or "content_creator"
        section: Config section (e.g., "model", "system_prompt")
        key: Specific key to update
        value: New value
    """
    if agent_type == "reddit":
        config = REDDIT_AGENT_CONFIG
    elif agent_type == "content_creator":
        config = CONTENT_CREATOR_CONFIG
    else:
        raise ValueError("Invalid agent_type. Use 'reddit' or 'content_creator'")
    
    if section in config:
        if isinstance(config[section], dict):
            config[section][key] = value
        else:
            config[section] = value
    else:
        config[section] = {key: value}

# =============================================================================
# 5. PRESET CONFIGURATIONS
# Quick presets for different content styles and agent behaviors.
# =============================================================================

PRESETS = {
    "viral_focused": {
        "content_creator": {
            "model": {"temperature": 0.8},
            "system_prompt": CONTENT_CREATOR_CONFIG["system_prompt"] + "\n\nFOCUS: Prioritize viral potential and shareability above all else. Use trending language, memes, and current references.",
            "tool_prompts": {
                "content_generation_prompt": CONTENT_CREATOR_CONFIG["tool_prompts"]["content_generation_prompt"] + "\n\nEXTRA FOCUS: Make this as viral and shareable as possible. Use trending formats and language."
            }
        }
    },
    
    "educational_focused": {
        "content_creator": {
            "model": {"temperature": 0.5},
            "system_prompt": CONTENT_CREATOR_CONFIG["system_prompt"] + "\n\nFOCUS: Prioritize accuracy and educational value. Ensure content is informative and well-researched.",
            "tool_prompts": {
                "content_generation_prompt": CONTENT_CREATOR_CONFIG["tool_prompts"]["content_generation_prompt"] + "\n\nEXTRA FOCUS: Ensure accuracy and educational value. Make complex topics easy to understand."
            }
        }
    },
    
    "conservative": {
        "reddit_agent": {
            "model": {"temperature": 0.2}
        },
        "content_creator": {
            "model": {"temperature": 0.4}
        }
    },
    
    "creative": {
        "reddit_agent": {
            "model": {"temperature": 0.6}
        },
        "content_creator": {
            "model": {"temperature": 0.9},
            "tool_prompts": {
                "content_generation_prompt": CONTENT_CREATOR_CONFIG["tool_prompts"]["content_generation_prompt"] + "\n\nEXTRA FOCUS: Be creative and unique. Use unexpected angles and creative approaches."
            }
        }
    }
}

def apply_preset(preset_name: str):
    """Apply a preset configuration"""
    if preset_name not in PRESETS:
        raise ValueError(f"Preset '{preset_name}' not found")
    
    preset = PRESETS[preset_name]
    
    # Apply reddit agent changes
    if "reddit_agent" in preset:
        for section, updates in preset["reddit_agent"].items():
            if isinstance(updates, dict):
                REDDIT_AGENT_CONFIG[section].update(updates)
            else:
                REDDIT_AGENT_CONFIG[section] = updates
    
    # Apply content creator changes  
    if "content_creator" in preset:
        for section, updates in preset["content_creator"].items():
            if isinstance(updates, dict):
                CONTENT_CREATOR_CONFIG[section].update(updates)
            else:
                CONTENT_CREATOR_CONFIG[section] = updates
