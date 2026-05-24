from pydantic import BaseModel, Field
from typing import List, Literal, Optional

class ResearchFact(BaseModel):
    fact: str = Field(description="A critical factual point or news update about the topic from authoritative sources")
    source: str = Field(description="Where this fact was retrieved from (e.g., website, news outlet, google grounding, etc.)")

class RedditSentiment(BaseModel):
    overall_sentiment: Literal["Positive", "Negative", "Mixed", "Divisive", "Neutral"] = Field(description="The overall prevailing sentiment on Reddit")
    dominant_themes: List[str] = Field(description="List of recurring topics, questions, or themes discussed by Reddit users")

class ContentAngle(BaseModel):
    angle: str = Field(description="A unique, engaging creative angle for content creation (e.g., a controversy, a human story, a surprising fact)")
    explanation: str = Field(description="A 1-sentence explanation of why this angle is highly compelling")

class ResearchBrief(BaseModel):
    topic: str = Field(description="The topic that was researched")
    factual_points: List[ResearchFact] = Field(description="3-5 essential facts gathered from google grounding")
    reddit_sentiment: RedditSentiment = Field(description="Reddit sentiment analysis and common discussion points")
    proposed_angles: List[ContentAngle] = Field(description="2-3 distinct creative angles based on research")

class TrendItem(BaseModel):
    trend: str = Field(description="Name or description of the trending topic")
    category: str = Field(description="Category of the trend (e.g., AI, gaming, finance)")
    sentiment: Literal["Positive", "Negative", "Mixed", "Controversial", "Growing", "Fading"]
    content_angle: str = Field(description="A compelling angle for a video or article based on this trend")

class TrendReport(BaseModel):
    category: str = Field(description="The category of trends researched")
    trends: List[TrendItem] = Field(description="List of current trending topics in the category")

class ScriptSegment(BaseModel):
    time_cue: str = Field(description="Timing cue in format MM:SS-MM:SS, e.g., '0:00-0:05'")
    narration: str = Field(description="The spoken narration text for the voiceover. Keep sentences short and punchy for captions.")
    vocal_emphasis: List[str] = Field(description="Words or phrases that should be spoken with emphasis")
    visual_cue: str = Field(description="Visual description of what to show on screen (B-roll, text overlay, facecam, etc.)")
    is_pause_after: bool = Field(default=False, description="Whether to add a brief pause after this segment")

class VideoScript(BaseModel):
    topic: str = Field(description="The main topic of the script")
    platform: Literal["youtube", "tiktok"] = Field(description="The platform target")
    hook: str = Field(description="A powerful, attention-grabbing hook for the first 3-5 seconds")
    segments: List[ScriptSegment] = Field(description="Structured chronological flow of the video script")
    call_to_action: str = Field(description="Final call to action (subscribe, follow, comment, check link)")

class ArticleSection(BaseModel):
    heading: str = Field(description="The heading or subheading for this section")
    content: str = Field(description="The detailed content of this section, formatted in Markdown")

class Article(BaseModel):
    title: str = Field(description="A compelling, SEO-optimized title for the article")
    introduction: str = Field(description="An engaging introduction paragraph that hooks the reader")
    sections: List[ArticleSection] = Field(description="Organized body sections of the article")
    conclusion: str = Field(description="A concise conclusion wrapping up key points")
    meta_description: str = Field(description="A compelling 1-2 sentence meta description for SEO (150-160 chars)")
    seo_keywords: List[str] = Field(description="List of 5-8 relevant SEO keywords")

class Tweet(BaseModel):
    index: int = Field(description="The post index (e.g., 1, 2, 3)")
    text: str = Field(description="The tweet content. Must be strictly under 280 characters. Do not include index numbering here.")

class XThread(BaseModel):
    topic: str = Field(description="The main topic of the thread")
    thread: List[Tweet] = Field(description="List of tweets in the thread. The first tweet is a strong hook. The thread must tell a coherent story.")
    hashtags: List[str] = Field(description="1-3 relevant hashtags")

class ContentAnalysis(BaseModel):
    verdict: Literal["POST", "TRASH"] = Field(description="Brutally honest publishing verdict")
    core_assessment: str = Field(description="Brutally honest explanation for the verdict (compelling elements if POST, weaknesses if TRASH)")
    actionable_fix: Optional[str] = Field(description="Single, high-impact suggestion to fix the content if TRASH, or 'Unsalvageable'")
