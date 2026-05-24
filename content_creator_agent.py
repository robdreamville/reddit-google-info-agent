"""
Compatibility wrapper for Content Creator Agent.
Wraps the new pipeline implementation, maintaining CLI interactive creator options.
"""
from typing import List, Dict, Any, Optional
import sys
import json
from app.agents.content_creator import ContentCreatorAgent as ModernContentCreatorAgent

class ContentCreatorAgent:
    def __init__(self, config_preset: Optional[str] = None):
        if config_preset:
            from app.core.config import apply_preset
            apply_preset(config_preset)
        self._agent = ModernContentCreatorAgent()
        self.config = self._agent.config
        self.shared_config = self._agent.shared_config

    def create_content(
        self,
        topic: str,
        platforms: List[str],
        content_type: str = "educational",
        duration: str = None,
        tone: str = "engaging"
    ) -> Dict[str, Any]:
        """Call the new validated pipeline."""
        return self._agent.create_content(
            topic=topic,
            platforms=platforms,
            content_type=content_type,
            duration=duration,
            tone=tone
        )

    def research_topic(self, topic: str, platform_focus: str = "all") -> str:
        from app.tools.content_tools import research_topic_for_content
        return research_topic_for_content.invoke({"topic": topic, "platform_focus": platform_focus})

    def analyze_content(self, content: str, platform: str) -> str:
        from app.tools.content_tools import analyze_content_performance
        return analyze_content_performance.invoke({"content_text": content, "platform": platform})

    def research_trending(self, category: str = "general") -> str:
        from app.tools.content_tools import research_trending_topics
        return research_trending_topics.invoke({"category": category})

    def get_analytics(self) -> Dict[str, Any]:
        from app.core.logger import AppLogger
        return AppLogger().get_analytics()

    def interactive_content_creator(self):
        print("🎬 Content Creator Agent Ready! (Compatibility CLI)")
        print("Generate content for YouTube, TikTok, Articles, and X.")
        print("Type 'quit' to exit.")
        print("-" * 60)
        
        while True:
            try:
                print("\n📝 Content Creation Options:")
                print("1. Quick Create - Enter topic for all platforms")
                print("2. Custom Create - Customize duration, tone, platforms")
                print("3. Trending Research - Research trending topics")
                print("4. View Analytics - See usage statistics")
                
                choice = input("\nChoose option (1-4) or 'quit': ").strip()
                
                if choice.lower() in ['quit', 'exit', 'q']:
                    print("Happy creating! 🎬")
                    break
                
                elif choice == "1":
                    topic = input("Enter topic: ").strip()
                    if topic:
                        print("\n🔄 Generating content for all platforms...")
                        result = self.create_content(topic, platforms=["youtube", "tiktok", "article", "x"])
                        self._display_content(result)
                        
                elif choice == "2":
                    topic = input("Topic: ").strip()
                    plat_input = input("Platforms (youtube,tiktok,article,x - comma separated): ").strip() or "youtube,tiktok"
                    platforms = [p.strip() for p in plat_input.split(',')]
                    duration = input("Video Duration (e.g. 15-30s): ").strip() or None
                    content_type = input("Type (educational/storytelling/etc): ").strip() or "educational"
                    tone = input("Tone (conversational/etc): ").strip() or "engaging"
                    
                    if topic:
                        print(f"\n🔄 Generating content...")
                        result = self.create_content(topic, platforms, content_type, duration, tone)
                        self._display_content(result)
                        
                elif choice == "3":
                    cat = input("Category (gaming/AI/etc): ").strip() or "general"
                    print(f"\n🔍 Researching trends for {cat}...")
                    trends = self.research_trending(cat)
                    print(f"\n📈 Current Trends:\n{trends}")
                    
                elif choice == "4":
                    print("\n📊 Usage Analytics:")
                    print(json.dumps(self.get_analytics(), indent=2))
                    
                else:
                    print("Invalid choice.")
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")

    def _display_content(self, result: Dict[str, Any]):
        print("\n" + "="*60)
        print(f"🎯 CONTENT GENERATED FOR: {result.get('topic', 'Unknown Topic').upper()}")
        print("="*60)
        for platform, content in result.get("content", {}).items():
            print(f"\n📱 {platform.upper()} SCRIPT/POST DATA:")
            print("-" * 30)
            print(json.dumps(content, indent=2))
            print("-" * 30)
        for platform, file_path in result.get("files", {}).items():
            print(f"💾 Saved {platform.upper()} content to: {file_path}")
        print("="*60)

if __name__ == "__main__":
    try:
        creator = ContentCreatorAgent()
        creator.interactive_content_creator()
    except Exception as e:
        print(f"❌ Error initializing Content Creator Agent: {str(e)}")
