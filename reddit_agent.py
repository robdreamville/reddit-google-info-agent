"""
Compatibility wrapper for Reddit Agent.
Wraps the refactored RedditAgent class, preserving local interactive CLI execution features.
"""
import sys
from app.agents.reddit_agent import RedditAgent as ModernRedditAgent

class RedditAgent:
    def __init__(self):
        self._agent = ModernRedditAgent()
        self.config = self._agent.config
        self.shared_config = self._agent.shared_config

    def chat(self, message: str) -> str:
        return self._agent.chat(message)

    def interactive_chat(self):
        print("🤖 Reddit Agent Ready! (Compatibility CLI)")
        print(f"🤖 Model: {self.config.model.name} (temp: {self.config.model.temperature})")
        print("Type 'quit' to exit.")
        print("-" * 60)
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                if not user_input:
                    continue
                response = self.chat(user_input)
                print(f"Agent: {response}")
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    try:
        agent = RedditAgent()
        agent.interactive_chat()
    except Exception as e:
        print(f"❌ Error initializing Reddit Agent: {str(e)}")
