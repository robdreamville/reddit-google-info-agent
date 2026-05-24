import os
import sys
import subprocess
import time
import webbrowser
import threading

def check_requirements():
    """Verify requirements are installed, install if missing."""
    print("🔄 Verifying system dependencies...")
    try:
        # Check if fastapi and uvicorn are available
        import fastapi
        import uvicorn
        import praw
        import pydantic
    except ImportError:
        print("📥 Missing dependencies. Running pip install...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
            print("✨ Dependencies installed successfully!")
        except Exception as e:
            print(f"❌ Failed to install dependencies: {e}")
            print("Please run: pip install -r requirements.txt manually.")
            sys.exit(1)

def verify_env_file():
    """Ensure .env exists with required parameters."""
    env_path = ".env"
    if not os.path.exists(env_path):
        print("⚠️ .env file not found. Creating a template .env file...")
        template = (
            "GEMINI_API_KEY=your_gemini_api_key_here\n"
            "REDDIT_CLIENT_ID=your_reddit_client_id_here\n"
            "REDDIT_CLIENT_SECRET=your_reddit_client_secret_here\n"
            "GOOGLE_API_KEY=your_gemini_api_key_here\n"
        )
        with open(env_path, "w") as f:
            f.write(template)
        print("📁 Template .env file created at the root. Please fill in your API credentials.")

def open_browser():
    """Wait for server boot and open dashboard in default browser."""
    time.sleep(1.5)
    url = "http://127.0.0.1:8000"
    print(f"🚀 Launching dashboard in browser: {url}")
    webbrowser.open(url)

if __name__ == "__main__":
    check_requirements()
    verify_env_file()
    
    # Start browser thread
    threading.Thread(target=open_browser, daemon=True).start()
    
    # Start Uvicorn FastAPI Server
    print("🔌 Starting NeuraStream Web Server on http://127.0.0.1:8000 ...")
    try:
        import uvicorn
        uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)
    except KeyboardInterrupt:
        print("\n👋 NeuraStream server stopped.")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        input("Press Enter to exit...")

#TODO: Make this agent a full content app that can reply to post see vids and pics to reply to and do all of that.