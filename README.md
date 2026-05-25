# NeuraStream - AI Agent Content Engine

A state-of-the-art content generation engine utilizing Gemini, Reddit APIs, and LangGraph. Features a FastAPI web interface for real-time log monitoring, analytics, and content generation.

## Core Features
- **Multi-Source Research**: Leverages Reddit and Google Search for comprehensive topic analysis.
- **Structured Content Generation**: Creates platform-native content for YouTube, TikTok, Articles, and X (Twitter).
- **Configurable Generation**: Fine-tune content by selecting a target audience and output format (e.g., Markdown, JSON).
- **Quality & Performance Analysis**: Includes a "Honest Critic" LLM call for performance prediction and local checks for readability, factuality, and explicit content.
- **Web UI Dashboard**: A FastAPI-powered frontend to control the agent, view analytics, and browse a detailed history of all runs.
- **Filterable History**: The history UI supports filtering by topic, date range, platform, and success status, with options to copy and export data.
- **Prompt Management**: All prompts are centrally managed in `app/core/prompts.py` and can be customized via preset files in `config/`.

## Setup
1. Clone the repository:
   ```sh
   git clone <your-repo-url>
   cd reddit_topic
   ```
2. Create and activate a Python virtual environment:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
4. Add your API keys and secrets to a new `.env` file (see `.env.example` for a template).

## .env File
Create a `.env` file in the root directory and add your keys:
```
GEMINI_API_KEY="your_gemini_api_key"
REDDIT_CLIENT_ID="your_reddit_client_id"
REDDIT_CLIENT_SECRET="your_reddit_client_secret"
```

## Usage
Run the FastAPI server:
```sh
uvicorn app.main:app --reload
```
Navigate to `http://127.0.0.1:8000` in your browser to access the dashboard.

## Logging
- All agent runs are logged in the `logs/` directory.
- Logs can be viewed, filtered, and exported from the "Log History" tab in the web UI.

## Extending
- **Add New Tools**: Create new tool functions in `app/tools/` and integrate them into the agent workflow in `app/agents/content_creator.py`.
- **Customize Prompts**: Modify the prompt templates in `app/core/prompts.py`.
- **Change Config**: Adjust default behaviors, model names, and presets in `config/app_config.py`.

