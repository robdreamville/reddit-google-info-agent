import os
import json
import threading
from datetime import datetime

LOG_DIR = os.path.join(os.path.dirname(__file__), 'logs')
LOG_FILE = os.path.join(LOG_DIR, 'sports_agent_runs.jsonl')
_lock = threading.Lock()

os.makedirs(LOG_DIR, exist_ok=True)

class SportsLogger:
    @staticmethod
    def log_run(run_data: dict):
        run_data['timestamp'] = datetime.utcnow().isoformat()
        with _lock:
            with open(LOG_FILE, 'a', encoding='utf-8') as f:
                f.write(json.dumps(run_data, ensure_ascii=False) + '\n')

    @staticmethod
    def log_tool_call(tool_name: str, input_data, output_data=None, error=None):
        return {
            'tool_name': tool_name,
            'input': input_data,
            'output': output_data,
            'error': error
        }
