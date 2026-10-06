"""Public evidence fixture: real HTTP/service/store, fake OpenAI transport only."""
import base64
import json
import os
import time
from pathlib import Path

import uvicorn

root = Path(__file__).resolve().parents[1] / '.agent_tmp' / 'onboarding-evidence' / 'fixture-state'
root.mkdir(parents=True, exist_ok=True)
os.environ['OH_PERSISTENCE_DIR'] = str(root / 'persist')
os.environ['CODEX_HOME'] = str(root / 'codex')
os.environ['OPENHANDS_SUPPRESS_BANNER'] = '1'
os.environ['OH_DISABLE_TELEMETRY'] = 'true'
os.environ['OH_SECRET_KEY'] = 'public-demo-fixture-only'
os.environ['OPENHANDS_AGENT_SERVER_CONFIG_PATH'] = str(root / 'config.json')
(root / 'config.json').write_text(json.dumps({
    'session_api_keys': ['fixture-session'], 'secret_key': 'public-demo-fixture-only',
    'enable_vscode': False, 'preload_tools': False,
    'workspace_path': str(root / 'workspace'),
    'conversations_path': str(root / 'conversations'),
}))
from openhands.agent_server.api import create_app
from openhands.agent_server.codex_auth import CodexAuthService
from openhands.agent_server.config import get_default_config
from openhands.agent_server.persistence import get_secrets_store
from openhands.sdk.llm.auth.openai import DeviceCode

def jwt(claims):
    payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip('=')
    return f'e30.{payload}.demo-signature'

class DemoProvider:
    attempt = 0
    polls = 0
    async def start(self):
        self.attempt += 1
        self.polls = 0
        return DeviceCode('http://127.0.0.1:18736/', f'DEMO-ONLY-{self.attempt}', 'public-fixture-handle', 1)
    async def poll(self, challenge):
        self.polls += 1
        if self.attempt % 2 == 1 or self.polls == 1:
            return None
        return await self.refresh('public-demo-placeholder')
    async def refresh(self, refresh_token):
        return {
            'id_token': jwt({'https://api.openai.com/auth': {'chatgpt_account_id': 'demo-fixture-account'}}),
            'access_token': jwt({'exp': int(time.time()) + 3600}),
            'refresh_token': 'demo-fixture-refresh-not-a-credential',
        }

config = get_default_config()
app = create_app(config)
app.state.codex_auth = CodexAuthService(get_secrets_store(config), root / 'codex' / 'auth.json', DemoProvider())
uvicorn.run(app, host='127.0.0.1', port=18735, log_level='warning')
