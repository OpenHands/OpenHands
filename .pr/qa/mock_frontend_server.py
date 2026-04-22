#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

DEFAULT_MODEL = "openhands/claude-opus-4-5-20251101"


def create_config(app_mode: str) -> dict:
    return {
        "app_mode": app_mode,
        "posthog_client_key": "test-posthog-key",
        "feature_flags": {
            "enable_billing": False,
            "hide_llm_settings": False,
            "enable_jira": False,
            "enable_jira_dc": False,
            "enable_linear": False,
            "hide_users_page": False,
            "hide_billing_page": False,
            "hide_integrations_page": False,
        },
        "providers_configured": [],
        "maintenance_start_time": None,
        "auth_url": None,
        "recaptcha_site_key": None,
        "faulty_models": [],
        "error_message": None,
        "updated_at": "2026-04-22T00:00:00Z",
        "github_app_slug": None,
    }


def create_settings() -> dict:
    agent_settings_schema = {
        "model_name": "AgentSettings",
        "sections": [
            {
                "key": "llm",
                "label": "LLM",
                "fields": [
                    {
                        "key": "llm.model",
                        "label": "Model",
                        "description": "Select the model to use for this conversation.",
                        "section": "llm",
                        "section_label": "LLM",
                        "value_type": "string",
                        "default": DEFAULT_MODEL,
                        "choices": [],
                        "depends_on": [],
                        "prominence": "critical",
                        "secret": False,
                        "required": True,
                    },
                    {
                        "key": "llm.api_key",
                        "label": "API Key",
                        "description": "Provide the API key used to authenticate requests for the selected model.",
                        "section": "llm",
                        "section_label": "LLM",
                        "value_type": "string",
                        "default": None,
                        "choices": [],
                        "depends_on": [],
                        "prominence": "critical",
                        "secret": True,
                        "required": False,
                    },
                    {
                        "key": "llm.base_url",
                        "label": "Base URL",
                        "description": "Override the model provider's default API base URL when needed.",
                        "section": "llm",
                        "section_label": "LLM",
                        "value_type": "string",
                        "default": None,
                        "choices": [],
                        "depends_on": [],
                        "prominence": "critical",
                        "secret": False,
                        "required": False,
                    },
                    {
                        "key": "llm.temperature",
                        "label": "Temperature",
                        "description": "Adjust randomness for non-deterministic model outputs.",
                        "section": "llm",
                        "section_label": "LLM",
                        "value_type": "number",
                        "default": None,
                        "choices": [],
                        "depends_on": [],
                        "prominence": "minor",
                        "secret": False,
                        "required": False,
                    },
                ],
            }
        ],
    }

    conversation_settings_schema = {
        "model_name": "ConversationSettings",
        "sections": [
            {
                "key": "conversation",
                "label": "Conversation",
                "fields": [
                    {
                        "key": "confirmation_mode",
                        "label": "Confirmation mode",
                        "description": "Require approval before applying changes.",
                        "section": "conversation",
                        "section_label": "Conversation",
                        "value_type": "boolean",
                        "default": False,
                        "choices": [],
                        "depends_on": [],
                        "prominence": "critical",
                        "secret": False,
                        "required": False,
                    }
                ],
            }
        ],
    }

    return {
        "llm_model": DEFAULT_MODEL,
        "llm_base_url": "",
        "agent": "CodeActAgent",
        "language": "en",
        "llm_api_key": None,
        "llm_api_key_set": False,
        "search_api_key_set": False,
        "confirmation_mode": False,
        "security_analyzer": "llm",
        "max_iterations": None,
        "remote_runtime_resource_factor": 1,
        "provider_tokens_set": {},
        "enable_default_condenser": True,
        "condenser_max_size": 240,
        "enable_sound_notifications": False,
        "user_consents_to_analytics": False,
        "enable_proactive_conversation_starters": False,
        "enable_solvability_analysis": False,
        "search_api_key": "",
        "is_new_user": False,
        "disabled_skills": [],
        "mcp_config": {"sse_servers": [], "stdio_servers": [], "shttp_servers": []},
        "max_budget_per_task": None,
        "email": "",
        "email_verified": True,
        "git_user_name": "openhands",
        "git_user_email": "openhands@all-hands.dev",
        "v1_enabled": True,
        "sandbox_grouping_strategy": "NO_GROUPING",
        "agent_settings_schema": agent_settings_schema,
        "agent_settings": {
            "schema_version": 1,
            "agent": "CodeActAgent",
            "llm": {
                "model": DEFAULT_MODEL,
                "api_key": None,
                "base_url": None,
                "temperature": None,
            },
            "condenser": {"enabled": True, "max_size": 240},
            "mcp_config": {"sse_servers": [], "stdio_servers": [], "shttp_servers": []},
            "critic": {"enabled": False, "mode": "finish_and_message"},
        },
        "conversation_settings_schema": conversation_settings_schema,
        "conversation_settings": {
            "schema_version": 1,
            "confirmation_mode": False,
            "security_analyzer": "llm",
        },
    }


def create_organizations() -> dict:
    org = {
        "id": "1",
        "name": "Personal Workspace",
        "contact_name": "Contact Name",
        "contact_email": "contact@example.com",
        "conversation_expiration": 86400,
        "remote_runtime_resource_factor": 2,
        "billing_margin": 0.15,
        "enable_proactive_conversation_starters": True,
        "sandbox_base_container_image": "ghcr.io/example/sandbox-base:latest",
        "sandbox_runtime_container_image": "ghcr.io/example/sandbox-runtime:latest",
        "org_version": 0,
        "agent_settings": {
            "agent": "CodeActAgent",
            "llm": {"model": DEFAULT_MODEL, "base_url": None},
            "condenser": {"enabled": True, "max_size": 240},
            "mcp_config": {"sse_servers": [], "stdio_servers": [], "shttp_servers": []},
        },
        "search_api_key": None,
        "sandbox_api_key": None,
        "max_budget_per_task": 25.0,
        "enable_solvability_analysis": False,
        "v1_enabled": True,
        "credits": 100,
        "is_personal": True,
    }
    return {"items": [org], "currentOrgId": "1"}


def create_me() -> dict:
    return {
        "org_id": "1",
        "user_id": "99",
        "email": "me@example.com",
        "role": "owner",
        "status": "active",
        "llm_api_key": "",
        "max_iterations": 20,
        "llm_model": DEFAULT_MODEL,
        "llm_base_url": "",
    }


def create_models() -> dict:
    return {
        "models": [
            "anthropic/claude-opus-4-5-20251101",
            "openai/gpt-4o",
            DEFAULT_MODEL,
        ],
        "verified_models": [
            "claude-opus-4-5-20251101",
            "gpt-4o",
        ],
        "verified_providers": ["openhands", "anthropic", "openai"],
        "default_model": DEFAULT_MODEL,
        "java_api_available": False,
    }


def create_provider_search_results() -> dict:
    return {
        "items": [
            {"name": "openhands", "verified": True},
            {"name": "anthropic", "verified": True},
            {"name": "openai", "verified": True},
        ],
        "next_page_id": None,
    }


def create_model_search_results() -> dict:
    return {
        "items": [
            {"provider": "openhands", "name": "claude-opus-4-5-20251101", "verified": True},
            {"provider": "anthropic", "name": "claude-opus-4-5-20251101", "verified": True},
            {"provider": "openai", "name": "gpt-4o", "verified": True},
        ],
        "next_page_id": None,
    }


class MockFrontendHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, directory: str, app_mode: str, **kwargs):
        self.app_mode = app_mode
        self.static_dir = directory
        super().__init__(*args, directory=directory, **kwargs)

    def log_message(self, format: str, *args):
        print(f"[{self.log_date_time_string()}] {format % args}")

    def _write_json(self, payload: dict, status: int = HTTPStatus.OK):
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def do_POST(self):
        if self.path == "/api/authenticate":
            self._write_json({"authenticated": True})
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not Found")

    def do_GET(self):
        if self.path == "/api/v1/web-client/config":
            self._write_json(create_config(self.app_mode))
            return
        if self.path == "/api/v1/settings":
            self._write_json(create_settings())
            return
        if self.path == "/api/v1/settings/agent-schema":
            self._write_json(create_settings()["agent_settings_schema"])
            return
        if self.path == "/api/v1/settings/conversation-schema":
            self._write_json(create_settings()["conversation_settings_schema"])
            return
        if self.path == "/api/options/models":
            self._write_json(create_models())
            return
        if self.path.startswith("/api/v1/config/providers/search"):
            self._write_json(create_provider_search_results())
            return
        if self.path.startswith("/api/v1/config/models/search"):
            self._write_json(create_model_search_results())
            return
        if self.path == "/api/options/security-analyzers":
            self._write_json(["llm", "none"])
            return
        if self.path == "/api/organizations":
            self._write_json(create_organizations())
            return
        if self.path == "/api/organizations/1/me":
            self._write_json(create_me())
            return
        if self.path == "/api/organizations/llm":
            self._write_json(create_settings())
            return

        resolved = Path(self.translate_path(self.path))
        if resolved.is_file():
            return super().do_GET()

        index_path = Path(self.static_dir) / "index.html"
        try:
            data = index_path.read_bytes()
        except FileNotFoundError:
            self.send_error(HTTPStatus.NOT_FOUND, "Not Found")
            return

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--mode", choices=["oss", "saas"], required=True)
    parser.add_argument("--directory", required=True)
    args = parser.parse_args()

    directory = os.path.abspath(args.directory)
    handler = lambda *handler_args, **handler_kwargs: MockFrontendHandler(
        *handler_args,
        directory=directory,
        app_mode=args.mode,
        **handler_kwargs,
    )
    server = ThreadingHTTPServer(("0.0.0.0", args.port), handler)
    print(
        f"Serving {args.mode} mock frontend from {directory} on http://0.0.0.0:{args.port}"
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
