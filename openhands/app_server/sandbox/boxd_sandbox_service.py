"""Sandbox service backed by boxd cloud microVMs.

boxd (https://boxd.sh) runs each VM as a dedicated KVM process with
sub-millisecond warm suspend/resume. Each conversation gets one boxd VM
with the agent-server image installed and a subdomain proxy on port 60000.

Persistent state model: v1 stores sandbox metadata (sandbox_spec_id,
session_api_key hash, owning user) in the VM's env dict — there is no
app-side SQL table. This trades a O(1) DB lookup for an O(N)
``compute.box.list()`` scan on search, which is acceptable at boxd's
default per-user VM quota.
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from typing import Any, AsyncGenerator

import base62
from fastapi import Request
from pydantic import Field

from openhands.agent_server.utils import utc_now
from openhands.app_server.errors import SandboxError
from openhands.app_server.sandbox.sandbox_models import (
    AGENT_SERVER,
    VSCODE,
    ExposedUrl,
    SandboxInfo,
    SandboxPage,
    SandboxStatus,
)
from openhands.app_server.sandbox.sandbox_service import (
    ALLOW_CORS_ORIGINS_VARIABLE,
    SESSION_API_KEY_VARIABLE,
    WEBHOOK_CALLBACK_VARIABLE,
    SandboxService,
    SandboxServiceInjector,
)
from openhands.app_server.sandbox.sandbox_spec_models import SandboxSpecInfo
from openhands.app_server.sandbox.sandbox_spec_service import SandboxSpecService
from openhands.app_server.services.injector import InjectorState
from openhands.app_server.user.user_context import UserContext

_logger = logging.getLogger(__name__)

# boxd VM status → internal SandboxStatus. Lowercased keys; SDK returns
# mixed-case strings, we normalize on read.
STATUS_MAPPING: dict[str, SandboxStatus] = {
    'running': SandboxStatus.RUNNING,
    'suspended': SandboxStatus.PAUSED,
    'starting': SandboxStatus.STARTING,
    'creating': SandboxStatus.STARTING,
    'error': SandboxStatus.ERROR,
    'failed': SandboxStatus.ERROR,
    'stopped': SandboxStatus.MISSING,
    'destroyed': SandboxStatus.MISSING,
}

AGENT_SERVER_PORT = 60000
VSCODE_PORT = 60001
AGENT_PROXY_NAME = 'agent'
VSCODE_PROXY_NAME = 'vscode'

# Env keys stashed on each boxd VM so search/get can reconstruct
# SandboxInfo without an app-side database.
SPEC_ID_ENV = 'OH_SANDBOX_SPEC_ID'
USER_ID_ENV = 'OH_CREATED_BY_USER_ID'
SESSION_KEY_HASH_ENV = 'OH_SESSION_API_KEY_HASH'

# All boxd VMs we create are prefixed with this so search/list can
# distinguish ours from VMs created out-of-band by the same user.
VM_NAME_PREFIX = 'oh-'


def _hash_session_api_key(session_api_key: str) -> str:
    """SHA-256 of the session API key for storage in the box's env dict."""
    return hashlib.sha256(session_api_key.encode()).hexdigest()
