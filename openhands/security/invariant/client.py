# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
import time
from typing import Any

import httpx


class InvariantClient:
    timeout: int = 120

    def __init__(self, server_url: str, session_id: str | None = None) -> None:
        self.server = server_url
        self.session_id, err = self._create_session(session_id)
        if err:
            raise RuntimeError(f'Failed to create session: {err}')
        self.Policy = self._Policy(self)
        self.Monitor = self._Monitor(self)

    def _create_session(
        self, session_id: str | None = None
    ) -> tuple[str | None, Exception | None]:
        elapsed = 0
        while elapsed < self.timeout:
            try:
                if session_id:
                    response = httpx.get(
                        f'{self.server}/session/new?session_id={session_id}', timeout=60
                    )
                else:
                    response = httpx.get(f'{self.server}/session/new', timeout=60)
                response.raise_for_status()
                return response.json().get('id'), None
            except (httpx.NetworkError, httpx.TimeoutException):
                elapsed += 1
                time.sleep(1)
            except httpx.HTTPError as http_err:
                return None, http_err
            except Exception as err:
                return None, err
        return None, ConnectionError('Connection timed out')

    def close_session(self) -> Exception | None:
        try:
            response = httpx.delete(
                f'{self.server}/session/?session_id={self.session_id}', timeout=60
            )
            response.raise_for_status()
        except (ConnectionError, httpx.TimeoutException, httpx.HTTPError) as err:
            return err
        return None

    class _Policy:
        def __init__(self, invariant: 'InvariantClient') -> None:
            self.server = invariant.server
            self.session_id = invariant.session_id
            self.policy_id: str | None = None

        def _create_policy(self, rule: str) -> tuple[str | None, Exception | None]:
            try:
                response = httpx.post(
                    f'{self.server}/policy/new?session_id={self.session_id}',
                    json={'rule': rule},
                    timeout=60,
                )
                response.raise_for_status()
                return response.json().get('policy_id'), None
            except (ConnectionError, httpx.TimeoutException, httpx.HTTPError) as err:
                return None, err

        def get_template(self) -> tuple[str | None, Exception | None]:
            try:
                response = httpx.get(
                    f'{self.server}/policy/template',
                    timeout=60,
                )
                response.raise_for_status()
                return response.json(), None
            except (ConnectionError, httpx.TimeoutException, httpx.HTTPError) as err:
                return None, err

        def from_string(self, rule: str) -> 'InvariantClient._Policy':
            policy_id, err = self._create_policy(rule)
            if err:
                raise err
            self.policy_id = policy_id
            return self

        def analyze(self, trace: list[dict[str, Any]]) -> tuple[Any, Exception | None]:
            try:
                response = httpx.post(
                    f'{self.server}/policy/{self.policy_id}/analyze?session_id={self.session_id}',
                    json={'trace': trace},
                    timeout=60,
                )
                response.raise_for_status()
                return response.json(), None
            except (ConnectionError, httpx.TimeoutException, httpx.HTTPError) as err:
                return None, err

    class _Monitor:
        def __init__(self, invariant: 'InvariantClient') -> None:
            self.server = invariant.server
            self.session_id = invariant.session_id
            self.policy = ''
            self.monitor_id: str | None = None

        def _create_monitor(self, rule: str) -> tuple[str | None, Exception | None]:
            try:
                response = httpx.post(
                    f'{self.server}/monitor/new?session_id={self.session_id}',
                    json={'rule': rule},
                    timeout=60,
                )
                response.raise_for_status()
                return response.json().get('monitor_id'), None
            except (ConnectionError, httpx.TimeoutException, httpx.HTTPError) as err:
                return None, err

        def from_string(self, rule: str) -> 'InvariantClient._Monitor':
            monitor_id, err = self._create_monitor(rule)
            if err:
                raise err
            self.monitor_id = monitor_id
            self.policy = rule
            return self

        def check(
            self,
            past_events: list[dict[str, Any]],
            pending_events: list[dict[str, Any]],
        ) -> tuple[Any, Exception | None]:
            try:
                response = httpx.post(
                    f'{self.server}/monitor/{self.monitor_id}/check?session_id={self.session_id}',
                    json={'past_events': past_events, 'pending_events': pending_events},
                    timeout=60,
                )
                response.raise_for_status()
                return response.json(), None
            except (ConnectionError, httpx.TimeoutException, httpx.HTTPError) as err:
                return None, err
