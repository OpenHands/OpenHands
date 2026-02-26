"""Slack integration routes for OpenHands.

Self-contained FastAPI router for the Slack integration plugin.
All routes receive configuration via the SlackPluginConfig rather
than importing from enterprise server constants directly.
"""

from __future__ import annotations

import html
import json
import logging
from urllib.parse import quote

import jwt
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    RedirectResponse,
)
from integrations.models import Message, SourceType
from integrations.slack.plugin import SlackPluginConfig
from integrations.slack.slack_manager import SlackManager
from server.auth.token_manager import TokenManager
from slack_sdk.oauth import AuthorizeUrlGenerator
from slack_sdk.signature import SignatureVerifier
from slack_sdk.web.async_client import AsyncWebClient
from storage.database import session_maker
from storage.slack_team_store import SlackTeamStore
from storage.slack_user import SlackUser
from storage.user_store import UserStore

from openhands.integrations.service_types import ProviderType
from openhands.server.shared import config as openhands_config
from openhands.server.shared import sio

_logger = logging.getLogger(__name__)


def create_slack_router(plugin_config: SlackPluginConfig) -> APIRouter:
    """Create and return a fully configured Slack APIRouter.

    This factory function encapsulates all route definitions and their
    dependencies, making the router self-contained and testable.

    Args:
        plugin_config: Slack plugin configuration values.

    Returns:
        A FastAPI APIRouter with all Slack endpoints registered.
    """
    router = APIRouter(prefix='/slack')

    signature_verifier = SignatureVerifier(
        signing_secret=plugin_config.signing_secret
    )
    authorize_url_generator = AuthorizeUrlGenerator(
        client_id=plugin_config.client_id,
        scopes=['app_mentions:read', 'chat:write'],
    )
    token_manager = TokenManager()
    slack_manager = SlackManager(token_manager)
    slack_team_store = SlackTeamStore.get_instance()

    # ------------------------------------------------------------------
    # OAuth flow
    # ------------------------------------------------------------------

    @router.get('/install')
    async def install(state: str = ''):
        """Forward into Slack OAuth."""
        url = authorize_url_generator.generate(state=state)
        return RedirectResponse(url)

    @router.get('/install-callback')
    async def install_callback(
        request: Request, code: str = '', state: str = '', error: str = ''
    ):
        """Callback from Slack authentication. Verifies, then forwards into Keycloak authentication."""
        if not code or error:
            _logger.warning(
                'slack_install_callback_error',
                extra={'code': code, 'state': state, 'error': error},
            )
            return _html_response(
                title='Error',
                description=html.escape(error or 'No code provided'),
                status_code=400,
            )

        if not openhands_config.jwt_secret:
            _logger.error('slack_install_callback_error JWT not configured.')
            return _html_response(
                title='Error',
                description=html.escape('JWT not configured'),
                status_code=500,
            )

        try:
            client = AsyncWebClient()
            oauth_response = await client.oauth_v2_access(
                client_id=plugin_config.client_id,
                client_secret=plugin_config.client_secret,
                redirect_uri=f'https://{request.url.netloc}{request.url.path}',
                code=code,
            )
            bot_access_token = oauth_response.get('access_token')
            team_id = oauth_response.get('team', {}).get('id')
            authed_user = oauth_response.get('authed_user') or {}

            payload = {}
            if state:
                payload = jwt.decode(
                    state,
                    openhands_config.jwt_secret.get_secret_value(),
                    algorithms=['HS256'],
                )
            payload['slack_user_id'] = authed_user.get('id')
            payload['bot_access_token'] = bot_access_token
            payload['team_id'] = team_id

            state = jwt.encode(
                payload,
                openhands_config.jwt_secret.get_secret_value(),
                algorithm='HS256',
            )

            scope = quote('openid email profile offline_access')
            redirect_uri = f'{plugin_config.host_url}/slack/keycloak-callback'
            auth_url = (
                f'{plugin_config.keycloak_server_url_ext}/realms/{plugin_config.keycloak_realm_name}'
                f'/protocol/openid-connect/auth'
                f'?client_id={plugin_config.keycloak_client_id}&response_type=code'
                f'&redirect_uri={redirect_uri}'
                f'&scope={scope}'
                f'&state={state}'
            )
            return RedirectResponse(auth_url)
        except Exception:
            _logger.error('unexpected_error', exc_info=True, stack_info=True)
            return _html_response(
                title='Error',
                description='Internal server Error',
                status_code=500,
            )

    @router.get('/keycloak-callback')
    async def keycloak_callback(
        request: Request,
        background_tasks: BackgroundTasks,
        code: str = '',
        state: str = '',
        error: str = '',
    ):
        host_url = plugin_config.host_url

        if not code or error:
            _logger.warning(
                'problem_retrieving_keycloak_tokens',
                extra={'code': code, 'state': state, 'error': error},
            )
            return _html_response(
                title='Error',
                description=html.escape(error or 'No code provided'),
                status_code=400,
            )

        if not openhands_config.jwt_secret:
            _logger.error('problem_retrieving_keycloak_tokens JWT not configured.')
            return _html_response(
                title='Error',
                description=html.escape('JWT not configured'),
                status_code=500,
            )

        payload: dict[str, str] = jwt.decode(
            state,
            openhands_config.jwt_secret.get_secret_value(),
            algorithms=['HS256'],
        )
        slack_user_id = payload['slack_user_id']
        bot_access_token = payload['bot_access_token']
        team_id = payload['team_id']

        redirect_uri = f'{host_url}{request.url.path}'
        (
            keycloak_access_token,
            keycloak_refresh_token,
        ) = await token_manager.get_keycloak_tokens(code, redirect_uri)
        if not keycloak_access_token or not keycloak_refresh_token:
            _logger.warning(
                'problem_retrieving_keycloak_tokens',
                extra={'code': code, 'state': state, 'error': error},
            )
            return _html_response(
                title='Failed to authenticate.',
                description=(
                    f'Please re-login into <a href="{host_url}" style="color:#ecedee;text-decoration:underline;">OpenHands Cloud</a>. '
                    f'Then try <a href="https://docs.all-hands.dev/usage/cloud/slack-installation" style="color:#ecedee;text-decoration:underline;">installing the OpenHands Slack App</a> again'
                ),
                status_code=400,
            )

        user_info = await token_manager.get_user_info(keycloak_access_token)
        keycloak_user_id = user_info['sub']
        user = await UserStore.get_user_by_id_async(keycloak_user_id)
        if not user:
            return _html_response(
                title='Failed to authenticate.',
                description=(
                    f'Please re-login into <a href="{host_url}" style="color:#ecedee;text-decoration:underline;">OpenHands Cloud</a>. '
                    f'Then try <a href="https://docs.all-hands.dev/usage/cloud/slack-installation" style="color:#ecedee;text-decoration:underline;">installing the OpenHands Slack App</a> again'
                ),
                status_code=400,
            )

        await token_manager.store_offline_token(
            keycloak_user_id, keycloak_refresh_token
        )

        idp: str = user_info.get('identity_provider', ProviderType.GITHUB)
        idp_type = 'oidc'
        if ':' in idp:
            idp, idp_type = idp.rsplit(':', 1)
            idp_type = idp_type.lower()
        await token_manager.store_idp_tokens(
            ProviderType(idp), keycloak_user_id, keycloak_access_token
        )

        if team_id and bot_access_token:
            slack_team_store.create_team(team_id, bot_access_token)
        else:
            bot_access_token = slack_team_store.get_team_bot_token(team_id)

        if not bot_access_token:
            _logger.error(
                f'Account linking failed, did not find slack team {team_id} for user {keycloak_user_id}'
            )
            return

        client = AsyncWebClient(token=bot_access_token)
        slack_user_info = await client.users_info(user=slack_user_id)
        slack_display_name = slack_user_info.data['user']['profile']['display_name']
        slack_user = SlackUser(
            keycloak_user_id=keycloak_user_id,
            org_id=user.current_org_id,
            slack_user_id=slack_user_id,
            slack_display_name=slack_display_name,
        )

        with session_maker(expire_on_commit=False) as session:
            session.query(SlackUser).filter(
                SlackUser.slack_user_id == slack_user_id
            ).delete()
            session.add(slack_user)
            session.commit()

        message = Message(source=SourceType.SLACK, message=payload)
        background_tasks.add_task(slack_manager.receive_message, message)
        return _html_response(
            title='OpenHands Authentication Successful!',
            description='It is now safe to close this tab.',
            status_code=200,
        )

    # ------------------------------------------------------------------
    # Webhook event handlers
    # ------------------------------------------------------------------

    @router.post('/on-event')
    async def on_event(request: Request, background_tasks: BackgroundTasks):
        if not plugin_config.webhooks_enabled:
            return JSONResponse({'success': 'slack_webhooks_disabled'})
        body = await request.body()
        payload = json.loads(body.decode())

        _logger.info('slack_on_event', extra={'payload': payload})

        if not signature_verifier.is_valid(
            body=body,
            timestamp=request.headers.get('x-slack-request-timestamp'),
            signature=request.headers.get('x-slack-signature'),
        ):
            raise HTTPException(status_code=403, detail='invalid_request')

        if 'challenge' in payload:
            return PlainTextResponse(payload['challenge'])

        if payload.get('type') != 'event_callback':
            return JSONResponse({'success': True})

        event = payload['event']
        user_msg = event['text']
        assert event['type'] == 'app_mention'
        client_msg_id = event['client_msg_id']
        message_ts = event['ts']
        thread_ts = event.get('thread_ts')
        channel_id = event['channel']
        slack_user_id = event['user']
        team_id = payload['team_id']

        # Deduplicate messages via Redis
        redis = sio.manager.redis
        key = f'slack_msg:{client_msg_id}'
        created = await redis.set(key, 1, nx=True, ex=60)
        if not created:
            _logger.info('slack_is_duplicate')
            return JSONResponse({'success': True})

        payload = {
            'message_ts': message_ts,
            'thread_ts': thread_ts,
            'channel_id': channel_id,
            'user_msg': user_msg,
            'slack_user_id': slack_user_id,
            'team_id': team_id,
        }

        message = Message(source=SourceType.SLACK, message=payload)
        background_tasks.add_task(slack_manager.receive_message, message)
        return JSONResponse({'success': True})

    @router.post('/on-form-interaction')
    async def on_form_interaction(
        request: Request, background_tasks: BackgroundTasks
    ):
        """Handle interactive form submissions (e.g. repository selection dropdown)."""
        if not plugin_config.webhooks_enabled:
            return JSONResponse({'success': 'slack_webhooks_disabled'})

        body = await request.body()
        form = await request.form()
        payload = json.loads(form.get('payload'))

        _logger.info('slack_on_form_interaction', extra={'payload': payload})

        if not signature_verifier.is_valid(
            body=body,
            timestamp=request.headers.get('X-Slack-Request-Timestamp'),
            signature=request.headers.get('X-Slack-Signature'),
        ):
            raise HTTPException(status_code=403, detail='invalid_request')

        assert payload['type'] == 'block_actions'
        selected_repository = payload['actions'][0]['selected_option']['value']
        if selected_repository == '-':
            selected_repository = None
        slack_user_id = payload['user']['id']
        channel_id = payload['container']['channel_id']
        team_id = payload['team']['id']
        attribs = payload['actions'][0]['action_id'].split('repository_select:')[-1]
        message_ts, thread_ts = attribs.split(':')
        thread_ts = None if thread_ts == 'None' else thread_ts

        payload = {
            'message_ts': message_ts,
            'thread_ts': thread_ts,
            'channel_id': channel_id,
            'slack_user_id': slack_user_id,
            'selected_repo': selected_repository,
            'team_id': team_id,
        }

        message = Message(source=SourceType.SLACK, message=payload)
        background_tasks.add_task(slack_manager.receive_message, message)
        return JSONResponse({'success': True})

    return router


def _html_response(title: str, description: str, status_code: int) -> HTMLResponse:
    content = (
        '<style>body{background:#0d0f11;color:#ecedee;font-family:sans-serif;display:flex;justify-content:center;align-items:center;}</style>'
        '<div style="box-sizing:border-box;border:1px solid #454545;padding:24px;width:384px;background:#24272e;border-radius:0.75rem;text-align:center;">'
        f'<h1 style="font-size:24px;">{title}</h1>'
        f'<p>{description}</p>'
        '<div>'
    )
    return HTMLResponse(content=content, status_code=status_code)
