import html
import json
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
from integrations.slack.slack_manager import SlackManager
from integrations.utils import (
    HOST_URL,
)
from server.auth.constants import (
    KEYCLOAK_CLIENT_ID,
    KEYCLOAK_REALM_NAME,
    KEYCLOAK_SERVER_URL_EXT,
)
from server.auth.token_manager import TokenManager
from server.constants import (
    SLACK_CLIENT_ID,
    SLACK_CLIENT_SECRET,
    SLACK_SIGNING_SECRET,
    SLACK_WEBHOOKS_ENABLED,
)
from server.logger import logger
from slack_sdk.oauth import AuthorizeUrlGenerator
from slack_sdk.signature import SignatureVerifier
from slack_sdk.web.async_client import AsyncWebClient
from storage.database import session_maker
from storage.slack_team_store import SlackTeamStore
from storage.slack_user import SlackUser
from storage.user_store import UserStore

from openhands.integrations.service_types import ProviderType
from openhands.server.shared import config, sio

signature_verifier = SignatureVerifier(signing_secret=SLACK_SIGNING_SECRET)
slack_router = APIRouter(prefix='/slack')

# Build https://slack.com/oauth/v2/authorize with sufficient query parameters
authorize_url_generator = AuthorizeUrlGenerator(
    client_id=SLACK_CLIENT_ID, scopes=['app_mentions:read', 'chat:write']
)
token_manager = TokenManager()

slack_manager = SlackManager(token_manager)
slack_team_store = SlackTeamStore.get_instance()


@slack_router.get('/install')
async def install(state: str = ''):
    """Forward into slack OAuth. (Most workflows can skip this and jump directly into slack authentication, so we skip OAuth state generation)"""
    url = authorize_url_generator.generate(state=state)
    return RedirectResponse(url)


@slack_router.get('/install-callback')
async def install_callback(
    request: Request, code: str = '', state: str = '', error: str = ''
):
    """Callback from slack authentication. Verifies, then forwards into keycloak authentication."""
    if not code or error:
        logger.warning(
            'slack_install_callback_error',
            extra={
                'code': code,
                'state': state,
                'error': error,
            },
        )
        return _html_response(
            title='Error',
            description=html.escape(error or 'No code provided'),
            status_code=400,
        )

    if not config.jwt_secret:
        logger.error('slack_install_callback_error JWT not configured.')
        return _html_response(
            title='Error',
            description=html.escape('JWT not configured'),
            status_code=500,
        )

    try:
        client = AsyncWebClient()  # no prepared token needed for this
        # Complete the installation by calling oauth.v2.access API method
        oauth_response = await client.oauth_v2_access(
            client_id=SLACK_CLIENT_ID,
            client_secret=SLACK_CLIENT_SECRET,
            redirect_uri=f'https://{request.url.netloc}{request.url.path}',
            code=code,
        )
        bot_access_token = oauth_response.get('access_token')
        team_id = oauth_response.get('team', {}).get('id')
        authed_user = oauth_response.get('authed_user') or {}

        # Create a state variable for keycloak oauth
        payload = {}
        if state:
            payload = jwt.decode(
                state, config.jwt_secret.get_secret_value(), algorithms=['HS256']
            )
        payload['slack_user_id'] = authed_user.get('id')
        payload['bot_access_token'] = bot_access_token
        payload['team_id'] = team_id

        state = jwt.encode(
            payload, config.jwt_secret.get_secret_value(), algorithm='HS256'
        )

        # Redirect into keycloak
        scope = quote('openid email profile offline_access')
        redirect_uri = f'{HOST_URL}/slack/keycloak-callback'
        auth_url = (
            f'{KEYCLOAK_SERVER_URL_EXT}/realms/{KEYCLOAK_REALM_NAME}/protocol/openid-connect/auth'
            f'?client_id={KEYCLOAK_CLIENT_ID}&response_type=code'
            f'&redirect_uri={redirect_uri}'
            f'&scope={scope}'
            f'&state={state}'
        )

        return RedirectResponse(auth_url)
    except Exception:  # type: ignore
        logger.error('unexpected_error', exc_info=True, stack_info=True)
        return _html_response(
            title='Error',
            description='Internal server Error',
            status_code=500,
        )


@slack_router.get('/keycloak-callback')
async def keycloak_callback(
    request: Request,
    background_tasks: BackgroundTasks,
    code: str = '',
    state: str = '',
    error: str = '',
):
    if not code or error:
        logger.warning(
            'problem_retrieving_keycloak_tokens',
            extra={
                'code': code,
                'state': state,
                'error': error,
            },
        )
        return _html_response(
            title='Error',
            description=html.escape(error or 'No code provided'),
            status_code=400,
        )

    if not config.jwt_secret:
        logger.error('problem_retrieving_keycloak_tokens JWT not configured.')
        return _html_response(
            title='Error',
            description=html.escape('JWT not configured'),
            status_code=500,
        )

    payload: dict[str, str] = jwt.decode(
        state, config.jwt_secret.get_secret_value(), algorithms=['HS256']
    )
    slack_user_id = payload['slack_user_id']
    bot_access_token = payload['bot_access_token']
    team_id = payload['team_id']

    # Retrieve the keycloak_user_id
    redirect_uri = f'{HOST_URL}{request.url.path}'
    (
        keycloak_access_token,
        keycloak_refresh_token,
    ) = await token_manager.get_keycloak_tokens(code, redirect_uri)
    if not keycloak_access_token or not keycloak_refresh_token:
        logger.warning(
            'problem_retrieving_keycloak_tokens',
            extra={
                'code': code,
                'state': state,
                'error': error,
            },
        )
        return _html_response(
            title='Failed to authenticate.',
            description=f'Please re-login into <a href="{HOST_URL}" style="color:#ecedee;text-decoration:underline;">OpenHands Cloud</a>. Then try <a href="https://docs.all-hands.dev/usage/cloud/slack-installation" style="color:#ecedee;text-decoration:underline;">installing the OpenHands Slack App</a> again',
            status_code=400,
        )

    user_info = await token_manager.get_user_info(keycloak_access_token)
    keycloak_user_id = user_info['sub']
    user = await UserStore.get_user_by_id_async(keycloak_user_id)
    if not user:
        return _html_response(
            title='Failed to authenticate.',
            description=f'Please re-login into <a href="{HOST_URL}" style="color:#ecedee;text-decoration:underline;">OpenHands Cloud</a>. Then try <a href="https://docs.all-hands.dev/usage/cloud/slack-installation" style="color:#ecedee;text-decoration:underline;">installing the OpenHands Slack App</a> again',
            status_code=400,
        )

    # These tokens are offline access tokens - store them!
    await token_manager.store_offline_token(keycloak_user_id, keycloak_refresh_token)

    idp: str = user_info.get('identity_provider', ProviderType.GITHUB)
    idp_type = 'oidc'
    if ':' in idp:
        idp, idp_type = idp.rsplit(':', 1)
        idp_type = idp_type.lower()
    await token_manager.store_idp_tokens(
        ProviderType(idp), keycloak_user_id, keycloak_access_token
    )

    # Retrieve bot token
    if team_id and bot_access_token:
        slack_team_store.create_team(team_id, bot_access_token)
    else:
        bot_access_token = slack_team_store.get_team_bot_token(team_id)

    if not bot_access_token:
        logger.error(
            f'Account linking failed, did not find slack team {team_id} for user {keycloak_user_id}'
        )
        return

    # Retrieve the display_name from slack
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
        # First delete any existing tokens
        session.query(SlackUser).filter(
            SlackUser.slack_user_id == slack_user_id
        ).delete()

        # Store the token
        session.add(slack_user)
        session.commit()

    message = Message(source=SourceType.SLACK, message=payload)

    background_tasks.add_task(slack_manager.receive_message, message)
    return _html_response(
        title='OpenHands Authentication Successful!',
        description='It is now safe to close this tab.',
        status_code=200,
    )


@slack_router.post('/on-event')
async def on_event(request: Request, background_tasks: BackgroundTasks):
    if not SLACK_WEBHOOKS_ENABLED:
        return JSONResponse({'success': 'slack_webhooks_disabled'})
    body = await request.body()
    payload = json.loads(body.decode())

    logger.info('slack_on_event', extra={'payload': payload})

    # First verify the signature
    if not signature_verifier.is_valid(
        body=body,
        timestamp=request.headers.get('x-slack-request-timestamp'),
        signature=request.headers.get('x-slack-signature'),
    ):
        raise HTTPException(status_code=403, detail='invalid_request')

    # Slack initially / periodically sends challenges and expects this response
    if 'challenge' in payload:
        return PlainTextResponse(payload['challenge'])

    # {"message": "slack_on_event", "severity": "INFO", "payload": {"token": "i8Al1OkFR99MafAxURXhRJ7b", "team_id": "T07E1S2M2Q6", "api_app_id": "A08MFF9S6FQ", "event": {"user": "U07G13E21DK", "type": "app_mention", "ts": "1744740589.879749", "client_msg_id": "4382e009-6717-4ed7-954b-f0eb3073b88e", "text": "<@U08MFFR1AR4> Flarglebargle!", "team": "T07E1S2M2Q6", "blocks": [{"type": "rich_text", "block_id": "ynJhY", "elements": [{"type": "rich_text_section", "elements": [{"type": "user", "user_id": "U08MFFR1AR4"}, {"type": "text", "text": " Flarglebargle!"}]}]}], "channel": "C08MYQ1PQS0", "event_ts": "1744740589.879749"}, "type": "event_callback", "event_id": "Ev08NE73GEUB", "event_time": 1744740589, "authorizations": [{"enterprise_id": None, "team_id": "T07E1S2M2Q6", "user_id": "U08MFFR1AR4", "is_bot": True, "is_enterprise_install": False}], "is_ext_shared_channel": False, "event_context": "4-eyJldCI6ImFwcF9tZW50aW9uIiwidGlkIjoiVDA3RTFTMk0yUTYiLCJhaWQiOiJBMDhNRkY5UzZGUSIsImNpZCI6IkMwOE1ZUTFQUVMwIn0"}}
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

    # Sometimes slack sends duplicates, so we need to make sure this is not a duplicate.
    redis = sio.manager.redis
    key = f'slack_msg:{client_msg_id}'
    created = await redis.set(key, 1, nx=True, ex=60)
    if not created:
        logger.info('slack_is_duplicate')
        return JSONResponse({'success': True})

    # TODO: Get team id
    payload = {
        'message_ts': message_ts,
        'thread_ts': thread_ts,
        'channel_id': channel_id,
        'user_msg': user_msg,
        'slack_user_id': slack_user_id,
        'team_id': team_id,
    }

    message = Message(
        source=SourceType.SLACK,
        message=payload,
    )

    background_tasks.add_task(slack_manager.receive_message, message)
    return JSONResponse({'success': True})


@slack_router.post('/on-form-interaction')
async def on_form_interaction(request: Request, background_tasks: BackgroundTasks):
    """We check the nonce to start a conversation"""
    if not SLACK_WEBHOOKS_ENABLED:
        return JSONResponse({'success': 'slack_webhooks_disabled'})

    body = await request.body()
    form = await request.form()
    payload = json.loads(form.get('payload'))

    logger.info('slack_on_form_interaction', extra={'payload': payload})

    # First verify the signature
    if not signature_verifier.is_valid(
        body=body,
        timestamp=request.headers.get('X-Slack-Request-Timestamp'),
        signature=request.headers.get('X-Slack-Signature'),
    ):
        raise HTTPException(status_code=403, detail='invalid_request')

    assert payload['type'] == 'block_actions'
    selected_repository = payload['actions'][0]['selected_option'][
        'value'
    ]  # Get the repository
    if selected_repository == '-':
        selected_repository = None
    slack_user_id = payload['user']['id']
    channel_id = payload['container']['channel_id']
    team_id = payload['team']['id']
    # Hack - get original message_ts from element name
    attribs = payload['actions'][0]['action_id'].split('repository_select:')[-1]
    message_ts, thread_ts = attribs.split(':')
    thread_ts = None if thread_ts == 'None' else thread_ts
    # Get the original message
    # Get the text message
    # Start the conversation

    payload = {
        'message_ts': message_ts,
        'thread_ts': thread_ts,
        'channel_id': channel_id,
        'slack_user_id': slack_user_id,
        'selected_repo': selected_repository,
        'team_id': team_id,
    }

    message = Message(
        source=SourceType.SLACK,
        message=payload,
    )

    background_tasks.add_task(slack_manager.receive_message, message)
    return JSONResponse({'success': True})


def _html_response(title: str, description: str, status_code: int) -> HTMLResponse:
    content = (
        '<style>body{background:#0d0f11;color:#ecedee;font-family:sans-serif;display:flex;justify-content:center;align-items:center;}</style>'
        '<div style="box-sizing:border-box;border:1px solid #454545;padding:24px;width:384px;background:#24272e;border-radius:0.75rem;text-align:center;">'
        f'<h1 style="font-size:24px;">{title}</h1>'
        f'<p>{description}</p>'
        '<div>'
    )
    return HTMLResponse(
        content=content,
        status_code=status_code,
    )
