"""Generic frontend-initiated analytics events.

A single ``POST /api/analytics/events`` endpoint accepts every event that the
browser needs to fire server-side because the underlying user action does not
hit any other backend endpoint (e.g. clicking the Pull Request button just
fills a chat suggestion).

How to add a new frontend event
-------------------------------
1. Define a ``BaseModel`` with a ``Literal[...]`` ``event_type`` field whose
   value is the exact PostHog event name to capture, plus any allowed event
   properties (each restricted to a constrained type so the frontend cannot
   smuggle arbitrary keys into PostHog).
2. Add the new model to ``FrontendEvent`` as another tagged-union member.
   Pydantic dispatches on ``event_type`` to pick the right schema, so unknown
   ``event_type`` values get rejected with HTTP 422 automatically.

Design notes
------------
- ``event_type`` is read from the validated body, not from a free-form query
  string, so only allow-listed events can ever reach PostHog.
- The handler swallows analytics exceptions so a PostHog/SDK outage cannot
  break the user-facing action that triggered the event.
- The endpoint always returns ``status='ok'`` once the payload validates;
  unauthenticated callers and analytics outages both no-op silently.
"""

import logging
from typing import Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from openhands.analytics import get_analytics_service, resolve_analytics_context
from openhands.app_server.user_auth import get_user_id

analytics_events_router = APIRouter(prefix='/api/analytics/events', tags=['Analytics'])

logger = logging.getLogger(__name__)


# Mirrors ``frontend/src/types/settings.ts::ProviderOptions``. Kept narrow so
# only known providers can be forwarded as PostHog properties.
GitProvider = Literal[
    'github',
    'gitlab',
    'bitbucket',
    'bitbucket_data_center',
    'azure_devops',
    'forgejo',
    'enterprise_sso',
]


class CreatePrButtonClickedEvent(BaseModel):
    """``create pr button clicked``: Pull Request button click in chat UI.

    Drives PostHog surveys (e.g. NPS) keyed off the PR action. The event
    name uses the lowercase-with-spaces convention shared by every other
    PostHog event in ``openhands/analytics/analytics_constants.py``.
    """

    event_type: Literal['create pr button clicked']
    git_provider: GitProvider | None = None


# Tagged union of every event the frontend may send. To add a new event,
# define another ``BaseModel`` above and append it here, e.g.:
#     FrontendEvent = Annotated[
#         CreatePrButtonClickedEvent | PushButtonClickedEvent,
#         Field(discriminator='event_type'),
#     ]
# With a single member the discriminator is unnecessary; pydantic still uses
# the ``Literal`` on ``event_type`` to reject unknown event types.
# ⚠️ When adding a second event, wrap in:
# Annotated[CreatePrButtonClickedEvent | NewEvent, Field(discriminator='event_type')]
FrontendEvent = CreatePrButtonClickedEvent


class AnalyticsEventResponse(BaseModel):
    status: str


@analytics_events_router.post('', response_model=AnalyticsEventResponse)
async def track_frontend_event(
    body: FrontendEvent,
    user_id: str | None = Depends(get_user_id),
) -> AnalyticsEventResponse:
    """Capture a frontend-initiated analytics event.

    ``body.event_type`` selects the PostHog event name; the remaining fields
    on the validated model become the event's properties. Telemetry failures
    are isolated so they never bubble up to the user's primary action.
    """
    try:
        analytics = get_analytics_service()
        if analytics and user_id:
            ctx = await resolve_analytics_context(user_id)
            analytics.capture(
                ctx=ctx,
                event=body.event_type,
                properties=body.model_dump(exclude={'event_type'}),
            )
    except Exception:
        logger.exception(
            'analytics:frontend_event:failed',
            extra={'event_type': body.event_type},
        )

    return AnalyticsEventResponse(status='ok')
