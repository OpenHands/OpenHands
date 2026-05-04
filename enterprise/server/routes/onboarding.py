"""Onboarding submission endpoint.

Receives user onboarding selections and fires analytics event.
"""

from datetime import datetime, timezone
from typing import Union

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from openhands.app_server.user_auth import get_user_id

onboarding_router = APIRouter(prefix='/api', tags=['Onboarding'])


class OnboardingSubmission(BaseModel):
    selections: dict[str, Union[str, list[str]]]
    # question_id -> selected option(s)
    # e.g., {"role": "software_engineer", "org_size": "solo", "use_case": ["new_features", "fixing_bugs"]}


class OnboardingResponse(BaseModel):
    status: str
    redirect_url: str


@onboarding_router.post('/onboarding', response_model=OnboardingResponse)
async def submit_onboarding(
    body: OnboardingSubmission,
    user_id: str | None = Depends(get_user_id),
) -> OnboardingResponse:
    """Submit onboarding form selections and fire analytics event."""
    # ACTV-03: onboarding completed
    try:
        from openhands.analytics import get_analytics_service, resolve_analytics_context

        analytics = get_analytics_service()
        if analytics and user_id:
            ctx = await resolve_analytics_context(user_id)

            analytics.track_onboarding_completed(
                ctx=ctx,
                selections=body.selections,
            )

            # Associate onboarding timestamp with org group
            if ctx.org_id:
                analytics.group_identify(
                    ctx=ctx,
                    group_type='org',
                    group_key=ctx.org_id,
                    properties={
                        'onboarding_completed_at': datetime.now(
                            timezone.utc
                        ).isoformat(),
                    },
                )
    except Exception:
        import logging

        logging.getLogger(__name__).exception('analytics:onboarding_completed:failed')

    return OnboardingResponse(status='ok', redirect_url='/')
