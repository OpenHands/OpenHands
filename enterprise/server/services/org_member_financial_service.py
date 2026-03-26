"""Service for managing organization member financial data."""

from uuid import UUID

from server.routes.org_models import (
    OrgMemberFinancialPage,
    OrgMemberFinancialResponse,
)
from storage.lite_llm_manager import LiteLlmManager
from storage.org_member_store import OrgMemberStore

from openhands.core.logger import openhands_logger as logger


class OrgMemberFinancialService:
    """Service for organization member financial data operations."""

    @staticmethod
    async def get_org_members_financial_data(
        org_id: UUID,
        page_id: str | None = None,
        limit: int = 10,
        email_filter: str | None = None,
    ) -> OrgMemberFinancialPage:
        """Get paginated financial data for organization members.

        Fetches member list from database and joins with financial data from LiteLLM.

        Args:
            org_id: Organization UUID
            page_id: Offset encoded as string (e.g., "0", "10", "20")
            limit: Maximum items per page (default 10)
            email_filter: Optional case-insensitive partial email match

        Returns:
            OrgMemberFinancialPage: Paginated response with financial data

        Raises:
            ValueError: If page_id is invalid
        """
        # Parse page_id to get offset
        offset = 0
        if page_id is not None:
            try:
                offset = int(page_id)
                if offset < 0:
                    raise ValueError('page_id must be non-negative')
            except ValueError as e:
                raise ValueError(f'Invalid page_id: {page_id}') from e

        # Fetch paginated members from database
        members, total_count = await OrgMemberStore.get_org_members_paginated(
            org_id=org_id,
            offset=offset,
            limit=limit,
            email_filter=email_filter,
        )

        if not members:
            return OrgMemberFinancialPage(
                items=[],
                current_page=(offset // limit) + 1,
                per_page=limit,
                next_page_id=None,
            )

        # Fetch financial data from LiteLLM for the entire team
        # This is a single API call that returns all team members' data
        try:
            financial_data = await LiteLlmManager.get_team_members_financial_data(
                str(org_id)
            )
        except Exception as e:
            logger.error(
                'Failed to fetch financial data from LiteLLM',
                extra={'org_id': str(org_id), 'error': str(e)},
            )
            # Return empty financial data if LiteLLM is unavailable
            financial_data = {}

        # Build response items by joining DB members with LiteLLM financial data
        items: list[OrgMemberFinancialResponse] = []
        for member in members:
            user = member.user
            user_id_str = str(member.user_id)

            # Get financial data for this user (or defaults if not found)
            user_financial = financial_data.get(user_id_str, {})
            spend = user_financial.get('spend', 0) or 0
            max_budget = user_financial.get('max_budget')

            # Calculate current budget (remaining)
            if max_budget is not None:
                current_budget = max(max_budget - spend, 0)
            else:
                # If no max_budget, current_budget is unlimited (represented as 0)
                current_budget = 0

            items.append(
                OrgMemberFinancialResponse(
                    user_id=user_id_str,
                    email=user.email if user else None,
                    lifetime_spend=spend,
                    current_budget=current_budget,
                    max_budget=max_budget,
                )
            )

        # Calculate current page (1-indexed)
        current_page = (offset // limit) + 1

        # Calculate next_page_id
        next_offset = offset + limit
        next_page_id = str(next_offset) if next_offset < total_count else None

        logger.debug(
            'OrgMemberFinancialService:get_org_members_financial_data:success',
            extra={
                'org_id': str(org_id),
                'items_count': len(items),
                'current_page': current_page,
                'total_count': total_count,
            },
        )

        return OrgMemberFinancialPage(
            items=items,
            current_page=current_page,
            per_page=limit,
            next_page_id=next_page_id,
        )
