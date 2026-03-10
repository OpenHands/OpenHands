from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import delete, select
from storage.database import a_session_maker
from storage.slack_team import SlackTeam


@dataclass
class SlackTeamStore:
    async def get_team_bot_token(self, team_id: str) -> str | None:
        """
        Get a team's bot access token by team_id
        """
        async with a_session_maker() as session:
            result = await session.execute(
                select(SlackTeam).where(SlackTeam.team_id == team_id)
            )
            team = result.scalar_one_or_none()
            return team.bot_access_token if team else None

    async def create_team(
        self,
        team_id: str,
        bot_access_token: str,
    ) -> SlackTeam:
        """
        Create a new SlackTeam
        """
        slack_team = SlackTeam(team_id=team_id, bot_access_token=bot_access_token)
        async with a_session_maker() as session:
            await session.execute(delete(SlackTeam).where(SlackTeam.team_id == team_id))
            session.add(slack_team)
            await session.commit()
        return slack_team

    @classmethod
    def get_instance(cls) -> SlackTeamStore:
        return SlackTeamStore()
