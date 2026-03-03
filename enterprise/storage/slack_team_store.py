from dataclasses import dataclass

from sqlalchemy.orm import sessionmaker
from storage.database import session_maker
from storage.slack_team import SlackTeam


@dataclass
class SlackTeamStore:
    session_maker: sessionmaker

    def get_team_bot_token(self, team_id: str) -> str | None:
        """
        Get a team's bot access token by team_id
        """
        with session_maker() as session:
            team = session.query(SlackTeam).filter(SlackTeam.team_id == team_id).first()
            return team.bot_access_token if team else None

    def create_team(
        self,
        team_id: str,
        bot_access_token: str,
    ) -> SlackTeam:
        """
        Create a new SlackTeam
        """
        slack_team = SlackTeam(team_id=team_id, bot_access_token=bot_access_token)
        with session_maker() as session:
            session.query(SlackTeam).filter(SlackTeam.team_id == team_id).delete()

            # Store the token
            session.add(slack_team)
            session.commit()
        return slack_team

    @classmethod
    def get_instance(cls):
        return SlackTeamStore(session_maker)
