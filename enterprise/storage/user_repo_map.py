from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column
from storage.base import Base


class UserRepositoryMap(Base):
    """
    Represents a map between user id and repo ids
    """

    __tablename__ = 'user-repos'

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String, nullable=False)
    repo_id: Mapped[str] = mapped_column(String, nullable=False)
    admin: Mapped[bool | None] = mapped_column(nullable=True)
