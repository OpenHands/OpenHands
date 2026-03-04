from sqlalchemy import Boolean, Column, Integer, String
from storage.base import Base


class UserRepositoryMap(Base):
    """Represents a map between user id and repo ids
    """

    __tablename__ = 'user-repos'
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, nullable=False)
    repo_id = Column(String, nullable=False)
    admin = Column(Boolean, nullable=True)
