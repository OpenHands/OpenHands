from sqlalchemy import Boolean, Column, Integer, String
from storage.base import Base


class StoredRepository(Base):  # type: ignore
    """Represents a repositories fetched from git providers.
    """

    __tablename__ = 'repos'
    id = Column(Integer, primary_key=True, autoincrement=True)
    repo_name = Column(String, nullable=False)
    repo_id = Column(String, nullable=False)  # {provider}##{id} format
    is_public = Column(Boolean, nullable=False)
    has_microagent = Column(Boolean, nullable=True)
    has_setup_script = Column(Boolean, nullable=True)
