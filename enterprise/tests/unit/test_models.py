"""
Test that the models are correctly defined.
"""

from uuid import uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from storage.base import Base
from storage.org import Org
from storage.org_member import OrgMember
from storage.user import User


@pytest.fixture
def engine():
    engine = create_engine('sqlite:///:memory:')
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture
def session_maker(engine):
    return sessionmaker(bind=engine)


def test_user_model(session_maker):
    """Test that the User model works correctly."""
    with session_maker() as session:
        # Create a test org
        org = Org(name='test_org')
        session.add(org)
        session.flush()

        # Create a test user
        test_user_id = uuid4()
        user = User(id=test_user_id, current_org_id=org.id, language='en')
        session.add(user)
        session.flush()

        # Create org_member relationship
        org_member = OrgMember(
            org_id=org.id,
            user_id=user.id,
            role_id=1,
            llm_api_key='test-api-key',
            status='active',
        )
        session.add(org_member)
        session.commit()

        # Query the user
        queried_user = session.query(User).filter(User.id == test_user_id).first()
        assert queried_user is not None
        assert queried_user.language == 'en'

        # Query the org
        queried_org = session.query(Org).filter(Org.id == org.id).first()
        assert queried_org is not None
        assert queried_org.name == 'test_org'

        # Query the org_member relationship
        queried_org_member = (
            session.query(OrgMember)
            .filter(OrgMember.org_id == org.id, OrgMember.user_id == user.id)
            .first()
        )
        assert queried_org_member is not None
        assert queried_org_member.llm_api_key.get_secret_value() == 'test-api-key'
