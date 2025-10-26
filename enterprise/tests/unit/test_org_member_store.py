import uuid
from unittest.mock import patch

# Mock the database module before importing OrgMemberStore
with patch('storage.database.engine'), patch('storage.database.a_engine'):
    from storage.org import Org
    from storage.org_member import OrgMember
    from storage.org_member_store import OrgMemberStore
    from storage.role import Role
    from storage.user import User


def test_get_org_members(session_maker):
    # Test getting org_members by org ID
    with session_maker() as session:
        # Create test data
        org = Org(name='test-org')
        session.add(org)
        session.flush()

        user1 = User(id=uuid.uuid4(), current_org_id=org.id)
        user2 = User(id=uuid.uuid4(), current_org_id=org.id)
        role = Role(name='admin', rank=1)
        session.add_all([user1, user2, role])
        session.flush()

        org_member1 = OrgMember(
            org_id=org.id,
            user_id=user1.id,
            role_id=role.id,
            llm_api_key='test-key-1',
            status='active',
        )
        org_member2 = OrgMember(
            org_id=org.id,
            user_id=user2.id,
            role_id=role.id,
            llm_api_key='test-key-2',
            status='active',
        )
        session.add_all([org_member1, org_member2])
        session.commit()
        org_id = org.id

    # Test retrieval
    with patch('storage.org_member_store.session_maker', session_maker):
        org_members = OrgMemberStore.get_org_members(org_id)
        assert len(org_members) == 2
        api_keys = [om.llm_api_key.get_secret_value() for om in org_members]
        assert 'test-key-1' in api_keys
        assert 'test-key-2' in api_keys


def test_get_user_orgs(session_maker):
    # Test getting org_members by user ID
    with session_maker() as session:
        # Create test data
        org1 = Org(name='test-org-1')
        org2 = Org(name='test-org-2')
        session.add_all([org1, org2])
        session.flush()

        user = User(id=uuid.uuid4(), current_org_id=org1.id)
        role = Role(name='admin', rank=1)
        session.add_all([user, role])
        session.flush()

        org_member1 = OrgMember(
            org_id=org1.id,
            user_id=user.id,
            role_id=role.id,
            llm_api_key='test-key-1',
            status='active',
        )
        org_member2 = OrgMember(
            org_id=org2.id,
            user_id=user.id,
            role_id=role.id,
            llm_api_key='test-key-2',
            status='active',
        )
        session.add_all([org_member1, org_member2])
        session.commit()
        user_id = user.id

    # Test retrieval
    with patch('storage.org_member_store.session_maker', session_maker):
        org_members = OrgMemberStore.get_user_orgs(user_id)
        assert len(org_members) == 2
        api_keys = [ou.llm_api_key.get_secret_value() for ou in org_members]
        assert 'test-key-1' in api_keys
        assert 'test-key-2' in api_keys


def test_get_org_member(session_maker):
    # Test getting org_member by org and user ID
    with session_maker() as session:
        # Create test data
        org = Org(name='test-org')
        session.add(org)
        session.flush()

        user = User(id=uuid.uuid4(), current_org_id=org.id)
        role = Role(name='admin', rank=1)
        session.add_all([user, role])
        session.flush()

        org_member = OrgMember(
            org_id=org.id,
            user_id=user.id,
            role_id=role.id,
            llm_api_key='test-key',
            status='active',
        )
        session.add(org_member)
        session.commit()
        org_id = org.id
        user_id = user.id

    # Test retrieval
    with patch('storage.org_member_store.session_maker', session_maker):
        retrieved_org_member = OrgMemberStore.get_org_member(org_id, user_id)
        assert retrieved_org_member is not None
        assert retrieved_org_member.org_id == org_id
        assert retrieved_org_member.user_id == user_id
        assert retrieved_org_member.llm_api_key.get_secret_value() == 'test-key'


def test_add_user_to_org(session_maker):
    # Test adding a user to an org
    with session_maker() as session:
        # Create test data
        org = Org(name='test-org')
        session.add(org)
        session.flush()

        user = User(id=uuid.uuid4(), current_org_id=org.id)
        role = Role(name='admin', rank=1)
        session.add_all([user, role])
        session.commit()
        org_id = org.id
        user_id = user.id
        role_id = role.id

    # Test creation
    with patch('storage.org_member_store.session_maker', session_maker):
        org_member = OrgMemberStore.add_user_to_org(
            org_id=org_id,
            user_id=user_id,
            role_id=role_id,
            llm_api_key='new-test-key',
            status='active',
        )

        assert org_member is not None
        assert org_member.org_id == org_id
        assert org_member.user_id == user_id
        assert org_member.role_id == role_id
        assert org_member.llm_api_key.get_secret_value() == 'new-test-key'
        assert org_member.status == 'active'


def test_update_user_role_in_org(session_maker):
    # Test updating user role in org
    with session_maker() as session:
        # Create test data
        org = Org(name='test-org')
        session.add(org)
        session.flush()

        user = User(id=uuid.uuid4(), current_org_id=org.id)
        role1 = Role(name='admin', rank=1)
        role2 = Role(name='user', rank=2)
        session.add_all([user, role1, role2])
        session.flush()

        org_member = OrgMember(
            org_id=org.id,
            user_id=user.id,
            role_id=role1.id,
            llm_api_key='test-key',
            status='active',
        )
        session.add(org_member)
        session.commit()
        org_id = org.id
        user_id = user.id
        role2_id = role2.id

    # Test update
    with patch('storage.org_member_store.session_maker', session_maker):
        updated_org_member = OrgMemberStore.update_user_role_in_org(
            org_id=org_id, user_id=user_id, role_id=role2_id, status='inactive'
        )

        assert updated_org_member is not None
        assert updated_org_member.role_id == role2_id
        assert updated_org_member.status == 'inactive'


def test_update_user_role_in_org_not_found(session_maker):
    # Test updating org_member that doesn't exist
    from uuid import uuid4

    with patch('storage.org_member_store.session_maker', session_maker):
        updated_org_member = OrgMemberStore.update_user_role_in_org(
            org_id=uuid4(), user_id=99999, role_id=1
        )
        assert updated_org_member is None


def test_remove_user_from_org(session_maker):
    # Test removing a user from an org
    with session_maker() as session:
        # Create test data
        org = Org(name='test-org')
        session.add(org)
        session.flush()

        user = User(id=uuid.uuid4(), current_org_id=org.id)
        role = Role(name='admin', rank=1)
        session.add_all([user, role])
        session.flush()

        org_member = OrgMember(
            org_id=org.id,
            user_id=user.id,
            role_id=role.id,
            llm_api_key='test-key',
            status='active',
        )
        session.add(org_member)
        session.commit()
        org_id = org.id
        user_id = user.id

    # Test removal
    with patch('storage.org_member_store.session_maker', session_maker):
        result = OrgMemberStore.remove_user_from_org(org_id, user_id)
        assert result is True

        # Verify it's removed
        retrieved_org_member = OrgMemberStore.get_org_member(org_id, user_id)
        assert retrieved_org_member is None


def test_remove_user_from_org_not_found(session_maker):
    # Test removing user from org that doesn't exist
    from uuid import uuid4

    with patch('storage.org_member_store.session_maker', session_maker):
        result = OrgMemberStore.remove_user_from_org(uuid4(), 99999)
        assert result is False
