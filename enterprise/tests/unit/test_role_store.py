from unittest.mock import patch

# Mock the database module before importing RoleStore
with patch('storage.database.engine'), patch('storage.database.a_engine'):
    from storage.role import Role
    from storage.role_store import RoleStore


def test_get_role_by_id(session_maker):
    # Test getting role by ID
    with session_maker() as session:
        # Create a test role
        role = Role(name='admin', rank=1)
        session.add(role)
        session.commit()
        role_id = role.id

    # Test retrieval
    with patch('storage.role_store.session_maker', session_maker):
        retrieved_role = RoleStore.get_role_by_id(role_id)
        assert retrieved_role is not None
        assert retrieved_role.id == role_id
        assert retrieved_role.name == 'admin'


def test_get_role_by_id_not_found(session_maker):
    # Test getting role by ID when it doesn't exist
    with patch('storage.role_store.session_maker', session_maker):
        retrieved_role = RoleStore.get_role_by_id(99999)
        assert retrieved_role is None


def test_get_role_by_name(session_maker):
    # Test getting role by name
    with session_maker() as session:
        # Create a test role
        role = Role(name='admin', rank=1)
        session.add(role)
        session.commit()
        role_id = role.id

    # Test retrieval
    with patch('storage.role_store.session_maker', session_maker):
        retrieved_role = RoleStore.get_role_by_name('admin')
        assert retrieved_role is not None
        assert retrieved_role.id == role_id
        assert retrieved_role.name == 'admin'


def test_get_role_by_name_not_found(session_maker):
    # Test getting role by name when it doesn't exist
    with patch('storage.role_store.session_maker', session_maker):
        retrieved_role = RoleStore.get_role_by_name('nonexistent')
        assert retrieved_role is None


def test_list_roles(session_maker):
    # Test listing all roles
    with session_maker() as session:
        # Create test roles
        role1 = Role(name='admin', rank=1)
        role2 = Role(name='user', rank=2)
        session.add_all([role1, role2])
        session.commit()

    # Test listing
    with patch('storage.role_store.session_maker', session_maker):
        roles = RoleStore.list_roles()
        assert len(roles) >= 2
        role_names = [role.name for role in roles]
        assert 'admin' in role_names
        assert 'user' in role_names


def test_create_role(session_maker):
    # Test creating a new role
    with patch('storage.role_store.session_maker', session_maker):
        role = RoleStore.create_role(name='moderator', rank=2)

        assert role is not None
        assert role.name == 'moderator'
        assert role.rank == 2
        assert role.id is not None
