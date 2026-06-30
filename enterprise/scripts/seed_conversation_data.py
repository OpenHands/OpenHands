#!/usr/bin/env python3
"""
Seed script to populate the database with realistic conversation data for testing.

Usage:
    # With default settings (assumes local PostgreSQL)
    python -m enterprise.scripts.seed_conversation_data

    # With custom database URL
    DATABASE_URL=postgresql://user:pass@host:5432/db python -m enterprise.scripts.seed_conversation_data

    # With options
    python -m enterprise.scripts.seed_conversation_data --org-count 3 --conversations-per-org 50
"""

import argparse
import random
import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

# Configuration
DEFAULT_DB_URL = 'postgresql://postgres:postgres@localhost:5432/openhands'


# Sample data
FIRST_NAMES = [
    'Sarah',
    'Michael',
    'Emma',
    'James',
    'Olivia',
    'William',
    'Sophia',
    'Benjamin',
    'Isabella',
    'Lucas',
    'Mia',
    'Henry',
    'Charlotte',
    'Alexander',
    'Amelia',
    'Daniel',
    'Harper',
    'Matthew',
    'Evelyn',
    'Sebastian',
    'Aria',
    'Jack',
    'Luna',
    'Owen',
]

LAST_NAMES = [
    'Chen',
    'Williams',
    'Rodriguez',
    'Kim',
    'Patel',
    'Johnson',
    'Martinez',
    'Anderson',
    'Thompson',
    'Garcia',
    'Lee',
    'Wilson',
    'Taylor',
    'Brown',
    'Davis',
    'Miller',
    'Moore',
    'Jackson',
    'Martin',
    'Thompson',
    'White',
    'Harris',
    'Clark',
    'Lewis',
]

DOMAINS = ['techcorp.io', 'acme.com', 'startupxyz.com', 'enterprise.net', 'devco.org']

LLM_MODELS = [
    'claude-sonnet-4-5',
    'claude-opus-4',
    'gpt-4o',
    'gpt-4-turbo',
    'gemini-1.5-pro',
    'claude-3-5-sonnet',
    'gpt-4o-mini',
    'claude-3-opus',
]

AGENT_KINDS = [
    'CodeAgent',
    'AnalysisAgent',
    'DebugAgent',
    'ReviewAgent',
    'DocumentAgent',
    'TestAgent',
    'RefactorAgent',
]

REPO_NAMES = [
    'frontend-app',
    'backend-api',
    'mobile-app',
    'data-pipeline',
    'ml-service',
    'auth-service',
    'payment-gateway',
    'notification-system',
    'analytics-dashboard',
    'admin-portal',
    'e-commerce-platform',
    'inventory-management',
]

BRANCHES = [
    'main',
    'develop',
    'feature/user-auth',
    'bugfix/login-issue',
    'release/v2.1',
]

TRIGGERS = ['manual', 'scheduled', 'webhook', 'api', 'cli']

CONVERSATION_TITLES = [
    'Fix authentication flow',
    'Implement new dashboard features',
    'Code review for PR #234',
    'Debug memory leak in service',
    'Add unit tests for auth module',
    'Refactor database queries',
    'Setup CI/CD pipeline',
    'Performance optimization',
    'Security audit fixes',
    'Update API documentation',
    'Migrate to new framework',
    'Add dark mode support',
    'Implement search functionality',
    'Fix responsive layout',
    'Add user analytics',
    'Optimize image loading',
    'Create API endpoints',
    'Write integration tests',
    'Fix CSS bugs',
    'Deploy to staging',
]

EXECUTIONS_STATUSES = ['running', 'idle', 'paused', 'finished', 'error', 'stuck']
SANDBOX_STATUSES = ['RUNNING', 'STARTING', 'PAUSED', 'ERROR', 'MISSING']


def random_email(first_name: str, last_name: str) -> str:
    """Generate a random email address."""
    domain = random.choice(DOMAINS)
    return f'{first_name.lower()}.{last_name.lower()}@{domain}'


def random_datetime(days_back: int = 90) -> datetime:
    """Generate a random datetime within the last N days."""
    now = datetime.now(UTC)
    random_days = random.uniform(0, days_back)
    return now - timedelta(days=random_days)


def generate_conversation_data(
    conversation_id: str,
    org_id: str,
    user_id: str,
    created_at: datetime,
) -> dict:
    """Generate realistic conversation metadata."""
    updated_at = created_at + timedelta(
        minutes=random.randint(5, 480), hours=random.randint(0, 72)
    )

    # Ensure updated_at is in the past
    if updated_at > datetime.now(UTC):
        updated_at = datetime.now(UTC) - timedelta(hours=random.randint(1, 24))

    execution_status = random.choice(EXECUTIONS_STATUSES)
    sandbox_status = random.choice(SANDBOX_STATUSES)

    # Running conversations should have RUNNING sandbox
    if execution_status == 'running':
        sandbox_status = 'RUNNING'

    # Completed conversations typically have finished status
    if random.random() < 0.7 and execution_status in ['finished', 'error', 'stuck']:
        sandbox_status = 'MISSING'

    prompt_tokens = random.randint(1000, 50000)
    completion_tokens = random.randint(500, 25000)
    cache_read_tokens = random.randint(0, 10000)
    cache_write_tokens = random.randint(0, 5000)

    # Rough cost estimation (varies by model)
    cost_per_1k_prompt = random.uniform(0.001, 0.015)
    cost_per_1k_completion = random.uniform(0.003, 0.075)
    accumulated_cost = (prompt_tokens / 1000) * cost_per_1k_prompt + (
        completion_tokens / 1000
    ) * cost_per_1k_completion

    return {
        'conversation_id': conversation_id,
        'conversation_version': 'V1',
        'title': random.choice(CONVERSATION_TITLES),
        'llm_model': random.choice(LLM_MODELS),
        'agent_kind': random.choice(AGENT_KINDS),
        'user_id': user_id,
        'created_at': created_at,
        'last_updated_at': updated_at,
        'sandbox_id': f'sb-{uuid.uuid4().hex[:12]}',
        'sandbox_status': sandbox_status,
        'runtime_url': f'https://runtime-{uuid.uuid4().hex[:8]}.example.com'
        if sandbox_status == 'RUNNING'
        else None,
        'execution_status': execution_status,
        'selected_repository': random.choice(REPO_NAMES),
        'selected_branch': random.choice(BRANCHES),
        'trigger': random.choice(TRIGGERS),
        'accumulated_cost': round(accumulated_cost, 4),
        'prompt_tokens': prompt_tokens,
        'completion_tokens': completion_tokens,
        'total_tokens': prompt_tokens + completion_tokens,
        'cache_read_tokens': cache_read_tokens,
        'cache_write_tokens': cache_write_tokens,
        'org_id': org_id,
    }


def create_tables_if_not_exist(engine) -> None:
    """Create the required tables if they don't exist."""
    with engine.connect() as conn:
        # Check if tables exist
        result = conn.execute(
            text("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_name = 'conversation_metadata'
            );
        """)
        )
        tables_exist = result.scalar()

        if not tables_exist:
            print("Tables don't exist. Running migrations...")
            # Run migrations - this assumes alembic is available
            import subprocess

            result = subprocess.run(
                ['python', '-m', 'alembic', 'upgrade', 'head'],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                print(f'Migration failed: {result.stderr}')
                raise Exception('Failed to run migrations')
            print('Migrations complete.')
        else:
            print('Tables already exist.')


def seed_data(
    db_url: str,
    org_count: int = 3,
    conversations_per_org: int = 30,
    users_per_org: int = 10,
) -> None:
    """Seed the database with conversation data.

    Uses the correct OpenHands schema: org, user, org_member tables.
    """

    engine = create_engine(db_url)
    create_tables_if_not_exist(engine)

    SessionLocal = sessionmaker(bind=engine)
    session: Session = SessionLocal()

    try:
        with session.begin():
            # Create orgs
            org_ids = []
            print(f'Creating {org_count} orgs...')
            for i in range(org_count):
                org_id = str(uuid.uuid4())
                org_ids.append(org_id)

                session.execute(
                    text("""
                        INSERT INTO org (id, name, created_at, updated_at, is_github_org, org_type)
                        VALUES (:id, :name, :created_at, :updated_at, false, 'team')
                        ON CONFLICT (id) DO NOTHING
                    """),
                    {
                        'id': org_id,
                        'name': f'Test Org {i + 1}',
                        'created_at': datetime.now(UTC),
                        'updated_at': datetime.now(UTC),
                    },
                )

            print(f'Created {len(org_ids)} orgs')

            # Create users and conversations for each org
            total_conversations = 0

            for org_id in org_ids:
                print(f'Creating users and conversations for org {org_id}...')

                user_ids = []
                for j in range(users_per_org):
                    user_id = str(uuid.uuid4())
                    user_ids.append(user_id)
                    first_name = random.choice(FIRST_NAMES)
                    last_name = random.choice(LAST_NAMES)
                    email = random_email(first_name, last_name)

                    # First user is the owner, next 2 are admins, rest are members
                    role_id = (
                        1 if j == 0 else (2 if j < 3 else 3)
                    )  # owner, admin, member

                    session.execute(
                        text("""
                            INSERT INTO "user" (id, current_org_id, role_id, email, created_at, updated_at)
                            VALUES (:id, :current_org_id, :role_id, :email, :created_at, :updated_at)
                            ON CONFLICT (id) DO NOTHING
                        """),
                        {
                            'id': user_id,
                            'current_org_id': org_id,
                            'role_id': role_id,
                            'email': email,
                            'created_at': datetime.now(UTC),
                            'updated_at': datetime.now(UTC),
                        },
                    )

                    # Add user to org_member
                    session.execute(
                        text("""
                            INSERT INTO org_member (org_id, user_id, role_id, _llm_api_key, status, agent_settings_diff, conversation_settings_diff, has_custom_llm_api_key, created_at, updated_at)
                            VALUES (:org_id, :user_id, :role_id, '', 'active', '{}', '{}', false, :created_at, :updated_at)
                            ON CONFLICT (org_id, user_id) DO NOTHING
                        """),
                        {
                            'org_id': org_id,
                            'user_id': user_id,
                            'role_id': role_id,
                            'created_at': datetime.now(UTC),
                            'updated_at': datetime.now(UTC),
                        },
                    )

                print(f'  Created {len(user_ids)} users')

                # Create conversations for this org
                for k in range(conversations_per_org):
                    conversation_id = str(uuid.uuid4())
                    user_id = random.choice(user_ids)
                    created_at = random_datetime(days_back=90)

                    conv_data = generate_conversation_data(
                        conversation_id=conversation_id,
                        org_id=org_id,
                        user_id=user_id,
                        created_at=created_at,
                    )

                    # Insert into conversation_metadata
                    session.execute(
                        text("""
                            INSERT INTO conversation_metadata (
                                conversation_id, conversation_version, title, llm_model,
                                agent_kind, user_id, created_at, last_updated_at,
                                sandbox_id, sandbox_status, runtime_url, execution_status,
                                selected_repository, selected_branch, trigger,
                                accumulated_cost, prompt_tokens, completion_tokens,
                                total_tokens, cache_read_tokens, cache_write_tokens
                            ) VALUES (
                                :conversation_id, :conversation_version, :title, :llm_model,
                                :agent_kind, :user_id, :created_at, :last_updated_at,
                                :sandbox_id, :sandbox_status, :runtime_url, :execution_status,
                                :selected_repository, :selected_branch, :trigger,
                                :accumulated_cost, :prompt_tokens, :completion_tokens,
                                :total_tokens, :cache_read_tokens, :cache_write_tokens
                            )
                            ON CONFLICT (conversation_id) DO NOTHING
                        """),
                        conv_data,
                    )

                    # Insert into conversation_metadata_saas
                    session.execute(
                        text("""
                            INSERT INTO conversation_metadata_saas (conversation_id, org_id)
                            VALUES (:conversation_id, :org_id)
                            ON CONFLICT (conversation_id) DO NOTHING
                        """),
                        {
                            'conversation_id': conversation_id,
                            'org_id': org_id,
                        },
                    )

                total_conversations += conversations_per_org
                print(f'  Created {conversations_per_org} conversations')

        print('\n✅ Seed complete!')
        print(f'   Organizations: {org_count}')
        print(f'   Total users: {org_count * users_per_org}')
        print(f'   Total conversations: {total_conversations}')
        print(f'\nDatabase: {db_url}')

    except Exception as e:
        session.rollback()
        print(f'❌ Error seeding data: {e}')
        raise
    finally:
        session.close()


def main():
    parser = argparse.ArgumentParser(
        description='Seed database with realistic conversation data for testing.'
    )
    parser.add_argument(
        '--db-url',
        type=str,
        default=None,
        help=f'Database URL (default: DATABASE_URL env var or {DEFAULT_DB_URL})',
    )
    parser.add_argument(
        '--org-count',
        type=int,
        default=3,
        help='Number of organizations to create (default: 3)',
    )
    parser.add_argument(
        '--conversations-per-org',
        type=int,
        default=30,
        help='Number of conversations per organization (default: 30)',
    )
    parser.add_argument(
        '--users-per-org',
        type=int,
        default=10,
        help='Number of users per organization (default: 10)',
    )

    args = parser.parse_args()

    db_url = args.db_url or DEFAULT_DB_URL
    if not db_url:
        print(
            'Error: No database URL provided. Set DATABASE_URL env var or use --db-url'
        )
        return 1

    seed_data(
        db_url=db_url,
        org_count=args.org_count,
        conversations_per_org=args.conversations_per_org,
        users_per_org=args.users_per_org,
    )

    return 0


if __name__ == '__main__':
    exit(main())
