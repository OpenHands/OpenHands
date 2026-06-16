#!/usr/bin/env python3
"""
Standalone script to create tables and seed conversation data.
This bypasses the full enterprise migration system for simpler local testing.

Usage:
    python enterprise/scripts/setup_and_seed.py
"""

import argparse
import random
import uuid
from datetime import datetime, timedelta, UTC

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

DEFAULT_DB_URL = "postgresql://postgres:postgres@localhost:5432/openhands"


# Sample data
FIRST_NAMES = [
    "Sarah", "Michael", "Emma", "James", "Olivia", "William", "Sophia", "Benjamin",
    "Isabella", "Lucas", "Mia", "Henry", "Charlotte", "Alexander", "Amelia", "Daniel",
    "Harper", "Matthew", "Evelyn", "Sebastian", "Aria", "Jack", "Luna", "Owen"
]

LAST_NAMES = [
    "Chen", "Williams", "Rodriguez", "Kim", "Patel", "Johnson", "Martinez", "Anderson",
    "Thompson", "Garcia", "Lee", "Wilson", "Taylor", "Brown", "Davis", "Miller",
    "Moore", "Jackson", "Martin", "White", "Harris", "Clark", "Lewis"
]

DOMAINS = ["techcorp.io", "acme.com", "startupxyz.com", "enterprise.net", "devco.org"]

LLM_MODELS = [
    "claude-sonnet-4-5", "claude-opus-4", "gpt-4o", "gpt-4-turbo",
    "gemini-1.5-pro", "claude-3-5-sonnet", "gpt-4o-mini", "claude-3-opus"
]

AGENT_KINDS = [
    "CodeAgent", "AnalysisAgent", "DebugAgent", "ReviewAgent",
    "DocumentAgent", "TestAgent", "RefactorAgent"
]

REPO_NAMES = [
    "frontend-app", "backend-api", "mobile-app", "data-pipeline",
    "ml-service", "auth-service", "payment-gateway", "notification-system",
    "analytics-dashboard", "admin-portal", "e-commerce-platform", "inventory-management"
]

BRANCHES = ["main", "develop", "feature/user-auth", "bugfix/login-issue", "release/v2.1"]

TRIGGERS = ["manual", "scheduled", "webhook", "api", "cli"]

CONVERSATION_TITLES = [
    "Fix authentication flow", "Implement new dashboard features",
    "Code review for PR #234", "Debug memory leak in service",
    "Add unit tests for auth module", "Refactor database queries",
    "Setup CI/CD pipeline", "Performance optimization",
    "Security audit fixes", "Update API documentation",
    "Migrate to new framework", "Add dark mode support",
    "Implement search functionality", "Fix responsive layout",
    "Add user analytics", "Optimize image loading",
    "Create API endpoints", "Write integration tests",
    "Fix CSS bugs", "Deploy to staging"
]

EXECUTIONS_STATUSES = ["running", "idle", "paused", "finished", "error", "stuck"]
SANDBOX_STATUSES = ["RUNNING", "STARTING", "PAUSED", "ERROR", "MISSING"]


def random_email(first_name: str, last_name: str) -> str:
    domain = random.choice(DOMAINS)
    return f"{first_name.lower()}.{last_name.lower()}@{domain}"


def random_datetime(days_back: int = 90) -> datetime:
    now = datetime.now(UTC)
    return now - timedelta(days=random.uniform(0, days_back))


def create_tables(engine):
    """Create required tables if they don't exist."""
    print("Creating tables...")
    
    with engine.connect() as conn:
        # Check if organizations table exists
        result = conn.execute(text("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = 'organizations'
            );
        """))
        tables_exist = result.scalar()
        
        if tables_exist:
            print("Tables already exist, skipping creation.")
            return
        
        # Create organizations table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS organizations (
                id UUID PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE NOT NULL,
                updated_at TIMESTAMP WITH TIME ZONE NOT NULL,
                is_github_org BOOLEAN DEFAULT FALSE,
                org_type VARCHAR(50) DEFAULT 'team',
                github_installation_id INTEGER,
                gitlab_group_id INTEGER
            );
        """))
        
        # Create users table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS users (
                id UUID PRIMARY KEY,
                email VARCHAR(255) UNIQUE NOT NULL,
                name VARCHAR(255),
                avatar_url TEXT,
                created_at TIMESTAMP WITH TIME ZONE NOT NULL,
                updated_at TIMESTAMP WITH TIME ZONE NOT NULL
            );
        """))
        
        # Create organization_members table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS organization_members (
                organization_id UUID NOT NULL,
                user_id UUID NOT NULL,
                role VARCHAR(50) NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE NOT NULL,
                updated_at TIMESTAMP WITH TIME ZONE NOT NULL,
                PRIMARY KEY (organization_id, user_id)
            );
        """))
        
        # Create conversation_metadata table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS conversation_metadata (
                conversation_id VARCHAR(255) PRIMARY KEY,
                conversation_version VARCHAR(50) NOT NULL,
                title TEXT,
                llm_model VARCHAR(255),
                agent_kind VARCHAR(255),
                user_id VARCHAR(255) NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE NOT NULL,
                last_updated_at TIMESTAMP WITH TIME ZONE NOT NULL,
                sandbox_id VARCHAR(255),
                sandbox_status VARCHAR(50),
                runtime_url TEXT,
                execution_status VARCHAR(50),
                selected_repository TEXT,
                selected_branch TEXT,
                trigger VARCHAR(50),
                accumulated_cost DECIMAL(10, 6) DEFAULT 0,
                prompt_tokens INTEGER DEFAULT 0,
                completion_tokens INTEGER DEFAULT 0,
                total_tokens INTEGER DEFAULT 0,
                cache_read_tokens INTEGER DEFAULT 0,
                cache_write_tokens INTEGER DEFAULT 0
            );
        """))
        
        # Create conversation_metadata_saas table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS conversation_metadata_saas (
                conversation_id VARCHAR(255) PRIMARY KEY,
                org_id UUID NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversation_metadata(conversation_id) ON DELETE CASCADE
            );
        """))
        
        conn.commit()
        print("Tables created successfully!")


def seed_data(
    engine,
    org_count: int = 3,
    conversations_per_org: int = 30,
    users_per_org: int = 10,
):
    """Seed the database with conversation data."""
    
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    
    try:
        with session.begin():
            # Create organizations
            org_ids = []
            print(f"\nCreating {org_count} organizations...")
            for i in range(org_count):
                org_id = str(uuid.uuid4())
                org_ids.append(org_id)
                
                session.execute(
                    text("""
                        INSERT INTO organizations (id, name, created_at, updated_at, is_github_org, org_type)
                        VALUES (:id, :name, :created_at, :updated_at, false, 'team')
                        ON CONFLICT (id) DO NOTHING
                    """),
                    {
                        "id": org_id,
                        "name": f"Test Org {i + 1}",
                        "created_at": datetime.now(UTC),
                        "updated_at": datetime.now(UTC),
                    }
                )
            
            print(f"Created {len(org_ids)} organizations")
            
            # Create users and conversations for each org
            total_conversations = 0
            
            for org_id in org_ids:
                print(f"\nCreating users and conversations for org...")
                
                user_ids = []
                for j in range(users_per_org):
                    user_id = str(uuid.uuid4())
                    user_ids.append(user_id)
                    first_name = random.choice(FIRST_NAMES)
                    last_name = random.choice(LAST_NAMES)
                    email = random_email(first_name, last_name)
                    
                    session.execute(
                        text("""
                            INSERT INTO users (id, email, name, avatar_url, created_at, updated_at)
                            VALUES (:id, :email, :name, :avatar_url, :created_at, :updated_at)
                            ON CONFLICT (id) DO NOTHING
                        """),
                        {
                            "id": user_id,
                            "email": email,
                            "name": f"{first_name} {last_name}",
                            "avatar_url": None,
                            "created_at": datetime.now(UTC),
                            "updated_at": datetime.now(UTC),
                        }
                    )
                    
                    # Add user to organization
                    role = "owner" if j == 0 else ("admin" if j < 3 else "member")
                    session.execute(
                        text("""
                            INSERT INTO organization_members (organization_id, user_id, role, created_at, updated_at)
                            VALUES (:org_id, :user_id, :role, :created_at, :updated_at)
                            ON CONFLICT (organization_id, user_id) DO NOTHING
                        """),
                        {
                            "org_id": org_id,
                            "user_id": user_id,
                            "role": role,
                            "created_at": datetime.now(UTC),
                            "updated_at": datetime.now(UTC),
                        }
                    )
                
                print(f"  Created {len(user_ids)} users")
                
                # Create conversations for this org
                for k in range(conversations_per_org):
                    conversation_id = str(uuid.uuid4())
                    user_id = random.choice(user_ids)
                    created_at = random_datetime(days_back=90)
                    
                    # Ensure updated_at is after created_at
                    updated_at = created_at + timedelta(
                        minutes=random.randint(5, 480),
                        hours=random.randint(0, 72)
                    )
                    if updated_at > datetime.now(UTC):
                        updated_at = datetime.now(UTC) - timedelta(hours=random.randint(1, 24))
                    
                    execution_status = random.choice(EXECUTIONS_STATUSES)
                    sandbox_status = random.choice(SANDBOX_STATUSES)
                    
                    if execution_status == "running":
                        sandbox_status = "RUNNING"
                    
                    if random.random() < 0.7 and execution_status in ["finished", "error", "stuck"]:
                        sandbox_status = "MISSING"
                    
                    prompt_tokens = random.randint(1000, 50000)
                    completion_tokens = random.randint(500, 25000)
                    cache_read_tokens = random.randint(0, 10000)
                    cache_write_tokens = random.randint(0, 5000)
                    
                    cost_per_1k_prompt = random.uniform(0.001, 0.015)
                    cost_per_1k_completion = random.uniform(0.003, 0.075)
                    accumulated_cost = (
                        (prompt_tokens / 1000) * cost_per_1k_prompt +
                        (completion_tokens / 1000) * cost_per_1k_completion
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
                        {
                            "conversation_id": conversation_id,
                            "conversation_version": "V1",
                            "title": random.choice(CONVERSATION_TITLES),
                            "llm_model": random.choice(LLM_MODELS),
                            "agent_kind": random.choice(AGENT_KINDS),
                            "user_id": user_id,
                            "created_at": created_at,
                            "last_updated_at": updated_at,
                            "sandbox_id": f"sb-{uuid.uuid4().hex[:12]}" if random.random() > 0.1 else None,
                            "sandbox_status": sandbox_status,
                            "runtime_url": f"https://runtime-{uuid.uuid4().hex[:8]}.example.com" if sandbox_status == "RUNNING" else None,
                            "execution_status": execution_status,
                            "selected_repository": random.choice(REPO_NAMES),
                            "selected_branch": random.choice(BRANCHES),
                            "trigger": random.choice(TRIGGERS),
                            "accumulated_cost": round(accumulated_cost, 4),
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": prompt_tokens + completion_tokens,
                            "cache_read_tokens": cache_read_tokens,
                            "cache_write_tokens": cache_write_tokens,
                        }
                    )
                    
                    # Insert into conversation_metadata_saas
                    session.execute(
                        text("""
                            INSERT INTO conversation_metadata_saas (conversation_id, org_id)
                            VALUES (:conversation_id, :org_id)
                            ON CONFLICT (conversation_id) DO NOTHING
                        """),
                        {
                            "conversation_id": conversation_id,
                            "org_id": org_id,
                        }
                    )
                
                total_conversations += conversations_per_org
                print(f"  Created {conversations_per_org} conversations")
        
        print(f"\n✅ Seed complete!")
        print(f"   Organizations: {org_count}")
        print(f"   Total users: {org_count * users_per_org}")
        print(f"   Total conversations: {total_conversations}")
        
    except Exception as e:
        session.rollback()
        print(f"❌ Error seeding data: {e}")
        raise
    finally:
        session.close()


def main():
    parser = argparse.ArgumentParser(
        description="Create tables and seed conversation data for testing."
    )
    parser.add_argument(
        "--db-url",
        type=str,
        default=None,
        help=f"Database URL (default: DATABASE_URL env var or {DEFAULT_DB_URL})"
    )
    parser.add_argument(
        "--org-count",
        type=int,
        default=3,
        help="Number of organizations to create (default: 3)"
    )
    parser.add_argument(
        "--conversations-per-org",
        type=int,
        default=30,
        help="Number of conversations per organization (default: 30)"
    )
    parser.add_argument(
        "--users-per-org",
        type=int,
        default=10,
        help="Number of users per organization (default: 10)"
    )
    
    args = parser.parse_args()
    
    db_url = args.db_url or DEFAULT_DB_URL
    print(f"Using database: {db_url}")
    
    engine = create_engine(db_url)
    
    # Test connection
    try:
        with engine.connect() as conn:
            print("✅ Connected to database")
    except Exception as e:
        print(f"❌ Could not connect to database: {e}")
        return 1
    
    # Create tables
    create_tables(engine)
    
    # Seed data
    seed_data(
        engine=engine,
        org_count=args.org_count,
        conversations_per_org=args.conversations_per_org,
        users_per_org=args.users_per_org,
    )
    
    print(f"\n📊 Database ready at: {db_url}")
    
    return 0


if __name__ == "__main__":
    exit(main())
