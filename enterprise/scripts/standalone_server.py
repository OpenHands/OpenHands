#!/usr/bin/env python3
"""
Standalone server to serve the admin dashboard API for testing.
Runs the conversation stats API with data from the seeded PostgreSQL database.

Usage:
    python enterprise/scripts/standalone_server.py
"""

import argparse
from datetime import UTC, datetime, timedelta
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from starlette.responses import HTMLResponse

DEFAULT_DB_URL = 'postgresql://postgres:postgres@localhost:5432/openhands'
DEFAULT_PORT = 8080


# Pydantic models
class ConversationStats(BaseModel):
    active_conversations: int
    running_runtimes: int
    completed_24h: int
    completed_7d: int
    completed_30d: int
    total_cost: float
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int


class Conversation(BaseModel):
    conversation_id: str
    title: str
    user_id: str
    user_name: Optional[str] = None
    user_email: Optional[str] = None
    llm_model: str
    agent_kind: str
    status: str
    sandbox_status: str
    created_at: str
    last_updated_at: str
    selected_repository: str
    selected_branch: str
    trigger: str
    accumulated_cost: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ConversationsResponse(BaseModel):
    conversations: list[Conversation]
    total: int
    page: int
    page_size: int
    total_pages: int


# Create FastAPI app
app = FastAPI(title='OpenHands Admin API', version='1.0.0')

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# Global engine and session
_engine = None
_SessionLocal = None


def get_engine(db_url: str):
    global _engine, _SessionLocal
    if _engine is None:
        _engine = create_engine(db_url)
        _SessionLocal = sessionmaker(bind=_engine)
    return _engine, _SessionLocal


@app.get('/')
async def root():
    return {'message': 'OpenHands Admin API', 'docs': '/docs'}


@app.get(
    '/api/organizations/{org_id}/conversations/stats', response_model=ConversationStats
)
async def get_conversation_stats(org_id: str, db_url: str = Query(None)):
    """Get aggregated conversation statistics for an organization."""
    db_url = db_url or DEFAULT_DB_URL
    engine, _ = get_engine(db_url)

    with engine.connect() as conn:
        now = datetime.now(UTC)
        time_24h_ago = now - timedelta(hours=24)
        time_7d_ago = now - timedelta(days=7)
        time_30d_ago = now - timedelta(days=30)

        # Get conversation counts
        stats = conn.execute(
            text("""
                SELECT
                    COUNT(*) FILTER (WHERE cm.execution_status IN ('running', 'idle', 'paused')) as active_conversations,
                    COUNT(*) FILTER (WHERE cm.sandbox_status = 'RUNNING') as running_runtimes,
                    COUNT(*) FILTER (WHERE cm.execution_status = 'finished' AND cm.last_updated_at >= :time_24h) as completed_24h,
                    COUNT(*) FILTER (WHERE cm.execution_status = 'finished' AND cm.last_updated_at >= :time_7d) as completed_7d,
                    COUNT(*) FILTER (WHERE cm.execution_status = 'finished' AND cm.last_updated_at >= :time_30d) as completed_30d,
                    COALESCE(SUM(cm.accumulated_cost), 0) as total_cost,
                    COALESCE(SUM(cm.prompt_tokens), 0) as total_prompt_tokens,
                    COALESCE(SUM(cm.completion_tokens), 0) as total_completion_tokens,
                    COALESCE(SUM(cm.total_tokens), 0) as total_tokens
                FROM conversation_metadata cm
                JOIN conversation_metadata_saas cms ON cm.conversation_id = cms.conversation_id
                WHERE cms.org_id = :org_id
            """),
            {
                'org_id': org_id,
                'time_24h': time_24h_ago,
                'time_7d': time_7d_ago,
                'time_30d': time_30d_ago,
            },
        ).fetchone()

        return ConversationStats(
            active_conversations=stats[0] or 0,
            running_runtimes=stats[1] or 0,
            completed_24h=stats[2] or 0,
            completed_7d=stats[3] or 0,
            completed_30d=stats[4] or 0,
            total_cost=float(stats[5] or 0),
            total_prompt_tokens=int(stats[6] or 0),
            total_completion_tokens=int(stats[7] or 0),
            total_tokens=int(stats[8] or 0),
        )


@app.get(
    '/api/organizations/{org_id}/conversations', response_model=ConversationsResponse
)
async def list_conversations(
    org_id: str,
    status: Optional[str] = None,
    time_window: Optional[str] = None,
    search: Optional[str] = None,
    sort_by: str = 'created_at',
    sort_order: str = 'desc',
    page: int = 1,
    page_size: int = 20,
    db_url: str = Query(None),
):
    """List conversations for an organization with filtering and pagination."""
    db_url = db_url or DEFAULT_DB_URL
    engine, _ = get_engine(db_url)

    with engine.connect() as conn:
        # Build query
        base_query = """
            SELECT
                cm.conversation_id,
                cm.title,
                cm.user_id,
                u.name as user_name,
                u.email as user_email,
                cm.llm_model,
                cm.agent_kind,
                cm.execution_status as status,
                cm.sandbox_status,
                cm.created_at,
                cm.last_updated_at,
                cm.selected_repository,
                cm.selected_branch,
                cm.trigger,
                cm.accumulated_cost,
                cm.prompt_tokens,
                cm.completion_tokens,
                cm.total_tokens
            FROM conversation_metadata cm
            JOIN conversation_metadata_saas cms ON cm.conversation_id = cms.conversation_id
            LEFT JOIN users u ON cm.user_id = u.id
            WHERE cms.org_id = :org_id
        """

        params = {'org_id': org_id}

        # Add filters
        if status and status != 'all':
            if status == 'running':
                base_query += " AND cm.execution_status = 'running'"
            elif status == 'finished':
                base_query += " AND cm.execution_status = 'finished'"
            elif status == 'error':
                base_query += " AND cm.execution_status IN ('error', 'stuck')"

        if time_window:
            now = datetime.now(UTC)
            if time_window == '24h':
                base_query += ' AND cm.created_at >= :time_filter'
                params['time_filter'] = now - timedelta(hours=24)
            elif time_window == '7d':
                base_query += ' AND cm.created_at >= :time_filter'
                params['time_filter'] = now - timedelta(days=7)
            elif time_window == '30d':
                base_query += ' AND cm.created_at >= :time_filter'
                params['time_filter'] = now - timedelta(days=30)

        if search:
            base_query += (
                ' AND (cm.title ILIKE :search OR cm.selected_repository ILIKE :search)'
            )
            params['search'] = f'%{search}%'

        # Count total
        count_query = f'SELECT COUNT(*) FROM ({base_query}) as subquery'
        total = conn.execute(text(count_query), params).scalar()

        # Add sorting and pagination
        valid_sort_columns = [
            'created_at',
            'last_updated_at',
            'title',
            'llm_model',
            'accumulated_cost',
        ]
        if sort_by not in valid_sort_columns:
            sort_by = 'created_at'

        sort_direction = 'DESC' if sort_order.lower() == 'desc' else 'ASC'
        base_query += (
            f' ORDER BY cm.{sort_by} {sort_direction} LIMIT :limit OFFSET :offset'
        )

        params['limit'] = page_size
        params['offset'] = (page - 1) * page_size

        # Execute
        rows = conn.execute(text(base_query), params).fetchall()

        conversations = [
            Conversation(
                conversation_id=row[0],
                title=row[1] or 'Untitled',
                user_id=row[2],
                user_name=row[3],
                user_email=row[4],
                llm_model=row[5] or 'unknown',
                agent_kind=row[6] or 'unknown',
                status=row[7] or 'unknown',
                sandbox_status=row[8] or 'unknown',
                created_at=row[9].isoformat() if row[9] else '',
                last_updated_at=row[10].isoformat() if row[10] else '',
                selected_repository=row[11] or 'unknown',
                selected_branch=row[12] or 'unknown',
                trigger=row[13] or 'unknown',
                accumulated_cost=float(row[14] or 0),
                prompt_tokens=int(row[15] or 0),
                completion_tokens=int(row[16] or 0),
                total_tokens=int(row[17] or 0),
            )
            for row in rows
        ]

        total_pages = (total + page_size - 1) // page_size

        return ConversationsResponse(
            conversations=conversations,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
        )


@app.get('/api/organizations')
async def list_organizations(db_url: str = Query(None)):
    """List all organizations with basic info."""
    db_url = db_url or DEFAULT_DB_URL
    engine, _ = get_engine(db_url)

    with engine.connect() as conn:
        orgs = conn.execute(
            text("""
                SELECT
                    o.id,
                    o.name,
                    COUNT(DISTINCT om.user_id) as member_count,
                    COUNT(cm.conversation_id) as conversation_count
                FROM organizations o
                LEFT JOIN organization_members om ON o.id = om.organization_id
                LEFT JOIN conversation_metadata_saas cms ON o.id = cms.org_id
                LEFT JOIN conversation_metadata cm ON cms.conversation_id = cm.conversation_id
                GROUP BY o.id, o.name
            """)
        ).fetchall()

        return [
            {
                'id': str(row[0]),
                'name': row[1],
                'member_count': row[2],
                'conversation_count': row[3],
            }
            for row in orgs
        ]


@app.get('/api/organizations/{org_id}')
async def get_organization(org_id: str, db_url: str = Query(None)):
    """Get organization details."""
    db_url = db_url or DEFAULT_DB_URL
    engine, _ = get_engine(db_url)

    with engine.connect() as conn:
        org = conn.execute(
            text("""
                SELECT
                    o.id,
                    o.name,
                    COUNT(DISTINCT om.user_id) as member_count,
                    COUNT(DISTINCT CASE WHEN om.role = 'owner' THEN om.user_id END) as owner_count,
                    COUNT(DISTINCT CASE WHEN om.role = 'admin' THEN om.user_id END) as admin_count
                FROM organizations o
                LEFT JOIN organization_members om ON o.id = om.organization_id
                WHERE o.id = :org_id
                GROUP BY o.id, o.name
            """),
            {'org_id': org_id},
        ).fetchone()

        if not org:
            raise HTTPException(status_code=404, detail='Organization not found')

        return {
            'id': str(org[0]),
            'name': org[1],
            'member_count': org[2],
            'owner_count': org[3],
            'admin_count': org[4],
        }


def create_standalone_frontend():
    """Create a simple standalone HTML frontend for the admin dashboard."""
    return r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OpenHands Admin Dashboard</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f0f0f; color: #e0e0e0; min-height: 100vh; }
        .container { max-width: 1400px; margin: 0 auto; padding: 2rem; }
        header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 2rem; padding-bottom: 1rem; border-bottom: 1px solid #333; }
        h1 { font-size: 1.5rem; font-weight: 600; }
        .org-select { background: #1a1a1a; color: #e0e0e0; border: 1px solid #333; padding: 0.5rem 1rem; border-radius: 6px; min-width: 200px; }

        /* Stats Grid */
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 2rem; }
        .stat-card { background: #1a1a1a; border-radius: 12px; padding: 1.5rem; border: 1px solid #333; }
        .stat-card h3 { font-size: 0.875rem; color: #888; margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 0.05em; }
        .stat-card .value { font-size: 2rem; font-weight: 700; color: #fff; }
        .stat-card .sub { font-size: 0.875rem; color: #666; margin-top: 0.25rem; }

        /* Filter Bar */
        .filter-bar { display: flex; gap: 1rem; margin-bottom: 1.5rem; flex-wrap: wrap; align-items: center; }
        .filter-bar input, .filter-bar select { background: #1a1a1a; color: #e0e0e0; border: 1px solid #333; padding: 0.5rem 1rem; border-radius: 6px; }
        .filter-bar input { flex: 1; min-width: 200px; }
        .filter-bar button { background: #3b82f6; color: white; border: none; padding: 0.5rem 1rem; border-radius: 6px; cursor: pointer; }
        .filter-bar button:hover { background: #2563eb; }

        /* Table */
        .table-container { background: #1a1a1a; border-radius: 12px; border: 1px solid #333; overflow: hidden; }
        table { width: 100%; border-collapse: collapse; }
        th { text-align: left; padding: 1rem; background: #252525; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; color: #888; }
        td { padding: 1rem; border-top: 1px solid #333; font-size: 0.875rem; }
        tr:hover td { background: #252525; }
        .status { display: inline-block; padding: 0.25rem 0.5rem; border-radius: 4px; font-size: 0.75rem; font-weight: 500; }
        .status.running { background: #22c55e20; color: #22c55e; }
        .status.finished { background: #3b82f620; color: #3b82f6; }
        .status.error { background: #ef444420; color: #ef4444; }
        .status.idle { background: #f59e0b20; color: #f59e0b; }
        .cost { font-family: monospace; color: #22c55e; }

        /* Pagination */
        .pagination { display: flex; justify-content: center; align-items: center; gap: 0.5rem; margin-top: 1.5rem; }
        .pagination button { background: #1a1a1a; color: #e0e0e0; border: 1px solid #333; padding: 0.5rem 1rem; border-radius: 6px; cursor: pointer; }
        .pagination button:disabled { opacity: 0.5; cursor: not-allowed; }
        .pagination span { color: #888; }

        .loading { text-align: center; padding: 2rem; color: #888; }
        .error { background: #ef444420; color: #ef4444; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🎯 OpenHands Admin Dashboard</h1>
            <select id="orgSelect" class="org-select">
                <option value="">Select Organization</option>
            </select>
        </header>

        <div id="error" class="error" style="display: none;"></div>

        <div class="stats-grid" id="stats">
            <div class="stat-card">
                <h3>Active Conversations</h3>
                <div class="value" id="activeConversations">-</div>
            </div>
            <div class="stat-card">
                <h3>Running Runtimes</h3>
                <div class="value" id="runningRuntimes">-</div>
            </div>
            <div class="stat-card">
                <h3>Completed (24H)</h3>
                <div class="value" id="completed24h">-</div>
            </div>
            <div class="stat-card">
                <h3>Completed (7D)</h3>
                <div class="value" id="completed7d">-</div>
            </div>
            <div class="stat-card">
                <h3>Total Cost</h3>
                <div class="value" id="totalCost">-</div>
                <div class="sub" id="totalTokens">- tokens</div>
            </div>
        </div>

        <div class="filter-bar">
            <input type="text" id="searchInput" placeholder="Search conversations...">
            <select id="statusFilter">
                <option value="all">All Status</option>
                <option value="running">Running</option>
                <option value="finished">Finished</option>
                <option value="error">Error</option>
            </select>
            <select id="timeWindow">
                <option value="">All Time</option>
                <option value="24h">Last 24 Hours</option>
                <option value="7d">Last 7 Days</option>
                <option value="30d">Last 30 Days</option>
            </select>
            <select id="sortBy">
                <option value="created_at">Sort: Created</option>
                <option value="last_updated_at">Sort: Updated</option>
                <option value="title">Sort: Title</option>
                <option value="accumulated_cost">Sort: Cost</option>
            </select>
            <button onclick="loadConversations(1)">Apply Filters</button>
        </div>

        <div class="table-container">
            <table>
                <thead>
                    <tr>
                        <th>Title</th>
                        <th>User</th>
                        <th>Model</th>
                        <th>Status</th>
                        <th>Created</th>
                        <th>Cost</th>
                        <th>Tokens</th>
                    </tr>
                </thead>
                <tbody id="conversationsTable">
                    <tr><td colspan="7" class="loading">Select an organization to view conversations</td></tr>
                </tbody>
            </table>
        </div>

        <div class="pagination" id="pagination"></div>
    </div>

    <script>
        const API_BASE = window.location.origin;
        let currentOrg = null;
        let currentPage = 1;

        async function loadOrganizations() {
            try {
                const res = await fetch(API_BASE + '/api/organizations');
                const orgs = await res.json();
                const select = document.getElementById('orgSelect');
                select.innerHTML = '<option value="">Select Organization</option>';
                orgs.forEach(org => {
                    select.innerHTML += `<option value="${org.id}">${org.name} (${org.conversation_count} convos)</option>`;
                });
            } catch (e) {
                showError('Failed to load organizations: ' + e.message);
            }
        }

        async function loadStats(orgId) {
            try {
                const res = await fetch(API_BASE + '/api/organizations/' + orgId + '/conversations/stats');
                const stats = await res.json();
                document.getElementById('activeConversations').textContent = stats.active_conversations;
                document.getElementById('runningRuntimes').textContent = stats.running_runtimes;
                document.getElementById('completed24h').textContent = stats.completed_24h;
                document.getElementById('completed7d').textContent = stats.completed_7d;
                document.getElementById('totalCost').textContent = '$' + stats.total_cost.toFixed(2);
                document.getElementById('totalTokens').textContent = formatNumber(stats.total_tokens) + ' tokens';
            } catch (e) {
                showError('Failed to load stats: ' + e.message);
            }
        }

        async function loadConversations(page = 1) {
            if (!currentOrg) return;
            currentPage = page;

            const search = document.getElementById('searchInput').value;
            const status = document.getElementById('statusFilter').value;
            const timeWindow = document.getElementById('timeWindow').value;
            const sortBy = document.getElementById('sortBy').value;

            const params = new URLSearchParams({ page, sort_by: sortBy });
            if (search) params.append('search', search);
            if (status !== 'all') params.append('status', status);
            if (timeWindow) params.append('time_window', timeWindow);

            try {
                const res = await fetch(API_BASE + '/api/organizations/' + currentOrg + '/conversations?' + params);
                const data = await res.json();
                renderTable(data.conversations);
                renderPagination(data);
            } catch (e) {
                showError('Failed to load conversations: ' + e.message);
            }
        }

        function renderTable(conversations) {
            const tbody = document.getElementById('conversationsTable');
            if (conversations.length === 0) {
                tbody.innerHTML = '<tr><td colspan="7" class="loading">No conversations found</td></tr>';
                return;
            }
            tbody.innerHTML = conversations.map(c => {
                const statusClass = c.status === 'running' ? 'running' : c.status === 'finished' ? 'finished' : 'error';
                const date = new Date(c.created_at).toLocaleDateString();
                return \`
                    <tr>
                        <td>\${c.title}</td>
                        <td>\${c.user_name || c.user_email || 'Unknown'}</td>
                        <td>\${c.llm_model}</td>
                        <td><span class="status \${statusClass}">\${c.status}</span></td>
                        <td>\${date}</td>
                        <td class="cost">$\${c.accumulated_cost.toFixed(4)}</td>
                        <td>\${formatNumber(c.total_tokens)}</td>
                    </tr>
                \`;
            }).join('');
        }

        function renderPagination(data) {
            const div = document.getElementById('pagination');
            let html = \`<button onclick="loadConversations(\${data.page - 1})" \${data.page <= 1 ? 'disabled' : ''}>Previous</button>\`;
            html += \`<span>Page \${data.page} of \${data.total_pages}</span>\`;
            html += \`<button onclick="loadConversations(\${data.page + 1})" \${data.page >= data.total_pages ? 'disabled' : ''}>Next</button>\`;
            div.innerHTML = html;
        }

        function formatNumber(n) {
            if (n >= 1000000) return (n / 1000000).toFixed(1) + 'M';
            if (n >= 1000) return (n / 1000).toFixed(1) + 'K';
            return n.toString();
        }

        function showError(msg) {
            document.getElementById('error').textContent = msg;
            document.getElementById('error').style.display = 'block';
        }

        document.getElementById('orgSelect').addEventListener('change', async (e) => {
            currentOrg = e.target.value;
            if (currentOrg) {
                await loadStats(currentOrg);
                await loadConversations(1);
            }
        });

        // Initialize
        loadOrganizations();
    </script>
</body>
</html>
"""


# Add static file route for the frontend
@app.get('/dashboard')
async def dashboard():
    return HTMLResponse(create_standalone_frontend())


def main():
    parser = argparse.ArgumentParser(description='Run standalone admin API server')
    parser.add_argument('--port', type=int, default=DEFAULT_PORT, help='Port to run on')
    parser.add_argument('--db-url', type=str, default=None, help='Database URL')
    args = parser.parse_args()

    db_url = args.db_url or DEFAULT_DB_URL
    print(f'Starting server on port {args.port}')
    print(f'Database: {db_url}')
    print(f'API Docs: http://localhost:{args.port}/docs')
    print(f'Dashboard: http://localhost:{args.port}/dashboard')

    uvicorn.run(app, host='0.0.0.0', port=args.port)


if __name__ == '__main__':
    main()
