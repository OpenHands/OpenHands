# DB Pool / Webhook Callback Validation

## Scope

This artifact records the evidence for a customer-reported OHE database pool
exhaustion issue. SaaS validation and feature-branch soak testing were
intentionally deferred.

Implementation branch: `alona/db-pool-webhook-fix`

Starting code commit: `6b0453254`

## Fixes

1. The webhook route's event and conversation-info dependencies now use
   FastAPI function scope, returning the request session before Starlette
   response background tasks execute.
2. `SetTitleCallbackProcessor` reads the conversation in a short DB context,
   closes it, performs title polling with no DB session held, and opens fresh
   short contexts only when it has data to write.
3. Concurrent title polls are deduplicated per conversation and app-server
   worker. The guard is always released in `finally`.
4. The title write reloads current conversation metadata after polling instead
   of saving a potentially stale pre-poll snapshot.

## Deterministic local regressions

Before the fix:

```text
title poll boundary: assert 3 open DB-backed contexts == 0
webhook dependency scope: assert None == "function"
concurrent title poll: second invocation timed out instead of deduplicating
```

After the fix:

```text
53 passed
pre-commit: ruff, ruff-format, mypy, and repository checks passed
```

Covered files:

- `tests/unit/app_server/test_set_title_callback_processor.py`
- `tests/unit/app_server/test_webhook_router_auth.py`
- `tests/unit/app_server/test_webhook_router_auto_title.py`
- `tests/unit/app_server/test_sql_event_callback_service.py`

## R02 controlled reproduction

R02 used the same affected application source as the customer's reported
release. A real UI-created OHE conversation was left unprompted so the runtime
title remained `null` and its `SetTitleCallbackProcessor` remained `ACTIVE`.
Traffic used the real authenticated runtime webhook endpoint and SDK-validated
`MessageEvent` JSON.

Load profile:

```text
400 events
20 events/second for 20 seconds
3 app-server workers
DB_POOL_SIZE=5
DB_MAX_OVERFLOW=10
PostgreSQL max_connections=79
PostgreSQL shared_buffers=184648kB
```

### Before / after

| Metric | Before | Fixed |
|---|---:|---:|
| HTTP 200 | 355 | 400 |
| HTTP 500 | 41 | 0 |
| QueuePool timeout log lines | 77 | 0 |
| Failures in sandbox auth path | 41 | 0 |
| Callback checkout failure lines | 14 | 0 |
| Request p50 | 9.472s | 0.247s |
| Request p95 | 30.058s | 1.077s |
| Request p99 | 31.043s | 1.201s |
| Max PostgreSQL sessions | 63 | 33 |
| Max active PostgreSQL sessions | 13 | 1 |
| Max idle in transaction | 46 | 1 |
| Health probe failures | 1 | 0 |
| App-server restarts | 0 | 0 |

The before run reproduced the customer's paths:

```text
webhook_router.py:276 valid_sandbox
remote_sandbox_service.py:430 get_sandbox_record_by_session_api_key
sql_event_callback_service.py get_active_callbacks
sqlalchemy.exc.TimeoutError: QueuePool limit of size 5 overflow 10 reached,
connection timed out, timeout 30.00
```

The fixed run logged six completed `title not available` polls over the
20-second load. This is consistent with at most one poll per worker at a time,
plus one retry after the first 12-second poll, instead of one slow poll per
accepted event.

## Customer-equivalent database profile

R02 was temporarily resized to a 16 GiB RDS class and live PostgreSQL values
were verified, not inferred:

```text
SHOW max_connections=300
SHOW shared_buffers=4GB
DB_POOL_SIZE=25
DB_MAX_OVERFLOW=10
starting PostgreSQL sessions=12
```

The same 400-event profile produced:

```text
result_counts={'200': 400}
request_p50_seconds=0.214
request_p95_seconds=0.583
request_p99_seconds=1.270
health_errors=[]
QueuePool timeout lines=0
callback exception lines=0
max PostgreSQL sessions=48
max active PostgreSQL sessions=1
max idle in transaction=3
app-server restarts=0
```

A fresh Replicated UI conversation also received an agent response while this
profile was active.

## Deployment identity

The fixed 5/10 and customer-profile tests mounted the exact worktree source
through a test-only ConfigMap. Local and in-pod SHA-256 hashes matched:

```text
38a179a3f1dabb8e12ae322a4df2e0a12965d3b363d7163321465ad51ac9cf5b  config.py
fa159a155e6eb1a9c6d60c0c7c95c751ea0a7f3c17aecd2fc71129a6f0a0dd67  webhook_router.py
ef53f8fe8fb2fa9faf3d7ffb4d079b2d06b9ccc54bf35325e0ff0610ebb1d337  set_title_callback_processor.py
```

No release was promoted. Validation used a test-only ConfigMap rather than a
published application image.

## Replicated reports

- Initial auth-blocked run:
  `/Users/alonaking/replicated-tests/alona/db-pool-webhook-fix/2026-07-24/db-pool-webhook-instance-2-122032/REPORT.md`
- Successful failing baseline:
  `/Users/alonaking/replicated-tests/alona/db-pool-webhook-fix/2026-07-24/db-pool-baseline-authenticated-instance-2-123255/REPORT.md`
- Fixed 5/10 replay:
  `/Users/alonaking/replicated-tests/alona/db-pool-webhook-fix/2026-07-24/db-pool-fixed-5-10-instance-2-130000/REPORT.md`
- Customer 300/4GB profile:
  `/Users/alonaking/replicated-tests/alona/db-pool-webhook-fix/2026-07-24/customer-db-300-profile-instance-2-130832/REPORT.md`

The successful reports contain UI conversation screenshots.

## Cleanup

R02 was restored after testing:

```text
base application source (test ConfigMap and mounts removed)
DB_POOL_SIZE=25
DB_MAX_OVERFLOW=10
RDS class=db.t4g.micro
RDS parameter group=default.postgres17 (in-sync)
SHOW max_connections=79
SHOW shared_buffers=184648kB
app-server ready, zero restarts
```

The temporary RDS parameter group was deleted.

## Remaining work

- Validate SaaS separately on a feature branch and soak environment.
- Consider KOTS exposure/documentation of pool settings as an operator
  improvement. It is not required for the demonstrated fix and must explain
  the per-worker/per-engine connection budget rather than presenting the
  values as a single global pool.
- For future headless setup, create a dedicated R02 test-user API key through
  the supported OHE API Keys UI and store it in the existing test secret
  mechanism. The persistent named GitLab browser session already prevented
  repeat login for the successful runs.
