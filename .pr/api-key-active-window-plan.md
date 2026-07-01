# Plan: Optional Active Window for API Keys (`not_before` & `expires_at`)

## Background / Current State

- The `api_keys` table already has an `expires_at` column (added in migration `022_create_api_keys_table.py`).
- The backend already stores `expires_at` (`storage/api_key.py`), accepts it on create (`storage/api_key_store.create_api_key`), and rejects expired keys at validation time (`storage/api_key_store.validate_api_key` → `server/auth/saas_user_auth.saas_user_auth_from_bearer`).
- The HTTP route (`server/routes/api_keys.py`) already accepts `expires_at` in `ApiKeyCreate`, returns it in `ApiKeyResponse`, and the `expires_at` field validator rejects past dates.
- The **frontend does not surface `expires_at`** anywhere (the create modal only has a name field, and the table has no expiration column).
- **`not_before` does not exist** in the schema, model, store, route, or UI.

The goal of this change is to:

1. Add a new `not_before` (optional "active from") timestamp to API keys.
2. Expose both `not_before` and `expires_at` in the UI so users can schedule a key's active window.
3. Have authentication honour both ends of that window (key is valid only when `not_before <= now < expires_at`, with each bound independently optional).

The plan is deliberately additive and backward-compatible: every existing key continues to work exactly as before, and the new fields default to `NULL` ("always valid" in their respective direction).

---

## 1. Database migration (enterprise)

**New file:** `enterprise/migrations/versions/128_add_not_before_to_api_keys.py`

- `revision = '128'`, `down_revision = '127'`.
- Inside `upgrade()`:
  - `op.add_column('api_keys', sa.Column('not_before', sa.DateTime(), nullable=True))`
  - `op.create_index('ix_api_keys_not_before', 'api_keys', ['not_before'], unique=False)` (cheap; useful for future "pending keys" dashboards and any cleanup job).
- `downgrade()` drops the index then the column.
- Follow the style of the existing `127_add_execution_status_to_conversation_metadata.py` (use `op.batch_alter_table` to stay consistent with the codebase's Postgres → SQLite-portable pattern).

No data backfill is needed: existing rows get `not_before = NULL` (i.e. immediately valid), preserving current behaviour.

---

## 2. SQLAlchemy model

**File:** `enterprise/storage/api_key.py`

Add the new mapped column next to `expires_at`:

```python
not_before: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
```

No relationship changes required.

---

## 3. `ApiKeyStore` — create + validate

**File:** `enterprise/storage/api_key_store.py`

### 3.1 `create_api_key`

- Add a `not_before: datetime | None = None` keyword argument (placed next to `expires_at` for readability).
- Apply the same naive-UTC stripping helper that `expires_at` already uses (the column is `TIMESTAMP WITHOUT TIME ZONE`, and the auth code re-attaches UTC for comparison).
- Persist the field on the new `ApiKey` record.
- Update the `get_or_create_system_api_key` method to continue to pass `not_before=None` (system keys are always valid from creation).

### 3.2 `validate_api_key` (this is the authentication-time check)

Extend the existing expiration block to a full active-window check:

```python
now = datetime.now(UTC)

# Lower bound: not_before
if key_record.not_before:
    not_before = key_record.not_before
    if not_before.tzinfo is None:
        not_before = not_before.replace(tzinfo=UTC)
    if now < not_before:
        logger.info(f'API key not yet active: {key_record.id}')
        return None

# Upper bound: expires_at
if key_record.expires_at:
    expires_at = key_record.expires_at
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=UTC)
    if expires_at < now:
        logger.info(f'API key has expired: {key_record.id}')
        return None
```

Notes:
- Both bounds are optional, independent, and combined with AND semantics.
- The two `logger.info` lines stay so audit logs continue to distinguish "expired" from the new "not yet active" case (matters for SOC2-style audit trails).
- We intentionally do **not** touch `last_used_at` when the key is rejected (today's code already does this correctly because the update is only reached on success — verified in the existing tests).

### 3.3 Internal helper (optional refactor)

To avoid duplicating the naive-UTC-normalisation logic, extract a small helper:

```python
def _as_utc_aware(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=UTC)
```

and use it in both `create_api_key` and `validate_api_key`. (Skip if you prefer to keep the diff small — it's a stylistic call.)

---

## 4. HTTP route + Pydantic models

**File:** `enterprise/server/routes/api_keys.py`

### 4.1 `ApiKeyCreate`

```python
class ApiKeyCreate(BaseModel):
    name: str | None = None
    not_before: datetime | None = None
    expires_at: datetime | None = None

    @field_validator('not_before')
    def validate_not_before(cls, v):
        if v and v < datetime.now(UTC):
            raise ValueError('not_before cannot be in the past')
        return v

    @field_validator('expires_at')
    def validate_expiration(cls, v):
        if v and v < datetime.now(UTC):
            raise ValueError('Expiration date cannot be in the past')
        return v

    @model_validator(mode='after')
    def validate_window(self):
        if self.not_before and self.expires_at and self.not_before >= self.expires_at:
            raise ValueError('not_before must be earlier than expires_at')
        return self
```

Pydantic v2 is already in use in this file (it imports `field_validator` from pydantic), so `model_validator` is available.

### 4.2 `ApiKeyResponse` & `ApiKeyCreateResponse`

Add `not_before: datetime | None = None` next to `expires_at` in both classes, and update `api_key_to_response` to populate it.

### 4.3 `create_api_key` handler

Pass the new field through:

```python
api_key = await api_key_store.create_api_key(
    user_id,
    key_data.name,
    expires_at=key_data.expires_at,
    not_before=key_data.not_before,
    org_id=effective_org_id,
)
```

The `CurrentApiKeyResponse` does **not** need to be updated — it is a "who am I" endpoint, not a metadata dump.

### 4.4 Error handling

Pydantic `ValueError`s from the validators bubble up as FastAPI `422 Unprocessable Entity` automatically, with the validator's message in the body. No additional handling required.

---

## 5. Authentication flow

**File:** `enterprise/server/auth/saas_user_auth.py`

No code change is required here. The bearer flow already goes:

```
saas_user_auth_from_bearer
  → api_key_store.validate_api_key(api_key)   # ← step 3.2 above
  → SaasUserAuth(... api_key_id, api_key_name, api_key_org_id)
```

Because we extended `validate_api_key` to reject keys outside the active window, the 401/403 path is reached automatically when a key is used too early or too late. The `SaasUserAuth` class does not need to know about `not_before` — it only needs the resolved user/org.

For observability, the existing `logger.info` lines we add in `validate_api_key` will surface in the same log stream that operators already monitor.

---

## 6. Backend tests

### 6.1 `tests/unit/test_api_key_store.py` (extend)

Add unit tests for `not_before` covering:

- `test_create_api_key_strips_timezone_from_not_before` — mirror of the existing `expires_at` test.
- `test_validate_api_key_not_yet_active_rejected` — `not_before` in the future → `validate_api_key` returns `None`.
- `test_validate_api_key_not_yet_active_timezone_naive` — naive UTC `not_before` in the future is also rejected.
- `test_validate_api_key_active_window_with_both_bounds` — key valid only between `not_before` and `expires_at`; rejected before, accepted inside, rejected after.
- `test_validate_api_key_only_not_before` — accepts after the start time, has no upper bound.
- `test_validate_api_key_only_expires_at` — existing behaviour remains unchanged (regression guard).

Use the same `async_session_maker` fixture pattern as the existing tests in this file (SQLite in-memory, per repo guidance in `AGENTS.md`).

### 6.2 `tests/unit/server/routes/test_api_keys.py` (extend)

Add a `TestCreateApiKey` class:

- Successful create with `not_before` and `expires_at` → response includes both fields.
- `not_before` in the past → 422.
- `expires_at` in the past → 422.
- `not_before >= expires_at` → 422.
- Both omitted → key created with `not_before=None`, `expires_at=None` (regression for the existing happy path).

Use `fastapi.testclient.TestClient` against the `api_router` (matching the pattern already in the file). The route pulls `user_id` from `get_user_id` and `effective_org_id` from `EFFECTIVE_ORG_ID`, so the tests will need to `app.dependency_overrides` both — the file already overrides `get_user_id` in some tests, so follow the local pattern.

### 6.3 `tests/unit/server/auth/test_saas_user_auth_effective_org.py` (or new test)

Add a single integration-style test:

- Insert a key with `not_before` in the future, call `saas_user_auth_from_bearer` with a `Request` carrying that key in `Authorization: Bearer …`, and assert the result is `None` (i.e. the key is rejected, not that an `SaasUserAuth` is produced).

---

## 7. Frontend types & API client

**File:** `frontend/src/api/api-keys.ts`

Extend the interfaces (note: the frontend mock uses string IDs and a `prefix`, but the live backend returns integer IDs and no `prefix`; this mismatch already exists in the mock and we should preserve it for mock compatibility):

```ts
export interface ApiKey {
  id: string;
  name: string;
  prefix: string;
  created_at: string;
  last_used_at: string | null;
  not_before: string | null;
  expires_at: string | null;
}

export interface CreateApiKeyResponse {
  id: string;
  name: string;
  key: string;
  prefix: string;
  created_at: string;
  not_before: string | null;
  expires_at: string | null;
}

export interface CreateApiKeyInput {
  name: string;
  not_before?: string | null; // ISO 8601 UTC
  expires_at?: string | null; // ISO 8601 UTC
}
```

Update `ApiKeysClient.createApiKey` to take the new input shape and pass it through.

---

## 8. Frontend mutation hook

**File:** `frontend/src/hooks/mutation/use-create-api-key.ts`

Change the mutation signature to accept the new input shape:

```ts
mutationFn: async (input: CreateApiKeyInput): Promise<CreateApiKeyResponse> =>
  ApiKeysClient.createApiKey(input),
```

`onSuccess` invalidation logic stays the same.

---

## 9. Frontend create modal

**File:** `frontend/src/components/features/settings/create-api-key-modal.tsx`

- Add two new state fields, `notBefore` and `expiresAt`, initialised to empty strings.
- Convert the existing `SettingsInput` (name) to keep using `type="text"`.
- Add two new `SettingsInput` fields with `type="datetime-local"` and `showOptionalTag` (the existing component already supports both — see `frontend/src/components/features/settings/settings-input.tsx`).
- Convert the local datetime string to a UTC ISO string before submitting: `new Date(localValue).toISOString()`; pass `undefined` when the input is empty.
- Submit through the updated mutation hook with the new payload.
- Add inline validation:
  - If `notBefore > expiresAt` and both are set, surface a `displayErrorToast` and do not submit.
  - Mirror the backend's "not in the past" check client-side for faster feedback.
- Reset all three fields on cancel / on successful creation.

The new `SettingsInput` `type="datetime-local"` will give us a native browser picker; we don't need to introduce a new dependency (`@heroui/react` is already in the bundle, and a native picker is the most accessible + lowest-risk option).

---

## 10. Frontend keys table

**File:** `frontend/src/components/features/settings/api-keys-manager.tsx`

- Add a new "Status" or "Active Window" column to the table.
- Compute the per-row status with a small helper:
  - `expired`  → `expires_at && new Date(expires_at) < now`
  - `pending`  → `not_before && new Date(not_before) > now`
  - `active`   → otherwise
- Render a coloured pill / dot with `i18n` strings (e.g. `SETTINGS$API_KEY_STATUS_ACTIVE`, `_PENDING`, `_EXPIRED`).
- For pending/expired keys, also dim the row (e.g. `opacity-60`) and show the relevant `not_before` / `expires_at` timestamp beneath the status.
- The `Delete` action should remain enabled for pending and active keys; consider disabling delete on expired keys (optional UX choice — recommend allowing it for now so users can clean up).
- Update the `formatDate` helper to accept `null` and return `"—"` (currently it returns `"Never"`, which is wrong for `not_before`/`expires_at` — only `last_used_at` is "Never" when null).

---

## 11. Frontend mock handlers

**File:** `frontend/src/mocks/api-keys-handlers.ts`

- Add `not_before: null, expires_at: null` to `DEFAULT_API_KEYS` and to the new-key construction in the `POST /api/keys` handler.
- Read `not_before` / `expires_at` from the request body in the POST handler.
- Add a 400 response for the same validation the backend performs (window order, not in the past) so the mock stays a faithful preview.
- Add at least one fixture row with a future `not_before` and one with a past `expires_at` to exercise the new column in the UI under the mock.

---

## 12. i18n

**File:** `frontend/src/i18n/declaration.ts` — add:

- `SETTINGS$API_KEY_ACTIVE_WINDOW`
- `SETTINGS$API_KEY_NOT_BEFORE`
- `SETTINGS$API_KEY_EXPIRES_AT`
- `SETTINGS$API_KEY_STATUS_ACTIVE`
- `SETTINGS$API_KEY_STATUS_PENDING`
- `SETTINGS$API_KEY_STATUS_EXPIRED`
- `SETTINGS$API_KEY_WINDOW_INVALID` (for the modal's "not_before must be < expires_at" error)

**File:** `frontend/src/i18n/translation.json` — provide default (English) values for each new key. Other locales fall back to English until translated.

---

## 13. Frontend tests

- `create-api-key-modal.test.tsx` (new or extend an existing one if present):
  - Submits with both timestamps.
  - Submits with neither (backward-compatible).
  - Shows an error toast when `notBefore > expiresAt`.
  - Resets all fields after successful creation.
- `api-keys-manager.test.tsx` (extend):
  - Renders the new column.
  - Renders the correct status pill for each of pending / active / expired fixture rows.
- `use-create-api-key.test.ts` (if present, otherwise inline in the modal test): asserts the mutation calls `ApiKeysClient.createApiKey` with `{ name, not_before, expires_at }`.

Use the existing vitest + testing-library stack — no new test infrastructure needed.

---

## 14. Verification checklist (before opening the PR)

Per `AGENTS.md` (enterprise section + the in-repo PR template), before pushing:

1. `cd enterprise && poetry run pre-commit run --all-files --show-diff-on-failure --config ./dev_config/python/.pre-commit-config.yaml` — must pass clean.
2. `cd enterprise && PYTHONPATH=".:$PYTHONPATH" poetry run pytest --confcutdir=tests/unit tests/unit/test_api_key_store.py tests/unit/server/routes/test_api_keys.py tests/unit/server/auth/ -p no:ddtrace` — all new and existing tests green.
3. `cd frontend && npm run lint:fix && npm run build` — must pass clean.
4. `cd frontend && npm run test -- -t "api-key"` (or run the full `npm run test`) — green.
5. Confirm the mock UI renders the new column under `npm run dev:mock` and that creating a key with both bounds populates the table correctly.
6. Manually exercise in the SaaS dev environment (or a local run with `ENABLE_V1=0` / the legacy stack): create a key with `not_before` 1 minute in the future, observe that requests with that key get a 401 until the time elapses, then succeed; create one with `expires_at` 1 minute in the future, observe it stop working after that minute.

---

## 15. Out of scope (explicitly)

- Auto-deletion of expired keys from the database (no schema or job changes; admins can prune via the existing `DELETE /api/keys/{id}` flow).
- Per-key scopes / permissions (a separate feature).
- Surfacing `not_before` / `expires_at` on the `CurrentApiKeyResponse` endpoint (not needed by the current consumers; easy follow-up if a downstream SDK wants it).
- Changing the LiteLLM BYOR key flow — those keys are managed by `LiteLlmManager` and are outside the `api_keys` table.

---

## 16. Risk assessment

- **Schema change is purely additive** (`not_before NULL`) → no migration risk for existing rows; downgrade path drops the column.
- **Validation logic is additive and defaults preserve behaviour** (keys with both fields `NULL` behave exactly as today, and the existing `expires_at` test suite already exercises the legacy path).
- **UI changes are isolated** to the API keys screen and its modals; no shared component is modified beyond adding fields to `SettingsInput`, which is already polymorphic via `type`.
- **Authentication impact is contained** to `validate_api_key`; the `SaasUserAuth` object contract is unchanged.

Security review (per the in-repo security skill):

- Inputs are validated both client- and server-side.
- Time comparisons happen in UTC; the existing `tzinfo` handling for `expires_at` is mirrored for `not_before`.
- The new validator (`not_before < expires_at`, neither in the past) prevents users from creating a degenerate window.
- Audit logging distinguishes "expired" from "not yet active" so abuse / clock-skew is visible.
- No new auth surface area; existing `Bearer` / `X-Session-API-Key` / `X-Access-Token` extraction is untouched.
