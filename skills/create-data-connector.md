---
name: create-data-connector
description: Create a new external API data connector for the B1 framework. Use when the user wants to connect to an external API (HubSpot, Mixpanel, Stripe, GitHub, etc.) and expose it as a B1 DataSource with filtering, pagination, and optional CRUD support.
---

# Create Data Connector

Guides creation of external API connectors following the 4-file NestJS pattern used by the B1 framework. Produces a fully working `IDataConnector` implementation registered via `ConnectorModule.forRoot()`, plus blueprint DataSource objects.

**Reference implementation:** `src/app-server-ts/src/api/spacex/` (read-only, public API, MongoDB-style POST body filtering).

---

## Phase 1: Gather Requirements

Collect the following from the user before generating code. Ask for all items; use sensible defaults where noted.

| Item | Example | Default |
|------|---------|---------|
| Provider key (lowercase) | `hubspot` | — (required) |
| Base URL | `https://api.hubapi.com` | — (required) |
| API version segment | `v3` | `""` (none) |
| Authentication method | See Phase 1b | `none` |
| Resources to expose | `contacts`, `companies` | — (at least one) |
| Primary key field per resource | `id` | `id` |
| CRUD level | `read-only` / `full` | `read-only` |
| API query style | A–E (see Phase 2) | — (required) |
| API docs URL | `https://docs.example.com` | `""` (optional) |
| Create sample search screen? | `yes` / `no` | `no` |

### Authentication methods

| Key | Description |
|-----|-------------|
| `none` | Public API, no credentials (e.g. SpaceX) |
| `env-token` | Bearer token from environment variable |
| `cached-oauth` | OAuth client-credentials with in-memory cache + 401 retry |
| `auth-server-secrets` | Token retrieved from the B1 auth server secrets API |

---

## Phase 2: API Query Style Assessment

Determine which pattern the target API uses for filtering/searching. This drives the filter translation file.

### Pattern A — REST Query Params

APIs where filters are appended as URL query parameters.
Examples: Stripe (`?status=active&limit=10`), many simple REST APIs.

```typescript
// Filter translation: B1 operator → query params
function buildQueryParams(query: QueryObject<unknown>): Record<string, string> {
  const params: Record<string, string> = {};
  if (query.filters && isFilterList(query.filters)) {
    for (const f of query.filters.filters) {
      if (!isFilterList(f)) {
        const c = f as FilterCriteria;
        // REST params typically only support equality
        if (c.operator === 'eq' || c.operator === '=') {
          params[c.field] = String(c.value);
        }
      }
    }
  }
  if (query.limit != null) params.limit = String(query.limit);
  if (query.offset != null) params.offset = String(query.offset);
  return params;
}
```

**Supported B1 operators:** `eq`. Other operators must be filtered client-side — document this in a comment.

### Pattern B — POST Body / MongoDB-style

APIs that accept a POST with MongoDB query operators.
Examples: SpaceX (`{ query: { field: { $gt: value } }, options: { limit, offset, sort } }`).

```typescript
// See spacex.filter.ts for the full reference implementation.
// Key: criteriaToMongo() maps B1 operators to MongoDB operators.
// Supports: eq, neq, gt, lt, gte, lte, contains, begins, ends, isNull, notcontains
```

**Supported B1 operators:** eq, neq, gt, lt, gte, lte, contains, notcontains, begins, ends, isNull.

### Pattern C — POST Body / Custom Syntax

APIs with their own query DSL sent as a POST body.
Examples: HubSpot Search API, Elasticsearch.

```typescript
// Each API has its own filter format — study the API docs.
// Map B1 operators to the API-specific filter syntax.
// Example (HubSpot-style):
function buildSearchBody(query: QueryObject<unknown>): Record<string, unknown> {
  const filters: unknown[] = [];
  if (query.filters && isFilterList(query.filters)) {
    for (const f of query.filters.filters) {
      if (!isFilterList(f)) {
        const c = f as FilterCriteria;
        filters.push({
          propertyName: c.field,
          operator: mapOperator(c.operator),
          value: String(c.value)
        });
      }
    }
  }
  return {
    filterGroups: [{ filters }],
    limit: query.limit ?? 50,
    after: query.offset ?? 0
  };
}
```

### Pattern D — GraphQL

APIs with a GraphQL endpoint.
Examples: GitHub GraphQL API, Shopify Storefront API.

```typescript
// Build a GraphQL query string with variables for filters.
// Map B1 filters to GraphQL query arguments or filter input types.
function buildGraphQLQuery(resource: string, query: QueryObject<unknown>): { query: string; variables: Record<string, unknown> } {
  // Resource-specific query with pagination variables
  return {
    query: `query ($first: Int, $after: String) { ${resource}(first: $first, after: $after) { nodes { ...fields } pageInfo { hasNextPage endCursor } } }`,
    variables: { first: query.limit ?? 50, after: query.offset ? String(query.offset) : null }
  };
}
```

**Supported B1 operators:** depends on the GraphQL schema — document per-resource.

### Pattern E — No Server-Side Filtering

Simple GET endpoints with no filtering support.
Examples: small reference APIs, config endpoints.

```typescript
// No filter translation needed — just GET the endpoint.
// All filtering/sorting handled client-side by the B1 grid.
```

### B1 Operator Reference

| B1 Operator | Aliases | Description |
|-------------|---------|-------------|
| `eq` | `=` | Equal |
| `neq` | `ne`, `<>` | Not equal |
| `gt` | `>` | Greater than |
| `lt` | `<` | Less than |
| `gte` | `ge`, `>=` | Greater than or equal |
| `lte` | `le`, `<=` | Less than or equal |
| `contains` | `matches` | String contains (case-insensitive) |
| `notcontains` | | String does not contain |
| `begins` | `startswith`, `beginsmatches` | Starts with |
| `ends` | `endswith` | Ends with |
| `isNull` | | Is null (value=true) or is not null (value=false) |

---

## Phase 3: Authentication Pattern Templates

Use the appropriate template based on the authentication method chosen in Phase 1.

### 1. No Auth (public API)

```typescript
// No auth headers needed. Just make the HTTP call.
const response = await firstValueFrom(
  this.httpService.get<ResponseType>(url, { timeout: 30000 })
);
```

### 2. Env Var Token

```typescript
// In connector constructor or as a class property:
private getAuthHeaders(): Record<string, string> {
  const token = process.env.{PROVIDER}_ACCESS_TOKEN;
  if (!token) {
    throw new Error('{PROVIDER}_ACCESS_TOKEN environment variable is not set');
  }
  return { Authorization: `Bearer ${token}` };
}

// In fetch method:
const response = await firstValueFrom(
  this.httpService.get<ResponseType>(url, {
    headers: { ...this.getAuthHeaders(), 'Content-Type': 'application/json' },
    timeout: 30000
  })
);
```

Environment variable name convention: `{PROVIDER}_ACCESS_TOKEN` (e.g. `HUBSPOT_ACCESS_TOKEN`).

### 3. Cached OAuth

```typescript
private cachedToken: { token: string; expiresAt: number } | null = null;

private async getAccessToken(): Promise<string> {
  if (this.cachedToken && Date.now() < this.cachedToken.expiresAt) {
    return this.cachedToken.token;
  }

  const response = await firstValueFrom(
    this.httpService.post<{ access_token: string; expires_in: number }>(
      '{TOKEN_URL}',
      new URLSearchParams({
        grant_type: 'client_credentials',
        client_id: process.env.{PROVIDER}_CLIENT_ID!,
        client_secret: process.env.{PROVIDER}_CLIENT_SECRET!
      }).toString(),
      { headers: { 'Content-Type': 'application/x-www-form-urlencoded' }, timeout: 30000 }
    )
  );

  this.cachedToken = {
    token: response.data.access_token,
    expiresAt: Date.now() + (response.data.expires_in - 60) * 1000
  };
  return this.cachedToken.token;
}

// On 401 response, clear cache and retry once:
private async fetchWithRetry<T>(requestFn: () => Promise<T>): Promise<T> {
  try {
    return await requestFn();
  } catch (error) {
    if (error?.response?.status === 401) {
      this.cachedToken = null;
      return await requestFn();
    }
    throw error;
  }
}
```

### 4. Auth Server Secrets

```typescript
private async getAccessToken(headers: Record<string, string>): Promise<string> {
  const response = await firstValueFrom(
    this.httpService.get<{ value: string }>(
      `${process.env.AUTH_URL}/api/secrets/key/oauth:accessToken:{provider}`,
      { headers, timeout: 30000 }
    )
  );
  return response.data.value;
}
```

---

## Phase 4: Generate Backend Code (4-File Pattern)

All files go under `src/app-server-ts/src/api/{provider}/`.

Replace `{provider}` (lowercase), `{Provider}` (PascalCase), and `{PROVIDER}` (UPPERCASE) throughout.

### File 1: `{provider}.types.ts`

```typescript
/**
 * Type definitions for the {Provider} API connector.
 */

/** Resources exposed by this connector. */
export const RESOURCES = new Set([
  // '{resource1}',
  // '{resource2}',
]);

// Add API-specific request/response types here.
// Example for POST-body APIs:
//
// export interface {Provider}QueryRequest {
//   query: Record<string, unknown>;
//   options: { limit?: number; offset?: number; sort?: Record<string, 1 | -1> };
// }
//
// export interface {Provider}QueryResponse<T> {
//   docs: T[];
//   totalDocs: number;
// }
```

### File 2: `{provider}.filter.ts`

```typescript
import type { FilterCriteria, FilterList, Operator, OrderBy, QueryObject } from '@buildone/app-server-tslib/utils';
import { isFilterList } from '@buildone/app-server-tslib/utils';

/**
 * Translates a B1 QueryObject into the {Provider} API's native query format.
 *
 * Supported B1 operators: [list supported operators here]
 * Unsupported operators fall back to equality or are ignored — document below.
 */
export function build{Provider}Query(query: QueryObject<unknown>): unknown {
  // Implementation depends on API query style (Pattern A–E from Phase 2).
  // See Phase 2 templates for the appropriate translation logic.
}
```

**Key imports:**
- `FilterCriteria`, `FilterList`, `Operator`, `OrderBy`, `QueryObject` from `@buildone/app-server-tslib/utils`
- `isFilterList` from `@buildone/app-server-tslib/utils`

### File 3: `{provider}.connector.ts`

```typescript
import { Injectable, Logger, MethodNotAllowedException } from '@nestjs/common';
import { HttpService } from '@nestjs/axios';
import { firstValueFrom } from 'rxjs';
import type { QueryObject } from '@buildone/app-server-tslib/utils';
import type { IDataConnector } from '@buildone/app-server-tslib/modules';
import { build{Provider}Query } from './{provider}.filter';
import { RESOURCES } from './{provider}.types';

/** Default {Provider} API base URL. Override with {PROVIDER}_API_BASE_URL env var. */
const DEFAULT_BASE_URL = '{base_url}';

@Injectable()
export class {Provider}Connector implements IDataConnector {
  private readonly logger = new Logger({Provider}Connector.name);
  private readonly baseUrl: string;

  constructor(private readonly httpService: HttpService) {
    this.baseUrl = process.env.{PROVIDER}_API_BASE_URL ?? DEFAULT_BASE_URL;
  }

  async fetch(object: string, query: QueryObject<unknown>): Promise<unknown[]> {
    // Build URL and query, call API, return array of records.
    // Wrap singleton responses in an array: Array.isArray(data) ? data : [data]
    throw new Error('Not implemented');
  }

  async create(object: string, records: Record<string, unknown>[]): Promise<unknown[]> {
    // For read-only connectors:
    throw new MethodNotAllowedException('{Provider} connector is read-only');
    // For full CRUD: implement creation logic
  }

  async update(object: string, records: Record<string, unknown>[]): Promise<unknown[]> {
    throw new MethodNotAllowedException('{Provider} connector is read-only');
  }

  async delete(object: string, ids: string[]): Promise<void> {
    throw new MethodNotAllowedException('{Provider} connector is read-only');
  }
}
```

**Key imports:**
- `IDataConnector` from `@buildone/app-server-tslib/modules`
- `HttpService` from `@nestjs/axios`
- `firstValueFrom` from `rxjs`
- `QueryObject` from `@buildone/app-server-tslib/utils`

**Patterns:**
- Use `Logger` with the connector class name
- Base URL configurable via `{PROVIDER}_API_BASE_URL` env var
- Always set `timeout: 30000` on HTTP calls
- Wrap singleton responses in arrays for `IDataConnector` compliance
- All 4 methods (`fetch`, `create`, `update`, `delete`) must be implemented — throw `MethodNotAllowedException` for unsupported operations

### File 4: `{provider}.module.ts`

```typescript
import { Module } from '@nestjs/common';
import { HttpModule } from '@nestjs/axios';
import { ConnectorModule } from '@buildone/app-server-tslib/modules';
import { {Provider}Connector } from './{provider}.connector';

@Module({
  imports: [
    ConnectorModule.forRoot({
      connectors: [{ provide: '{provider}', useClass: {Provider}Connector }],
      imports: [HttpModule]
    })
  ]
})
export class {Provider}Module {}
```

**Key import:** `ConnectorModule` from `@buildone/app-server-tslib/modules`

The `provide` string becomes the provider segment of the URL: `api/connector/{provider}/{object}/fetch`.

---

## Phase 5: Register Module

Update `src/app-server-ts/src/api/api.module.ts`:

1. Add import: `import { {Provider}Module } from './{provider}/{provider}.module';`
2. Add `{Provider}Module` to both the `imports` and `exports` arrays.

---

## Phase 6: Verify Build

```bash
cd src/app-server-ts && npx tsc --noEmit
```

Fix any TypeScript errors before proceeding.

---

## Phase 7: Create Blueprint DataSources via B1 MCP

For **each resource**, create a DataSource blueprint object using B1 MCP tools.

Required attributes:

| Attribute | Value |
|-----------|-------|
| `subtype` | `"DataConnector"` |
| `resourceName` | `"service/app/api/connector/{provider}/{resource}"` |
| `keyFields` | `"{primary-key}"` (from Phase 1) |
| `rowsToBatch` | `50` |
| `fetchMode` | `"REPLACE"` |

Use `mcp__B1__update_blueprint` to create each DataSource object, and `mcp__B1__Patch_Blueprint` to set attributes.

---

## Phase 8: Create Sample Search Screen (OPTIONAL — skip by default)

**This phase is skipped unless the user explicitly requests it.** Ask for confirmation before proceeding.

If requested:

1. Search for grid/search screen templates in the Samples module using `mcp__B1__query_blueprint`
2. Use `mcp__B1__Create_Screen_from_Template` with the chosen template and DataSource
3. Map placeholder fields to actual resource fields
4. Export with `mcp__B1__Export_Blueprint_Objects` and preview

---

## Important Rules

1. **Always use B1 MCP** for blueprint operations — never edit JSON files directly.
2. **Provider key must be lowercase** — used in file names, folder name, module provide string, and URL segment.
3. **All 4 `IDataConnector` methods are required** — throw `MethodNotAllowedException` for unsupported operations.
4. **Use `firstValueFrom()`** to convert rxjs Observables from `HttpService` — never use `.toPromise()`.
5. **Set `timeout: 30000`** on all HTTP calls to external APIs.
6. **Wrap singleton responses in arrays** — `IDataConnector.fetch()` must always return `unknown[]`.
7. **Document unsupported operators** — add a comment in the filter file listing which B1 operators are not supported by the target API.
8. **`resourceName` pattern:** `service/app/api/connector/{provider}/{resource}` — the `service/app/` prefix is required for DataSource blueprints.
9. **One module per provider** — multiple resources are handled by the single connector class dispatching on the `object` parameter.
10. **Never modify the SpaceX reference files** — they are the reference implementation, not a template to overwrite.

---

## Error Handling

### Build Errors

| Error | Fix |
|-------|-----|
| `Cannot find module '@buildone/app-server-tslib/...'` | Check import paths — use `/modules` for `IDataConnector`, `ConnectorModule`; use `/utils` for filter types |
| `Class does not implement IDataConnector` | Ensure all 4 methods (`fetch`, `create`, `update`, `delete`) are implemented |
| `Module not found` | Verify the module is imported in `api.module.ts` |

### Runtime Errors

| Error | Fix |
|-------|-----|
| `404 on /api/connector/{provider}/{resource}/fetch` | Check `provide` string in module matches URL segment; verify module is in `api.module.ts` imports |
| `MethodNotAllowedException` | Expected for read-only connectors on create/update/delete |
| `ETIMEDOUT` | External API unreachable — check base URL and network access |

### Blueprint Errors

| Error | Fix |
|-------|-----|
| DataSource returns no data | Verify `resourceName` matches the connector URL pattern exactly |
| Wrong data shape | Ensure `fetch()` returns an array of flat objects with the expected `keyFields` property |

---

## Examples

### Example 1: Read-Only Public API (SpaceX Pattern)

- Provider: `spacex`
- Auth: `none`
- Query style: **Pattern B** (POST body, MongoDB-style)
- Resources: `launches`, `rockets`, `crew`, etc.
- CRUD: read-only

Reference files:
- `src/app-server-ts/src/api/spacex/spacex.connector.ts`
- `src/app-server-ts/src/api/spacex/spacex.filter.ts`
- `src/app-server-ts/src/api/spacex/spacex.types.ts`
- `src/app-server-ts/src/api/spacex/spacex.module.ts`

### Example 2: Token-Authenticated REST API (HubSpot Pattern)

- Provider: `hubspot`
- Auth: `env-token` (`HUBSPOT_ACCESS_TOKEN`)
- Query style: **Pattern C** (POST body, custom syntax for search) or **Pattern A** (REST params for list)
- Resources: `contacts`, `companies`, `deals`
- CRUD: full

### Example 3: REST Params API (Stripe Pattern)

- Provider: `stripe`
- Auth: `env-token` (`STRIPE_SECRET_KEY`)
- Query style: **Pattern A** (REST query params)
- Resources: `customers`, `charges`, `subscriptions`
- CRUD: full
- Note: Stripe uses `?starting_after=` cursor pagination instead of offset

---

## Reference Files

### SpaceX Reference Implementation
- `src/app-server-ts/src/api/spacex/spacex.connector.ts` — IDataConnector implementation
- `src/app-server-ts/src/api/spacex/spacex.filter.ts` — MongoDB-style filter translation
- `src/app-server-ts/src/api/spacex/spacex.types.ts` — type definitions and resource set
- `src/app-server-ts/src/api/spacex/spacex.module.ts` — ConnectorModule.forRoot() registration
- `src/app-server-ts/src/api/api.module.ts` — module import/export location

### Framework Source
- `IDataConnector` interface: `@buildone/app-server-tslib/modules`
- `ConnectorModule`: `@buildone/app-server-tslib/modules`
- Filter types (`FilterCriteria`, `FilterList`, `QueryObject`, etc.): `@buildone/app-server-tslib/utils`
- `isFilterList` utility: `@buildone/app-server-tslib/utils`

### Related Documentation

Resolve the knowledge base path before reading these files:
```
KNOWLEDGE_PATH=$(grep -s '^BUILDONE_KNOWLEDGE_FILES_PATH=' /workspace/.env | head -1 | cut -d'=' -f2- | tr -d '"' | tr -d "'"); \
if [ -z "$KNOWLEDGE_PATH" ]; then KNOWLEDGE_PATH="/knowledge"; \
elif [[ "$KNOWLEDGE_PATH" != /* ]]; then KNOWLEDGE_PATH="/workspace/$KNOWLEDGE_PATH"; fi
```

- `KNOWLEDGE_PATH/architecture_info/connectors.md` — full connector framework docs
- `KNOWLEDGE_PATH/blueprint_dsl/CLAUDE.md` — blueprint DSL reference
- `KNOWLEDGE_PATH/blueprint_dsl/templates-and-create-from-template.md` — template usage
