# Mock Service Worker (MSW) Guide

## Overview

[Mock Service Worker (MSW)](https://mswjs.io/) is an API mocking library that intercepts outgoing network requests at the network level. Unlike traditional mocking that patches `fetch` or `axios`, MSW uses a Service Worker in the browser and direct request interception in Node.js—making mocks transparent to your application code.

We use MSW in this project for:
- **Testing**: Write reliable unit and integration tests without real network calls
- **Development**: Run the frontend with mocked APIs when the backend isn't available or when working on features with pending backend APIs

The same mock handlers work in both environments, so you write them once and reuse everywhere.

## Relevant Files

- `src/mocks/handlers.ts` - Main handler registry that combines all domain handlers
- `src/mocks/*-handlers.ts` - Domain-specific handlers (auth, billing, conversation, etc.)
- `src/mocks/browser.ts` - Browser setup for development mode
- `src/mocks/node.ts` - Node.js setup for tests
- `vitest.setup.ts` - Global test setup with MSW lifecycle hooks

## Development Workflow

### Running with Mocked APIs

```sh
# Run with API mocking enabled
npm run dev:mock

# Run with API mocking + SaaS mode simulation
npm run dev:mock:saas
```

These commands set `VITE_MOCK_API=true` which activates the MSW Service Worker to intercept requests.

> [!NOTE]
> **OpenHands vs SaaS Mode**
>
> OpenHands runs in two modes:
> - **OpenHands mode**: For local/self-hosted deployments where users provide their own LLM API keys and configure git providers manually
> - **SaaS mode**: For the cloud offering with billing, managed API keys, and OAuth-based GitHub integration
>
> Use `dev:mock:saas` when working on SaaS-specific features like billing, API key management, or subscription flows.


## Writing Tests

### Service Layer Mocking (Recommended)

For most tests, mock at the service layer using `vi.spyOn`. This approach is explicit, test-scoped, and makes the scenario being tested clear.

```typescript
import { vi } from "vitest";
import SettingsService from "#/api/settings-service/settings-service.api";

const getSettingsSpy = vi.spyOn(SettingsService, "getSettings");
getSettingsSpy.mockResolvedValue({
  llm_model: "openai/gpt-4o",
  llm_api_key_set: true,
  // ... other settings
});
```

Use `mockResolvedValue` for success scenarios and `mockRejectedValue` for error scenarios:

```typescript
getSettingsSpy.mockRejectedValue(new Error("Failed to fetch settings"));
```

### Network Layer Mocking (Advanced)

For tests that need actual network-level behavior (WebSockets, testing retry logic, etc.), use `server.use()` to override handlers per test.

> [!IMPORTANT]
> **Reuse the global server instance** - Don't create new `setupServer()` calls in individual tests. The project already has a global MSW server configured in `vitest.setup.ts` that handles lifecycle (`server.listen()`, `server.resetHandlers()`, `server.close()`). Use `server.use()` to add runtime handlers for specific test scenarios.

```typescript
import { http, HttpResponse } from "msw";
import { server } from "#/mocks/node";

it("should handle server errors", async () => {
  server.use(
    http.get("/api/my-endpoint", () => {
      return new HttpResponse(null, { status: 500 });
    }),
  );
  // ... test code
});
```

For WebSocket testing, see `__tests__/helpers/msw-websocket-setup.ts` for utilities.

## Adding New API Mocks

When adding new API endpoints, create mocks in both places to maintain 1:1 similarity with the backend:

### 1. Add to `src/mocks/` (for development)

Create or update a domain-specific handler file:

```typescript
// src/mocks/my-feature-handlers.ts
import { http, HttpResponse } from "msw";

export const MY_FEATURE_HANDLERS = [
  http.get("/api/my-feature", () => {
    return HttpResponse.json({
      data: "mock response",
    });
  }),
];
```

Register in `handlers.ts`:

```typescript
import { MY_FEATURE_HANDLERS } from "./my-feature-handlers";

export const handlers = [
  // ... existing handlers
  ...MY_FEATURE_HANDLERS,
];
```

### 2. Mock in tests for specific scenarios

In your test files, spy on the service method to control responses per test case:

```typescript
import { vi } from "vitest";
import MyFeatureService from "#/api/my-feature-service.api";

const spy = vi.spyOn(MyFeatureService, "getData");
spy.mockResolvedValue({ data: "test-specific response" });
```

See `__tests__/routes/llm-settings.test.tsx` for a real-world example of service layer mocking.

> [!TIP]
> For guidance on creating service APIs, see `src/api/README.md`.

## Best Practices

- **Keep mocks close to real API contracts** - Update mocks when backend changes
- **Use service layer mocking for most tests** - It's simpler and more explicit
- **Reserve network layer mocking for integration tests** - WebSockets, retry logic, etc.
- **Export mock data from handler files** - Reuse in tests (e.g., `MOCK_DEFAULT_USER_SETTINGS`)
