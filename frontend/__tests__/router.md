# Testing with React Router

## Overview

React Router components and hooks require a routing context to function. In tests, we need to provide this context while maintaining control over the routing state.

This guide covers the two main approaches used in the OpenHands frontend:

1. **`createRoutesStub`** - Creates a complete route structure for testing components with their actual route configuration, loaders, and nested routes.
2. **`MemoryRouter`** - Provides a minimal routing context for components that just need router hooks to work.

Choose your approach based on what your component actually needs from the router.

## When to Use Each Approach

### `createRoutesStub` (Recommended)

Use `createRoutesStub` when your component:
- Relies on route parameters (`useParams`)
- Uses loader data (`useLoaderData`) or `clientLoader`
- Has nested routes or uses `<Outlet />`
- Needs to test navigation between routes

> [!NOTE]
> `createRoutesStub` is intended for unit testing **reusable components** that depend on router context. For testing full route/page components, consider E2E tests (Playwright, Cypress) instead.

```typescript
import { createRoutesStub } from "react-router";
import { render } from "@testing-library/react";

const RouterStub = createRoutesStub([
  {
    Component: MyRouteComponent,
    path: "/conversations/:conversationId",
  },
]);

render(<RouterStub initialEntries={["/conversations/123"]} />);
```

**With nested routes and loaders:**

```typescript
const RouterStub = createRoutesStub([
  {
    Component: SettingsScreen,
    clientLoader,
    path: "/settings",
    children: [
      {
        Component: () => <div data-testid="llm-settings" />,
        path: "/settings",
      },
      {
        Component: () => <div data-testid="git-settings" />,
        path: "/settings/integrations",
      },
    ],
  },
]);

render(<RouterStub initialEntries={["/settings/integrations"]} />);
```

> [!TIP]
> When using `clientLoader` from a Route module, you may encounter type mismatches. Use `@ts-expect-error` as a workaround:

```typescript
import { clientLoader } from "@/routes/settings";

const RouterStub = createRoutesStub([
  {
    path: "/settings",
    Component: SettingsScreen,
    // @ts-expect-error: loader types won't align between test and app code
    loader: clientLoader,
  },
]);
```

### `MemoryRouter`

Use `MemoryRouter` when your component:
- Only needs basic routing context to render
- Uses `<Link>` components but you don't need to test navigation
- Doesn't depend on specific route parameters or loaders

```typescript
import { MemoryRouter } from "react-router";
import { render } from "@testing-library/react";

render(
  <MemoryRouter>
    <MyComponent />
  </MemoryRouter>
);
```

**With initial route:**

```typescript
render(
  <MemoryRouter initialEntries={["/some/path"]}>
    <MyComponent />
  </MemoryRouter>
);
```

## Anti-patterns to Avoid

### Using `BrowserRouter` in tests

`BrowserRouter` interacts with the actual browser history API, which can cause issues in test environments:

```typescript
// ❌ Avoid
render(
  <BrowserRouter>
    <MyComponent />
  </BrowserRouter>
);

// ✅ Use MemoryRouter instead
render(
  <MemoryRouter>
    <MyComponent />
  </MemoryRouter>
);
```

### Mocking router hooks when `createRoutesStub` would work

Mocking hooks like `useParams` directly can be brittle and doesn't test the actual routing behavior:

```typescript
// ❌ Avoid when possible
vi.mock("react-router", async () => {
  const actual = await vi.importActual("react-router");
  return {
    ...actual,
    useParams: () => ({ conversationId: "123" }),
  };
});

// ✅ Prefer createRoutesStub - tests real routing behavior
const RouterStub = createRoutesStub([
  {
    Component: MyComponent,
    path: "/conversations/:conversationId",
  },
]);

render(<RouterStub initialEntries={["/conversations/123"]} />);
```

## Common Patterns

### Combining with `QueryClientProvider`

Many components need both routing and TanStack Query context:

```typescript
import { createRoutesStub } from "react-router";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { retry: false },
  },
});

const RouterStub = createRoutesStub([
  {
    Component: MyComponent,
    path: "/",
  },
]);

render(<RouterStub />, {
  wrapper: ({ children }) => (
    <QueryClientProvider client={queryClient}>
      {children}
    </QueryClientProvider>
  ),
});
```

### Testing navigation behavior

Verify that user interactions trigger the expected navigation:

```typescript
import { createRoutesStub } from "react-router";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

const RouterStub = createRoutesStub([
  {
    Component: HomeScreen,
    path: "/",
  },
  {
    Component: () => <div data-testid="settings-screen" />,
    path: "/settings",
  },
]);

render(<RouterStub initialEntries={["/"]} />);

const user = userEvent.setup();
await user.click(screen.getByRole("link", { name: /settings/i }));

expect(screen.getByTestId("settings-screen")).toBeInTheDocument();
```

## See Also

### Codebase Examples

- [settings.test.tsx](__tests__/routes/settings.test.tsx) - `createRoutesStub` with nested routes and loaders
- [home-screen.test.tsx](__tests__/routes/home-screen.test.tsx) - `createRoutesStub` with navigation testing
- [chat-interface.test.tsx](__tests__/components/chat/chat-interface.test.tsx) - `MemoryRouter` usage

### Official Documentation

- [React Router Testing Guide](https://reactrouter.com/start/framework/testing) - Official guide on testing with `createRoutesStub`
- [MemoryRouter API](https://reactrouter.com/api/declarative-routers/MemoryRouter) - API reference for `MemoryRouter`
