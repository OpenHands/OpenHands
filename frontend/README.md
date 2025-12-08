# Getting Started with the OpenHands Frontend

## Overview

This is the frontend of the OpenHands project. It is a React application that provides a web interface for the OpenHands project.

## Tech Stack

- Remix SPA Mode (React + Vite + React Router)
- TypeScript
- Redux
- TanStack Query
- Tailwind CSS
- i18next
- React Testing Library
- Vitest
- Mock Service Worker

## Getting Started

### Prerequisites

- Node.js 20.x or later
- `npm`, `bun`, or any other package manager that supports the `package.json` file

### Installation

```sh
# Clone the repository
git clone https://github.com/OpenHands/OpenHands.git

# Change the directory to the frontend
cd OpenHands/frontend

# Install the dependencies
npm install
```

### Running the Application in Development Mode

We use `msw` to mock the backend API. To start the application with the mocked backend, run the following command:

```sh
npm run dev
```

This will start the application in development mode. Open [http://localhost:3001](http://localhost:3001) to view it in the browser.

**NOTE: The backend is _partially_ mocked using `msw`. Therefore, some features may not work as they would with the actual backend.**

See the [Development.md](../Development.md) for extra tips on how to run in development mode.

### Running the Application with the Actual Backend (Production Mode)

To run the application with the actual backend:

```sh
# Build the application from the root directory
make build

# Start the application
make run
```
Or to run backend and frontend separately.

```sh
# Start the backend from the root directory
make start-backend

# Serve the frontend
make start-frontend or
cd frontend && npm start -- --port 3001
```

Start frontend with Mock Service Worker (MSW), see testing for more info.
```sh
npm run dev:mock or npm run dev:mock:saas
```

### Environment Variables

The frontend application uses the following environment variables:

| Variable                    | Description                                                            | Default Value    |
| --------------------------- | ---------------------------------------------------------------------- | ---------------- |
| `VITE_BACKEND_BASE_URL`     | The backend hostname without protocol (used for WebSocket connections) | `localhost:3000` |
| `VITE_BACKEND_HOST`         | The backend host with port for API connections                         | `127.0.0.1:3000` |
| `VITE_MOCK_API`             | Enable/disable API mocking with MSW                                    | `false`          |
| `VITE_MOCK_SAAS`            | Simulate SaaS mode in development                                      | `false`          |
| `VITE_USE_TLS`              | Use HTTPS/WSS for backend connections                                  | `false`          |
| `VITE_FRONTEND_PORT`        | Port to run the frontend application                                   | `3001`           |
| `VITE_INSECURE_SKIP_VERIFY` | Skip TLS certificate verification                                      | `false`          |
| `VITE_GITHUB_TOKEN`         | GitHub token for repository access (used in some tests)                | -                |

You can create a `.env` file in the frontend directory with these variables based on the `.env.sample` file.

### Project Structure

```sh
frontend
├── __tests__ # Tests
├── public
├── src
│   ├── api # API calls
│   ├── assets
│   ├── components
│   ├── context # Local state management
│   ├── hooks # Custom hooks
│   ├── i18n # Internationalization
│   ├── mocks # MSW mocks for development
│   ├── routes # React Router file-based routes
│   ├── services
│   ├── state # Redux state management
│   ├── types
│   ├── utils # Utility/helper functions
│   └── root.tsx # Entry point
└── .env.sample # Sample environment variables
```

#### Components

Components are organized into folders based on their **domain**, **feature**, or **shared functionality**.

```sh
components
├── features # Domain-specific components
├── layout
├── modals
└── ui # Shared UI components
```

### Features

- Real-time updates with WebSockets
- Internationalization
- Router data loading with Remix
- User authentication with GitHub OAuth (if saas mode is enabled)

## Testing

### Testing Framework and Tools

We use the following testing tools:
- **Test Runner**: Vitest
- **Rendering**: React Testing Library
- **User Interactions**: @testing-library/user-event
- **API Mocking**: [Mock Service Worker (MSW)](https://mswjs.io/)
- **Code Coverage**: Vitest with V8 coverage

### Running Tests

To run all tests:
```sh
npm run test
```

To run tests with coverage:
```sh
npm run test:coverage
```

### Testing Best Practices

1. **Component Testing**
   - Test components in isolation
   - Use our custom [`renderWithProviders()`](https://github.com/OpenHands/OpenHands/blob/ce26f1c6d3feec3eedf36f823dee732b5a61e517/frontend/test-utils.tsx#L56-L85) that wraps the components we want to test in our providers. It is especially useful for components that use Redux
   - Use `render()` from React Testing Library to render components
   - Prefer querying elements by role, label, or test ID over CSS selectors
   - Test both rendering and interaction scenarios

2. **User Event Simulation**
   - Use `userEvent` for simulating realistic user interactions
   - Test keyboard events, clicks, typing, and other user actions
   - Handle edge cases like disabled states, empty inputs, etc.

3. **Mocking**
   - We test components that make network requests by mocking those requests with Mock Service Worker (MSW)
   - Use `vi.fn()` to create mock functions for callbacks and event handlers
   - Mock external dependencies and API calls (more info)[https://mswjs.io/docs/getting-started]
   - Verify mock function calls using `.toHaveBeenCalledWith()`, `.toHaveBeenCalledTimes()`

4. **Accessibility Testing**
   - Use `toBeInTheDocument()` to check element presence
   - Test keyboard navigation and screen reader compatibility
   - Verify correct ARIA attributes and roles

5. **State and Prop Testing**
   - Test component behavior with different prop combinations
   - Verify state changes and conditional rendering
   - Test error states and loading scenarios

6. **Internationalization (i18n) Testing**
   - Test translation keys and placeholders
   - Verify text rendering across different languages

Example Test Structure:
```typescript
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi } from "vitest";

describe("ComponentName", () => {
  it("should render correctly", () => {
    render(<Component />);
    expect(screen.getByRole("button")).toBeInTheDocument();
  });

  it("should handle user interactions", async () => {
    const mockCallback = vi.fn();
    const user = userEvent.setup();

    render(<Component onClick={mockCallback} />);
    const button = screen.getByRole("button");

    await user.click(button);
    expect(mockCallback).toHaveBeenCalledOnce();
  });
});
```

### Example Tests in the Codebase

For real-world examples of testing, check out these test files:

1. **Chat Input Component Test**:
   [`__tests__/components/chat/chat-input.test.tsx`](https://github.com/OpenHands/OpenHands/blob/main/frontend/__tests__/components/chat/chat-input.test.tsx)
   - Demonstrates comprehensive testing of a complex input component
   - Covers various scenarios like submission, disabled states, and user interactions

2. **File Explorer Component Test**:
   [`__tests__/components/file-explorer/file-explorer.test.tsx`](https://github.com/OpenHands/OpenHands/blob/main/frontend/__tests__/components/file-explorer/file-explorer.test.tsx)
   - Shows testing of a more complex component with multiple interactions
   - Illustrates testing of nested components and state management

### Test Coverage

- Aim for high test coverage, especially for critical components
- Focus on testing different scenarios and edge cases
- Use code coverage reports to identify untested code paths

### Continuous Integration

Tests are automatically run during:
- Pre-commit hooks
- Pull request checks
- CI/CD pipeline

## Contributing

Please read the [CONTRIBUTING.md](../CONTRIBUTING.md) file for details on our code of conduct, and the process for submitting pull requests to us.

## Troubleshooting

TODO
