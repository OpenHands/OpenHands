# Test Helpers

This directory contains reusable test utilities and components for the OpenHands frontend test suite.

## Files

### `websocket-test-components.tsx`
Contains React test components for accessing and displaying WebSocket-related store values:
- `ConnectionStatusComponent` - Displays WebSocket connection state
- `EventStoreComponent` - Displays event store values (events count, UI events count, latest event ID)
- `OptimisticUserMessageStoreComponent` - Displays optimistic user message store values
- `ErrorMessageStoreComponent` - Displays error message store values

These components are designed to be used in tests to verify that WebSocket events are properly processed and stored.



### `msw-websocket-setup.ts`
Contains MSW (Mock Service Worker) setup utilities for WebSocket testing:
- `createWebSocketLink()` - Creates a WebSocket link for MSW testing
- `createWebSocketMockServer()` - Creates and configures an MSW server for WebSocket testing
- `createWebSocketTestSetup()` - Creates a complete WebSocket testing setup
- `conversationWebSocketTestSetup()` - Standard setup for conversation WebSocket handler tests

## Usage

```typescript
import {
  ConnectionStatusComponent,
  EventStoreComponent,
} from "./__tests__/helpers/websocket-test-components";
import { conversationWebSocketTestSetup } from "./__tests__/helpers/msw-websocket-setup";

// Set up MSW server
const { wsLink, server } = conversationWebSocketTestSetup();

// Render components with WebSocket context (helper function defined in test file)
renderWithWebSocketContext(<ConnectionStatusComponent />);
```

## Benefits

- **Reusability**: Test components and utilities can be shared across multiple test files
- **Maintainability**: Changes to test setup only need to be made in one place
- **Consistency**: Ensures consistent test setup across different WebSocket-related tests
- **Readability**: Test files are cleaner and focus on test logic rather than setup boilerplate
