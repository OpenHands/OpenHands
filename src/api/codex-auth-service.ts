import { CodexAuthClient } from "@openhands/typescript-client/clients";
import type { Backend } from "./backend-registry/types";
import { getAgentServerClientOptions } from "./agent-server-client-options";

export type {
  CodexAuthStatus,
  CodexDeviceChallenge,
} from "@openhands/typescript-client";

// Capture the backend for every operation, including cancellation on unmount.
async function withClient<T>(
  backend: Backend,
  operation: (client: CodexAuthClient) => Promise<T>,
): Promise<T> {
  const client = new CodexAuthClient({
    ...getAgentServerClientOptions({ host: backend.host }),
    apiKey: backend.apiKey ?? undefined,
  });
  try {
    return await operation(client);
  } finally {
    client.close();
  }
}

export const CodexAuthService = {
  getStatus: (backend: Backend) => withClient(backend, (c) => c.getStatus()),
  start: (backend: Backend) => withClient(backend, (c) => c.startDeviceLogin()),
  poll: (backend: Backend, handle: string) =>
    withClient(backend, (c) => c.pollDeviceLogin(handle)),
  cancel: (backend: Backend, handle: string) =>
    withClient(backend, (c) => c.cancelDeviceLogin(handle)),
  logout: (backend: Backend) => withClient(backend, (c) => c.logout()),
};
