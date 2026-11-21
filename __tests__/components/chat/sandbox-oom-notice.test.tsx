import {
  act,
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import { batchGetCloudSandboxes } from "#/api/cloud/sandbox-service.api";
import { SandboxOomNotice } from "#/components/features/chat/sandbox-oom-notice";

vi.mock("#/api/cloud/sandbox-service.api", () => ({
  batchGetCloudSandboxes: vi.fn(),
}));

const message = "CHAT_INTERFACE$SANDBOX_OUT_OF_MEMORY";
const sandbox = {
  id: "sandbox-1",
  created_by_user_id: "user-1",
  sandbox_spec_id: "agent-server",
  session_api_key: null,
  exposed_urls: [],
  created_at: "2026-11-16T10:00:00Z",
  status: "RUNNING" as const,
};
const oom = {
  reason: "OOMKilled",
  exit_code: 137,
  finished_at: "2026-11-16T11:17:32Z",
};
let client: QueryClient;

beforeEach(() => {
  window.localStorage.clear();
  __resetActiveStoreForTests();
  setRegisteredBackends([
    {
      id: "beta",
      name: "Beta",
      kind: "cloud",
      host: "https://beta.example.com",
      apiKey: "test-key",
    },
  ]);
  setActiveSelection({ backendId: "beta" });
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  vi.mocked(batchGetCloudSandboxes).mockResolvedValue([
    { ...sandbox, last_termination: oom },
  ]);
});

afterEach(() => {
  cleanup();
  client.clear();
  vi.clearAllMocks();
  __resetActiveStoreForTests();
});

function mount(sandboxId = sandbox.id) {
  return render(
    <QueryClientProvider client={client}>
      <ActiveBackendProvider>
        <SandboxOomNotice sandboxId={sandboxId} />
      </ActiveBackendProvider>
    </QueryClientProvider>,
  );
}

it("explains an OOM after recovery, keeps dismissal on refresh, and shows a later OOM", async () => {
  mount();
  await screen.findByText(message);
  fireEvent.click(screen.getByTestId("error-message-banner-dismiss"));

  await act(() => client.refetchQueries());
  expect(screen.queryByText(message)).not.toBeInTheDocument();

  vi.mocked(batchGetCloudSandboxes).mockResolvedValue([
    {
      ...sandbox,
      last_termination: { ...oom, finished_at: "2026-11-16T12:00:00Z" },
    },
  ]);
  await act(() => client.refetchQueries());
  expect(await screen.findByText(message)).toBeVisible();
});

it.each([undefined, { ...oom, reason: "Error" }])(
  "does not infer OOM from an unexplained failure or exit 137: %j",
  async (termination) => {
    vi.mocked(batchGetCloudSandboxes).mockResolvedValue([
      { ...sandbox, status: "ERROR", last_termination: termination },
    ]);
    mount();
    await waitFor(() => expect(client.isFetching()).toBe(0));
    expect(screen.queryByText(message)).not.toBeInTheDocument();
  },
);

it("does not carry an OOM notice to a different sandbox", async () => {
  const view = mount();
  await screen.findByText(message);
  view.unmount();
  vi.mocked(batchGetCloudSandboxes).mockResolvedValue([
    { ...sandbox, id: "sandbox-2" },
  ]);
  mount("sandbox-2");
  await waitFor(() => expect(client.isFetching()).toBe(0));
  expect(screen.queryByText(message)).not.toBeInTheDocument();
});
