import React from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { AcpPermissionPanel } from "#/components/features/chat/acp-permission-panel";
import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";

const mocks = vi.hoisted(() => ({
  list: vi.fn(),
  decide: vi.fn(),
  client: vi.fn(),
  backend: { id: "local", kind: "local", connectionRevision: 1 },
  orgId: null as string | null,
}));
vi.mock("@openhands/typescript-client/clients", () => ({
  ConversationClient: class {
    constructor(options: unknown) {
      mocks.client(options);
    }
    listACPPermissions = mocks.list;
    respondToACPPermission = mocks.decide;
  },
}));
vi.mock("#/contexts/active-backend-context", () => ({
  useActiveBackend: () => ({ backend: mocks.backend, orgId: mocks.orgId }),
}));
vi.mock("#/api/agent-server-client-options", () => ({
  getAgentServerClientOptions: (target: unknown) => target,
}));
vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: (key: string) => key }),
}));

const conversation = {
  id: "conversation-a",
  agent_kind: "acp",
  sandbox_status: "RUNNING",
  conversation_url: "https://runtime-a/api/conversations/conversation-a",
  session_api_key: "fixture-key",
} as AppConversation;
const request = () => ({
  request_id: "request-a",
  title: "<script>operation</script>",
  details: '{"command":"review me"}',
  details_truncated: false,
  expires_at: new Date(Date.now() + 60000).toISOString(),
  options: [
    { id: "allow", name: "Allow once", kind: "allow_once" },
    { id: "reject", name: "Reject once", kind: "reject_once" },
  ],
});
function setup(value = conversation) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  const wrap = (current: AppConversation) => (
    <QueryClientProvider client={queryClient}>
      <AcpPermissionPanel conversation={current} />
    </QueryClientProvider>
  );
  return { ...render(wrap(value)), queryClient, wrap };
}
beforeEach(() => {
  vi.clearAllMocks();
  mocks.backend = { id: "local", kind: "local", connectionRevision: 1 };
  mocks.list.mockResolvedValue({ items: [request()] });
  mocks.decide.mockResolvedValue({ success: true });
});

describe("ACP permission panel", () => {
  it("waits for an explicit offered choice and sends the rejecting second option", async () => {
    const { queryClient } = setup();
    const reject = await screen.findByRole("button", { name: "Reject once" });
    expect(screen.getByText("<script>operation</script>")).toBeInTheDocument();
    expect(screen.getByText('{"command":"review me"}')).toBeInTheDocument();
    expect(mocks.decide).not.toHaveBeenCalled();
    fireEvent.click(reject);
    await waitFor(() =>
      expect(mocks.decide).toHaveBeenCalledWith("conversation-a", "request-a", {
        option_id: "reject",
      }),
    );
    expect(mocks.client).toHaveBeenCalledWith(
      expect.objectContaining({
        conversationUrl: conversation.conversation_url,
        sessionApiKey: "fixture-key",
      }),
    );
    expect(
      JSON.stringify(
        queryClient
          .getQueryCache()
          .getAll()
          .map((query) => query.queryKey),
      ),
    ).not.toContain("fixture-key");
  });
  it("disables incomplete approvals and expired decisions", async () => {
    mocks.list.mockResolvedValue({
      items: [{ ...request(), details_truncated: true }],
    });
    const { rerender, wrap } = setup();
    expect(
      await screen.findByRole("button", { name: "Allow once" }),
    ).toBeDisabled();
    expect(screen.getByRole("button", { name: "Reject once" })).toBeEnabled();
    mocks.list.mockResolvedValue({
      items: [{ ...request(), expires_at: "2000-01-01T00:00:00Z" }],
    });
    rerender(wrap({ ...conversation, id: "expired" }));
    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: "Reject once" }),
      ).toBeDisabled(),
    );
    expect(mocks.decide).not.toHaveBeenCalled();
  });
  it("shows failures and refreshes stale requests without granting them", async () => {
    mocks.decide.mockRejectedValue(new Error("stale request"));
    setup();
    fireEvent.click(await screen.findByRole("button", { name: "Allow once" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("ERROR$GENERIC");
    mocks.list.mockResolvedValue({ items: [] });
    fireEvent.click(screen.getByRole("button", { name: "BUTTON$REFRESH" }));
    await waitFor(() =>
      expect(
        screen.queryByRole("button", { name: "Allow once" }),
      ).not.toBeInTheDocument(),
    );
    expect(mocks.decide).toHaveBeenCalledTimes(1);
  });
  it("does not expose previous requests after conversation, backend or cloud readiness changes", async () => {
    const { rerender, wrap } = setup();
    await screen.findByRole("button", { name: "Allow once" });
    mocks.list.mockResolvedValue({ items: [] });
    rerender(wrap({ ...conversation, session_api_key: "rotated-fixture-key" }));
    await waitFor(() =>
      expect(mocks.client).toHaveBeenCalledWith(
        expect.objectContaining({ sessionApiKey: "rotated-fixture-key" }),
      ),
    );
    expect(
      screen.queryByRole("button", { name: "Allow once" }),
    ).not.toBeInTheDocument();
    mocks.backend = { id: "cloud", kind: "cloud", connectionRevision: 2 };
    mocks.list.mockResolvedValue({ items: [] });
    rerender(
      wrap({
        ...conversation,
        id: "other",
        conversation_url: "https://runtime-b/api/conversations/other",
        session_api_key: "other-fixture-key",
      }),
    );
    await waitFor(() => expect(mocks.list).toHaveBeenCalledWith("other"));
    expect(
      screen.queryByRole("button", { name: "Allow once" }),
    ).not.toBeInTheDocument();
    const calls = mocks.list.mock.calls.length;
    rerender(wrap({ ...conversation, sandbox_status: "PAUSED" }));
    expect(screen.queryByRole("region")).not.toBeInTheDocument();
    expect(mocks.list).toHaveBeenCalledTimes(calls);
    expect(mocks.decide).not.toHaveBeenCalled();
  });

  it("recovers a failed fetch and sends explicit cancellation only once", async () => {
    mocks.list.mockRejectedValue(new Error("offline"));
    let complete!: (value: unknown) => void;
    mocks.decide.mockImplementation(
      () =>
        new Promise((resolve) => {
          complete = resolve;
        }),
    );
    setup();
    await screen.findByRole("alert");
    expect(
      screen.queryByRole("button", { name: "Allow once" }),
    ).not.toBeInTheDocument();
    mocks.list.mockResolvedValue({ items: [request()] });
    fireEvent.click(screen.getByRole("button", { name: "BUTTON$REFRESH" }));
    fireEvent.click(
      await screen.findByRole("button", { name: "BUTTON$CANCEL" }),
    );
    await waitFor(() =>
      expect(mocks.decide).toHaveBeenCalledWith("conversation-a", "request-a", {
        option_id: null,
      }),
    );
    expect(screen.getByRole("button", { name: "Allow once" })).toBeDisabled();
    fireEvent.click(screen.getByRole("button", { name: "BUTTON$CANCEL" }));
    expect(mocks.decide).toHaveBeenCalledTimes(1);
    mocks.list.mockResolvedValue({ items: [] });
    complete({ success: true });
    await waitFor(() =>
      expect(
        screen.queryByRole("button", { name: "Allow once" }),
      ).not.toBeInTheDocument(),
    );
  });
});
