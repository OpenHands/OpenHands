import React from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, afterEach, expect, it, vi } from "vitest";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import {
  __resetActiveStoreForTests,
  setRegisteredBackends,
  setActiveSelection,
} from "#/api/backend-registry/active-store";
import { CodexAuthService } from "#/api/codex-auth-service";
import { CodexAuthCard } from "#/components/features/settings/codex-auth-card";

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string) =>
      ({
        SETTINGS$CODEX_SIGN_IN: "Sign in with ChatGPT",
        SETTINGS$CODEX_CONNECTED: "Connected to ChatGPT",
        BUTTON$DISCONNECT: "Disconnect",
        BUTTON$CANCEL: "Cancel",
        BUTTON$COPY: "Copy",
        BUTTON$COPIED: "Copied",
      })[key] ?? key,
  }),
}));

const disconnected = {
  connected: false,
  state: "disconnected" as const,
  expires_at: null,
};
const connected = {
  connected: true,
  state: "connected" as const,
  expires_at: null,
};
const challenge = {
  device_code: "opaque",
  user_code: "ABCD-EFGH",
  verification_uri: "https://auth.openai.com/codex/device",
  expires_at: Date.now() + 600000,
  interval_seconds: 1,
};

function renderCard() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <ActiveBackendProvider>
        <CodexAuthCard />
      </ActiveBackendProvider>
    </QueryClientProvider>,
  );
}

beforeEach(() => {
  __resetActiveStoreForTests();
  vi.spyOn(CodexAuthService, "getStatus").mockResolvedValue(disconnected);
  vi.spyOn(CodexAuthService, "start").mockResolvedValue(challenge);
  vi.spyOn(CodexAuthService, "poll").mockResolvedValue(connected);
  vi.spyOn(CodexAuthService, "cancel").mockResolvedValue({
    ...disconnected,
    state: "cancelled",
  });
  vi.spyOn(CodexAuthService, "logout").mockResolvedValue(disconnected);
  vi.spyOn(window, "open").mockReturnValue(null);
});
afterEach(() => {
  vi.restoreAllMocks();
  __resetActiveStoreForTests();
});

// @spec CAO-001 — Login publishes connected status without exposing credentials.
it("opens verification, displays a copyable code, polls and disconnects", async () => {
  const user = userEvent.setup();
  renderCard();
  const login = await screen.findByRole("button", {
    name: "Sign in with ChatGPT",
  });
  await waitFor(() => expect(login).toBeEnabled());
  await user.click(login);
  expect(await screen.findByText("ABCD-EFGH")).toBeInTheDocument();
  expect(window.open).toHaveBeenCalledWith(
    challenge.verification_uri,
    "_blank",
    "noopener,noreferrer",
  );
  await user.click(screen.getByTestId("copy-to-clipboard"));
  expect(await navigator.clipboard.readText()).toBe("ABCD-EFGH");
  const disconnect = await screen.findByRole(
    "button",
    { name: "Disconnect" },
    { timeout: 3000 },
  );
  expect(screen.getByText("Connected to ChatGPT")).toBeInTheDocument();
  await user.click(disconnect);
  expect(
    await screen.findByRole("button", { name: "Sign in with ChatGPT" }),
  ).toBeInTheDocument();
});

// @spec CAO-002 — Cancelling cannot accept a late connected response.
it("cancels an in-flight poll and ignores its late success", async () => {
  let resolvePoll!: (value: typeof connected) => void;
  vi.mocked(CodexAuthService.poll).mockImplementation(
    () =>
      new Promise((resolve) => {
        resolvePoll = resolve;
      }),
  );
  const user = userEvent.setup();
  renderCard();
  const login = await screen.findByRole("button", {
    name: "Sign in with ChatGPT",
  });
  await waitFor(() => expect(login).toBeEnabled());
  await user.click(login);
  await waitFor(() => expect(CodexAuthService.poll).toHaveBeenCalled(), {
    timeout: 3000,
  });
  await user.click(screen.getByRole("button", { name: "Cancel" }));
  resolvePoll(connected);
  await waitFor(() =>
    expect(
      screen.queryByRole("button", { name: "Disconnect" }),
    ).not.toBeInTheDocument(),
  );
  expect(CodexAuthService.cancel).toHaveBeenCalledWith(
    expect.anything(),
    "opaque",
  );
});

it("detects a saved login on reload and cancels an abandoned challenge", async () => {
  vi.mocked(CodexAuthService.getStatus).mockResolvedValue(connected);
  const view = renderCard();
  expect(
    await screen.findByRole("button", { name: "Disconnect" }),
  ).toBeInTheDocument();
  view.unmount();
  vi.mocked(CodexAuthService.getStatus).mockResolvedValue(disconnected);
  const next = renderCard();
  const login = await screen.findByRole("button", {
    name: "Sign in with ChatGPT",
  });
  await waitFor(() => expect(login).toBeEnabled());
  await userEvent.click(login);
  await screen.findByText("ABCD-EFGH");
  next.unmount();
  await waitFor(() =>
    expect(CodexAuthService.cancel).toHaveBeenCalledWith(
      expect.anything(),
      "opaque",
    ),
  );
});

// @spec CAO-002 — Switching backends cancels work on the previous server.
it("cancels the previous backend and ignores its late poll on a different server", async () => {
  const oldBackend = {
    id: "old",
    name: "Old",
    host: "http://old.example",
    apiKey: "old-key",
    kind: "local" as const,
  };
  const newBackend = {
    ...oldBackend,
    id: "new",
    name: "New",
    host: "http://new.example",
  };
  setRegisteredBackends([oldBackend, newBackend]);
  setActiveSelection({ backendId: "old" });
  let resolvePoll!: (value: typeof connected) => void;
  vi.mocked(CodexAuthService.poll).mockImplementation(
    () =>
      new Promise((resolve) => {
        resolvePoll = resolve;
      }),
  );
  renderCard();
  const login = await screen.findByRole("button", {
    name: "Sign in with ChatGPT",
  });
  await waitFor(() => expect(login).toBeEnabled());
  await userEvent.click(login);
  await waitFor(() => expect(CodexAuthService.poll).toHaveBeenCalled(), {
    timeout: 3000,
  });
  act(() => setActiveSelection({ backendId: "new" }));
  await waitFor(() =>
    expect(CodexAuthService.cancel).toHaveBeenCalledWith(oldBackend, "opaque"),
  );
  await act(async () => resolvePoll(connected));
  expect(
    screen.queryByRole("button", { name: "Disconnect" }),
  ).not.toBeInTheDocument();
  expect(
    screen.getByRole("button", { name: "Sign in with ChatGPT" }),
  ).toBeEnabled();
});
