import React from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { __resetActiveStoreForTests } from "#/api/backend-registry/active-store";
import {
  ActiveBackendProvider,
  useActiveBackendContext,
} from "#/contexts/active-backend-context";
import {
  NavigationProvider,
  type NavigationContextValue,
} from "#/contexts/navigation-context";
import { BackendFormModal } from "#/components/features/backends/backend-form-modal";

const getServerInfoMock = vi.hoisted(() => vi.fn());

vi.mock("@openhands/typescript-client/clients", () => ({
  ServerClient: vi.fn(function ServerClientMock() {
    return {
      getServerInfo: getServerInfoMock,
    };
  }),
}));

function renderWithProviders(
  ui: React.ReactElement,
  navigation?: NavigationContextValue,
) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <ActiveBackendProvider>
        {navigation ? (
          <NavigationProvider value={navigation}>{ui}</NavigationProvider>
        ) : (
          ui
        )}
      </ActiveBackendProvider>
    </QueryClientProvider>,
  );
}

function TestSeed({
  onMount,
  children,
}: {
  onMount: (ctx: ReturnType<typeof useActiveBackendContext>) => void;
  children: React.ReactNode;
}) {
  const ctx = useActiveBackendContext();
  React.useEffect(() => {
    onMount(ctx);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  return children as React.ReactElement;
}

beforeEach(() => {
  window.localStorage.clear();
  getServerInfoMock.mockReset();
  getServerInfoMock.mockResolvedValue({ version: "1.28.0" });
  __resetActiveStoreForTests();
});

afterEach(() => {
  window.localStorage.clear();
  vi.unstubAllEnvs();
  delete (window as unknown as Record<string, unknown>)
    .__AGENT_CANVAS_LOCK_TO_CLOUD__;
  __resetActiveStoreForTests();
});

describe("BackendFormModal – edit mode (BackendForm entry point)", () => {
  it("pre-fills fields with the backend data passed as prop", () => {
    renderWithProviders(
      <BackendFormModal
        mode="edit"
        backend={{
          id: "seeded-id",
          name: "My Server",
          host: "http://localhost:9000",
          apiKey: "sk-old",
          kind: "local",
        }}
        onClose={vi.fn()}
      />,
    );

    expect(screen.getByTestId("edit-backend-name")).toHaveValue("My Server");
    expect(screen.getByTestId("edit-backend-host")).toHaveValue(
      "http://localhost:9000",
    );
    expect(screen.getByTestId("edit-backend-api-key")).toHaveValue("sk-old");
  });

  it("disables Save when name is cleared", async () => {
    renderWithProviders(
      <BackendFormModal
        mode="edit"
        backend={{
          id: "seeded-id",
          name: "My Server",
          host: "http://localhost:9000",
          apiKey: "",
          kind: "local",
        }}
        onClose={vi.fn()}
      />,
    );

    const submit = screen.getByTestId(
      "edit-backend-submit",
    ) as HTMLButtonElement;
    expect(submit).not.toBeDisabled();

    const user = userEvent.setup();
    const nameInput = screen.getByTestId("edit-backend-name");
    await user.clear(nameInput);

    expect(submit).toBeDisabled();
  });

  it("calls updateBackend and onClose on successful submit", async () => {
    let backendId = "";
    const onClose = vi.fn();
    renderWithProviders(
      <TestSeed
        onMount={(ctx) => {
          backendId = ctx.addBackend({
            name: "Local Seed",
            host: "http://localhost:9000",
            apiKey: "sk-old",
            kind: "local",
          }).id;
        }}
      >
        <BackendFormModal
          mode="edit"
          backend={{
            id: backendId,
            name: "Local Seed",
            host: "http://localhost:9000",
            apiKey: "sk-old",
            kind: "local",
          }}
          onClose={onClose}
        />
      </TestSeed>,
    );

    await waitFor(() => {
      expect(screen.getByTestId("edit-backend-name")).toHaveValue("Local Seed");
    });

    const user = userEvent.setup();

    // Verify the submit button is enabled with pre-filled values
    const submitBtn = screen.getByTestId("edit-backend-submit");
    expect(submitBtn).not.toBeDisabled();

    await user.click(submitBtn);

    await waitFor(() => expect(onClose).toHaveBeenCalled());

    const stored = JSON.parse(
      window.localStorage.getItem("openhands-backends") ?? "[]",
    );
    const updated = stored.find((b: { id: string }) => b.id === backendId);
    expect(updated).toMatchObject({
      id: backendId,
      name: "Local Seed",
      host: "http://localhost:9000",
      kind: "local",
    });
  });

  it("shows connection error and keeps modal open when probe fails", async () => {
    // Default: health check succeeds (otherwise the modal won't render cleanly).
    // Override just before the submit action so the submit-time probe fails.
    let backendId = "";
    renderWithProviders(
      <TestSeed
        onMount={(ctx) => {
          backendId = ctx.addBackend({
            name: "Offline Server",
            host: "http://localhost:9999",
            apiKey: "sk-key",
            kind: "local",
          }).id;
        }}
      >
        <BackendFormModal
          mode="edit"
          backend={{
            id: backendId,
            name: "Offline Server",
            host: "http://localhost:9999",
            apiKey: "sk-key",
            kind: "local",
          }}
          onClose={vi.fn()}
        />
      </TestSeed>,
    );

    await waitFor(() => {
      expect(screen.getByTestId("edit-backend-name")).toHaveValue(
        "Offline Server",
      );
    });

    const user = userEvent.setup();
    const nameInput = screen.getByTestId("edit-backend-name");
    await user.clear(nameInput);
    await user.type(nameInput, "Retry Server");

    // Now make the submit-time probe fail (health check already consumed the default mock).
    getServerInfoMock.mockRejectedValueOnce(new Error("Connection refused"));

    await user.click(screen.getByTestId("edit-backend-submit"));

    expect(await screen.findByTestId("edit-backend-error")).toHaveTextContent(
      "Connection refused",
    );
  });

  it("preserves cloud kind when editing a cloud backend", async () => {
    let backendId = "";
    const onClose = vi.fn();
    renderWithProviders(
      <TestSeed
        onMount={(ctx) => {
          backendId = ctx.addBackend({
            name: "OpenHands Cloud",
            host: "https://app.all-hands.dev",
            apiKey: "sk-cloud",
            kind: "cloud",
          }).id;
        }}
      >
        <BackendFormModal
          mode="edit"
          backend={{
            id: backendId,
            name: "OpenHands Cloud",
            host: "https://app.all-hands.dev",
            apiKey: "sk-cloud",
            kind: "cloud",
          }}
          onClose={onClose}
        />
      </TestSeed>,
    );

    await waitFor(() => {
      expect(screen.getByTestId("edit-backend-name")).toHaveValue(
        "OpenHands Cloud",
      );
    });

    const user = userEvent.setup();
    const nameInput = screen.getByTestId("edit-backend-name");
    await user.clear(nameInput);
    await user.type(nameInput, "Renamed Cloud");

    const apiKeyInput = screen.getByTestId("edit-backend-api-key");
    await user.clear(apiKeyInput);
    await user.type(apiKeyInput, "new-token");

    await user.click(screen.getByTestId("edit-backend-submit"));

    await waitFor(() => expect(onClose).toHaveBeenCalled());

    const stored = JSON.parse(
      window.localStorage.getItem("openhands-backends") ?? "[]",
    );
    const updated = stored.find((b: { id: string }) => b.id === backendId);
    expect(updated).toMatchObject({
      id: backendId,
      kind: "cloud",
    });
  });

  it("locks add mode to Cloud login when VITE_LOCK_TO_CLOUD is set", () => {
    vi.stubEnv("VITE_LOCK_TO_CLOUD", "https://cloud.example.com");

    renderWithProviders(<BackendFormModal mode="add" onClose={vi.fn()} />);

    expect(screen.getByTestId("add-backend-cloud-title")).toBeVisible();
    expect(screen.getByTestId("add-backend-login-button")).toBeVisible();
    expect(screen.queryByTestId("add-backend-host")).not.toBeInTheDocument();
    expect(screen.queryByTestId("add-backend-api-key")).not.toBeInTheDocument();
    expect(
      screen.queryByTestId("add-backend-advanced-toggle"),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByTestId("add-backend-cloud-host"),
    ).not.toBeInTheDocument();
  });
});
