import React from "react";
import { HttpError } from "@openhands/typescript-client";
import { useCloudCurrentUserId } from "#/hooks/query/use-cloud-current-user-id";
import { useAllCloudOrganizations } from "#/hooks/query/use-cloud-organizations";
import {
  act,
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { QueryClient } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AgentServerUIProviders } from "#/components/providers/agent-server-ui-providers";
import {
  __resetActiveStoreForTests,
  getActiveSelection,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import {
  getCloudOrganizations,
  getCloudOrganizationMe,
  getCurrentCloudApiKey,
} from "#/api/cloud/organization-service.api";
import SettingsService from "#/api/settings-service/settings-service.api";
import { DEFAULT_SETTINGS } from "#/services/settings";
import { useSettings } from "#/hooks/query/use-settings";

vi.mock("#/api/cloud/organization-service.api", () => ({
  getCloudOrganizations: vi.fn(),
  getCloudOrganizationMe: vi.fn(),
  getCurrentCloudApiKey: vi.fn(),
}));
vi.mock("#/api/settings-service/settings-service.api", () => ({
  default: { getSettings: vi.fn() },
}));
vi.mock("#/hooks/query/use-free-models", () => ({
  useHydrateFreeModels: vi.fn(),
}));
vi.mock("#/components/providers/telemetry-provider", () => ({
  TelemetryProvider: ({ children }: { children: React.ReactNode }) => children,
}));

const cloud = {
  id: "cloud",
  name: "Cloud",
  host: "https://cloud.example",
  kind: "cloud" as const,
  authMode: "cookie" as const,
  apiKey: "",
};

function IdentityConsumer() {
  const users = useCloudCurrentUserId();
  return <span aria-label="Current user">{users.cloud?.userId}</span>;
}

function SettingsConsumer() {
  const settings = useSettings();
  return settings.isSuccess ? (
    <input aria-label="Draft" defaultValue="saved" />
  ) : null;
}

function OrganizationConsumer() {
  const organizations = useAllCloudOrganizations();
  return <span>{organizations.cloud.isFetching ? "Fetching" : "Settled"}</span>;
}

beforeEach(() => {
  vi.stubEnv("VITE_BACKEND_BASE_URL", "http://localhost:9000");
  localStorage.clear();
  __resetActiveStoreForTests();
  vi.mocked(getCloudOrganizations).mockReset();
  vi.mocked(getCurrentCloudApiKey).mockReset();
  vi.mocked(getCloudOrganizationMe).mockResolvedValue({
    orgId: "chosen-org",
    userId: "test-user",
    role: null,
  });
  vi.mocked(SettingsService.getSettings).mockReset();
  vi.mocked(SettingsService.getSettings).mockResolvedValue(DEFAULT_SETTINGS);
  setRegisteredBackends([cloud]);
  setActiveSelection({ backendId: cloud.id, orgId: "removed-org" });
});

afterEach(() => {
  cleanup();
  vi.unstubAllEnvs();
});

describe("Cloud organization startup", () => {
  function renderConsumer() {
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false, retryDelay: 0 } },
    });
    render(
      <AgentServerUIProviders queryClient={client} resolveCloudOrganization>
        <SettingsConsumer />
        <OrganizationConsumer />
      </AgentServerUIProviders>,
    );
    return client;
  }

  it("keeps a valid page and unsaved input during a failed background refresh", async () => {
    setActiveSelection({ backendId: cloud.id, orgId: "chosen-org" });
    vi.mocked(getCloudOrganizations).mockResolvedValue({
      items: [
        { id: "chosen-org", name: "Chosen" },
        { id: "other-org", name: "Other" },
      ],
      currentOrgId: "other-org",
    });
    const client = renderConsumer();
    const input = await screen.findByLabelText("Draft");
    fireEvent.change(input, { target: { value: "unsaved" } });
    vi.mocked(getCloudOrganizations).mockRejectedValue(new Error("offline"));
    await act(async () => {
      await client.refetchQueries({ queryKey: ["cloud-organizations"] });
    });
    expect(screen.getByLabelText("Draft")).toBe(input);
    expect(input).toHaveValue("unsaved");
    expect(getActiveSelection()?.orgId).toBe("chosen-org");
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("keeps a saved selection usable after transient startup failures", async () => {
    vi.mocked(getCloudOrganizations).mockRejectedValue(new Error("offline"));
    renderConsumer();
    await screen.findByLabelText("Draft");
    expect(getActiveSelection()?.orgId).toBe("removed-org");
    await screen.findByText("Settled");
    expect(getCloudOrganizations).toHaveBeenCalledTimes(3);
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("offers Retry when startup fails without a saved selection", async () => {
    setActiveSelection({ backendId: cloud.id, orgId: null });
    vi.mocked(getCloudOrganizations).mockRejectedValue(new Error("offline"));
    renderConsumer();
    await screen.findByRole("alert");
    expect(SettingsService.getSettings).not.toHaveBeenCalled();
    vi.mocked(getCloudOrganizations).mockResolvedValue({
      items: [{ id: "restored-org", name: "Restored" }],
      currentOrgId: "restored-org",
    });
    fireEvent.click(screen.getByRole("button", { name: "BACKEND$AUTH_RETRY" }));
    await screen.findByLabelText("Draft");
    expect(getActiveSelection()?.orgId).toBe("restored-org");
  });

  it.each([401, 403])(
    "does not bypass an HTTP %s authorization failure",
    async (status) => {
      vi.mocked(getCloudOrganizations).mockRejectedValue(
        new HttpError(status, "Unauthorized"),
      );
      renderConsumer();
      await screen.findByRole("alert");
      expect(SettingsService.getSettings).not.toHaveBeenCalled();
      expect(getCloudOrganizations).toHaveBeenCalledTimes(1);
    },
  );

  it("leaves an embedded host's login visible without opting into recovery", () => {
    vi.mocked(getCloudOrganizations).mockReturnValue(new Promise(() => {}));
    render(
      <AgentServerUIProviders>
        <button>Host login</button>
      </AgentServerUIProviders>,
    );
    expect(
      screen.getByRole("button", { name: "Host login" }),
    ).toBeInTheDocument();
    expect(getCloudOrganizations).not.toHaveBeenCalled();
  });

  it("keeps cached identity while memberships refetch or temporarily fail", async () => {
    setActiveSelection({ backendId: cloud.id, orgId: "chosen-org" });
    vi.mocked(getCloudOrganizations).mockResolvedValue({
      items: [{ id: "chosen-org", name: "Chosen" }],
      currentOrgId: "chosen-org",
    });
    const client = new QueryClient({
      defaultOptions: { queries: { retryDelay: 0 } },
    });
    render(
      <AgentServerUIProviders queryClient={client} resolveCloudOrganization>
        <IdentityConsumer />
      </AgentServerUIProviders>,
    );
    await screen.findByText("test-user");
    let reject!: (error: Error) => void;
    vi.mocked(getCloudOrganizations).mockImplementation(
      () =>
        new Promise((_, fail) => {
          reject = fail;
        }),
    );
    let refetch!: Promise<void>;
    await act(async () => {
      refetch = client.refetchQueries({ queryKey: ["cloud-organizations"] });
    });
    expect(screen.getByLabelText("Current user")).toHaveTextContent(
      "test-user",
    );
    vi.mocked(getCloudOrganizations).mockRejectedValue(new Error("offline"));
    await act(async () => {
      reject(new Error("offline"));
      await refetch;
    });
    expect(screen.getByLabelText("Current user")).toHaveTextContent(
      "test-user",
    );
  });

  it("blocks org-scoped consumers when membership is empty and allows another backend", async () => {
    const local = {
      ...cloud,
      id: "local",
      name: "Local",
      kind: "local" as const,
      host: "http://localhost:9000",
    };
    setRegisteredBackends([cloud, local]);
    vi.mocked(getCloudOrganizations).mockResolvedValue({
      items: [],
      currentOrgId: null,
    });
    renderConsumer();
    await screen.findByRole("alert");
    await waitFor(() => expect(getActiveSelection()?.orgId).toBeNull());
    expect(SettingsService.getSettings).not.toHaveBeenCalled();
    fireEvent.change(screen.getByRole("combobox"), {
      target: { value: "local" },
    });
    await screen.findByLabelText("Draft");
    expect(getActiveSelection()?.backendId).toBe("local");
  });

  it("does not block a local backend on an inactive cloud's pending memberships", async () => {
    const local = {
      ...cloud,
      id: "local",
      kind: "local" as const,
      host: "http://localhost:9000",
    };
    setRegisteredBackends([cloud, local]);
    setActiveSelection({ backendId: local.id });
    vi.mocked(getCloudOrganizations).mockReturnValue(new Promise(() => {}));
    renderConsumer();
    await screen.findByLabelText("Draft");
    expect(getActiveSelection()?.backendId).toBe("local");
  });

  it("waits for the API key binding before mounting consumers", async () => {
    setRegisteredBackends([
      { ...cloud, authMode: "api-key", apiKey: "test-key" },
    ]);
    setActiveSelection({ backendId: cloud.id, orgId: "other-org" });
    vi.mocked(getCloudOrganizations).mockResolvedValue({
      items: [
        { id: "other-org", name: "Other" },
        { id: "bound-org", name: "Bound" },
      ],
      currentOrgId: "other-org",
    });
    let resolve!: (
      value: Awaited<ReturnType<typeof getCurrentCloudApiKey>>,
    ) => void;
    vi.mocked(getCurrentCloudApiKey).mockReturnValue(
      new Promise((done) => {
        resolve = done;
      }),
    );
    renderConsumer();
    await waitFor(() => expect(getCurrentCloudApiKey).toHaveBeenCalled());
    expect(SettingsService.getSettings).not.toHaveBeenCalled();
    await act(async () => resolve({ orgId: "bound-org", isLegacyKey: false }));
    await screen.findByLabelText("Draft");
    expect(getActiveSelection()?.orgId).toBe("bound-org");
  });

  it("gates a later backend switch until the newly selected cloud is resolved", async () => {
    const other = {
      ...cloud,
      id: "second-cloud",
      host: "https://second.example",
    };
    setRegisteredBackends([cloud, other]);
    let resolve!: (
      value: Awaited<ReturnType<typeof getCloudOrganizations>>,
    ) => void;
    vi.mocked(getCloudOrganizations).mockImplementation(async (backend) => {
      if (backend?.id === other.id)
        return new Promise((done) => {
          resolve = done;
        });
      return {
        items: [{ id: "first-org", name: "First" }],
        currentOrgId: "first-org",
      };
    });
    const requestedOrgs: (string | null | undefined)[] = [];
    vi.mocked(SettingsService.getSettings).mockImplementation(async () => {
      requestedOrgs.push(getActiveSelection()?.orgId);
      return DEFAULT_SETTINGS;
    });
    renderConsumer();
    await screen.findByLabelText("Draft");
    act(() =>
      setActiveSelection({ backendId: other.id, orgId: "removed-org" }),
    );
    expect(screen.queryByLabelText("Draft")).not.toBeInTheDocument();
    expect(requestedOrgs).toEqual(["first-org"]);
    await act(async () =>
      resolve({
        items: [{ id: "second-org", name: "Second" }],
        currentOrgId: "second-org",
      }),
    );
    await screen.findByLabelText("Draft");
    expect(requestedOrgs).toEqual(["first-org", "second-org"]);
  });

  it("does not request settings until the persisted organization is validated and repaired", async () => {
    let resolve!: (
      value: Awaited<ReturnType<typeof getCloudOrganizations>>,
    ) => void;
    vi.mocked(getCloudOrganizations).mockReturnValue(
      new Promise((done) => {
        resolve = done;
      }),
    );
    const requestedOrgs: (string | null | undefined)[] = [];
    vi.mocked(SettingsService.getSettings).mockImplementation(async () => {
      requestedOrgs.push(getActiveSelection()?.orgId);
      return DEFAULT_SETTINGS;
    });
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false, retryDelay: 0 } },
    });
    render(
      <AgentServerUIProviders queryClient={client} resolveCloudOrganization>
        <SettingsConsumer />
      </AgentServerUIProviders>,
    );
    await act(async () => {});
    expect(SettingsService.getSettings).not.toHaveBeenCalled();
    await act(async () =>
      resolve({
        items: [{ id: "current-org", name: "Current" }],
        currentOrgId: "current-org",
      }),
    );
    await screen.findByLabelText("Draft");
    expect(requestedOrgs).toEqual(["current-org"]);
    await waitFor(() =>
      expect(getActiveSelection()?.orgId).toBe("current-org"),
    );
  });
});
