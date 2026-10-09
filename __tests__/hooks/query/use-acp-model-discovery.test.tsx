import React from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { useAcpModelDiscovery } from "#/hooks/query/use-acp-model-discovery";
import {
  readRememberedAcpModels,
  rememberAcpModels,
} from "#/utils/remembered-acp-models";

const backendMock = vi.hoisted(() => ({
  current: {
    backend: { id: "local-1", kind: "local" as "local" | "cloud" },
    orgId: null as string | null,
  },
}));
vi.mock("#/contexts/active-backend-context", () => ({
  useActiveBackend: () => backendMock.current,
}));

vi.mock("#/hooks/query/use-get-secrets", () => ({
  useSearchSecrets: () => ({
    data: [{ name: "PI_AUTH_JSON" }, { name: "ANTHROPIC_API_KEY" }],
    isLoading: false,
  }),
}));

const discoverModels = vi.hoisted(() => vi.fn());
vi.mock("#/api/acp-service/acp-service.api", () => ({
  default: {
    discoverModels: (...args: unknown[]) => discoverModels(...args),
  },
}));

function wrapper({ children }: { children: React.ReactNode }) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}

const PI_DISCOVERY = {
  agent_name: "pi-acp",
  agent_version: "0.0.33",
  current_model_id: "anthropic/claude-opus-4-8",
  available_models: [
    { model_id: "anthropic/claude-opus-4-8", name: "Claude Opus 4.8" },
    { model_id: "anthropic/claude-sonnet-5", name: null },
  ],
  supports_runtime_model_switch: true,
  error: null,
};

beforeEach(() => {
  vi.clearAllMocks();
  window.localStorage.clear();
  backendMock.current = {
    backend: { id: "local-1", kind: "local" },
    orgId: null,
  };
});

describe("useAcpModelDiscovery", () => {
  it("asks the local agent-server with every saved secret", async () => {
    discoverModels.mockResolvedValue(PI_DISCOVERY);

    const { result } = renderHook(() => useAcpModelDiscovery("pi"), {
      wrapper,
    });

    await waitFor(() =>
      expect(result.current.defaultModelId).toBe("anthropic/claude-opus-4-8"),
    );
    expect(discoverModels).toHaveBeenCalledWith("pi", [
      "ANTHROPIC_API_KEY",
      "PI_AUTH_JSON",
    ]);
    const models = [
      { id: "anthropic/claude-opus-4-8", label: "Claude Opus 4.8" },
      { id: "anthropic/claude-sonnet-5", label: "anthropic/claude-sonnet-5" },
    ];
    expect(result.current.models).toEqual(models);
    expect(result.current.source).toBe("live");
    expect(readRememberedAcpModels("local-1", "pi")).toEqual(models);
  });

  it("asks with only the secrets the launch would receive", async () => {
    discoverModels.mockResolvedValue(PI_DISCOVERY);

    renderHook(
      () =>
        useAcpModelDiscovery("pi", { secretRefs: ["PI_AUTH_JSON", "GONE"] }),
      { wrapper },
    );

    await waitFor(() =>
      expect(discoverModels).toHaveBeenCalledWith("pi", ["PI_AUTH_JSON"]),
    );
  });

  it("sends no secrets when the profile allows none", async () => {
    discoverModels.mockResolvedValue(PI_DISCOVERY);

    renderHook(() => useAcpModelDiscovery("pi", { secretRefs: [] }), {
      wrapper,
    });

    await waitFor(() => expect(discoverModels).toHaveBeenCalledWith("pi", []));
  });

  it("has no models when the server cannot answer and none were seen", async () => {
    discoverModels.mockRejectedValue(new Error("404"));

    const { result } = renderHook(() => useAcpModelDiscovery("pi"), {
      wrapper,
    });

    await waitFor(() => expect(discoverModels).toHaveBeenCalled());
    await waitFor(() => expect(result.current.isDiscovering).toBe(false));
    expect(result.current.discovery).toBeNull();
    expect(result.current.models).toEqual([]);
    expect(result.current.source).toBe("none");
  });

  it("does not ask on a cloud backend", () => {
    backendMock.current = {
      backend: { id: "cloud-1", kind: "cloud" },
      orgId: null,
    };

    const { result } = renderHook(() => useAcpModelDiscovery("claude-code"), {
      wrapper,
    });

    expect(discoverModels).not.toHaveBeenCalled();
    expect(result.current.models).toEqual([]);
    expect(result.current.source).toBe("none");
  });

  it("offers the list the agent last reported on this backend", () => {
    backendMock.current = {
      backend: { id: "cloud-1", kind: "cloud" },
      orgId: null,
    };
    const models = [{ id: "sonnet", label: "Sonnet" }];
    rememberAcpModels("cloud-1", "claude-code", models);
    rememberAcpModels("cloud-2", "codex", [{ id: "gpt-5.5", label: "GPT" }]);

    const { result } = renderHook(() => useAcpModelDiscovery("claude-code"), {
      wrapper,
    });

    expect(result.current.models).toEqual(models);
    expect(result.current.source).toBe("remembered");
    expect(result.current.defaultModelId).toBeNull();
  });

  it("picks up a list remembered after it mounted", () => {
    backendMock.current = {
      backend: { id: "cloud-1", kind: "cloud" },
      orgId: null,
    };
    const { result } = renderHook(() => useAcpModelDiscovery("claude-code"), {
      wrapper,
    });
    expect(result.current.models).toEqual([]);

    const models = [{ id: "sonnet", label: "Sonnet" }];
    act(() => rememberAcpModels("cloud-1", "claude-code", models));

    expect(result.current.models).toEqual(models);
    expect(result.current.source).toBe("remembered");
  });

  it("does not ask without a provider", () => {
    renderHook(() => useAcpModelDiscovery(null), { wrapper });

    expect(discoverModels).not.toHaveBeenCalled();
  });
});
