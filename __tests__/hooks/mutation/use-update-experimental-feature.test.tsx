import React from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { act, renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { experimentalFeaturesService } from "#/api/experimental-features-service/experimental-features-service.api";
import { experimentalFeaturesQueryKey } from "#/hooks/query/use-experimental-features";
import { useUpdateExperimentalFeature } from "#/hooks/mutation/use-update-experimental-feature";

const backends = vi.hoisted(() => ({
  a: { id: "backend-a", host: "http://a", apiKey: "", kind: "local" },
  b: { id: "backend-b", host: "http://b", apiKey: "", kind: "local" },
}));
const active = vi.hoisted(() => ({ backend: backends.a }));

vi.mock("#/contexts/active-backend-context", () => ({
  useActiveBackend: () => ({ backend: active.backend, orgId: null }),
}));

const reactions = (enabled: boolean) => ({
  key: "conversation_reactions",
  label: "Conversation reactions",
  description: "Reactions.",
  enabled,
});

type Feature = ReturnType<typeof reactions>;

const deferred = <T,>() => {
  let resolve: (value: T) => void = () => {};
  const promise = new Promise<T>((r) => {
    resolve = r;
  });
  return { promise, resolve };
};

const setup = () => {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  for (const backend of [backends.a, backends.b]) {
    queryClient.setQueryData(experimentalFeaturesQueryKey(backend), [
      reactions(false),
    ]);
  }
  const wrapper = ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
  const hook = renderHook(() => useUpdateExperimentalFeature(), { wrapper });
  return { queryClient, ...hook };
};

describe("useUpdateExperimentalFeature", () => {
  afterEach(() => {
    active.backend = backends.a;
    vi.restoreAllMocks();
  });

  it("sends the switch to, and caches it for, the backend it was flipped on", async () => {
    const answer = deferred<Feature>();
    const update = vi
      .spyOn(experimentalFeaturesService, "updateExperimentalFeature")
      .mockReturnValue(answer.promise);
    vi.spyOn(
      experimentalFeaturesService,
      "listExperimentalFeatures",
    ).mockImplementation(async (backend) => [
      reactions(backend.id === "backend-a"),
    ]);
    const { queryClient, result, rerender } = setup();

    act(() => {
      result.current.mutate({ key: "conversation_reactions", enabled: true });
    });
    // The user switches to backend B before A answers.
    active.backend = backends.b;
    rerender();
    await act(async () => {
      answer.resolve(reactions(true));
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(update).toHaveBeenCalledWith(
      expect.objectContaining({ backend: backends.a, enabled: true }),
      expect.anything(),
    );
    expect(
      queryClient.getQueryData(experimentalFeaturesQueryKey(backends.a)),
    ).toEqual([reactions(true)]);
    expect(
      queryClient.getQueryData(experimentalFeaturesQueryKey(backends.b)),
    ).toEqual([reactions(false)]);
  });

  it("does not let an older read undo the switch", async () => {
    const staleRead = deferred<Feature[]>();
    vi.spyOn(
      experimentalFeaturesService,
      "listExperimentalFeatures",
    ).mockReturnValueOnce(staleRead.promise);
    vi.spyOn(
      experimentalFeaturesService,
      "listExperimentalFeatures",
    ).mockResolvedValue([reactions(true)]);
    vi.spyOn(
      experimentalFeaturesService,
      "updateExperimentalFeature",
    ).mockResolvedValue(reactions(true));
    const { queryClient, result } = setup();
    const key = experimentalFeaturesQueryKey(backends.a);
    // A read that started before the switch and answers after it.
    void queryClient
      .fetchQuery({
        queryKey: key,
        queryFn: () =>
          experimentalFeaturesService.listExperimentalFeatures(
            backends.a as never,
          ),
      })
      .catch(() => {});

    await act(async () => {
      result.current.mutate({ key: "conversation_reactions", enabled: true });
    });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    await act(async () => {
      staleRead.resolve([reactions(false)]);
    });

    await waitFor(() =>
      expect(queryClient.getQueryData(key)).toEqual([reactions(true)]),
    );
  });
});
