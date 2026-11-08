import React from "react";
import { describe, expect, it, vi } from "vitest";
import { act, renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { experimentalFeaturesService } from "#/api/experimental-features-service/experimental-features-service.api";
import { EXPERIMENTAL_FEATURES_QUERY_KEY } from "#/hooks/query/use-experimental-features";
import { useUpdateExperimentalFeature } from "#/hooks/mutation/use-update-experimental-feature";

const active = vi.hoisted(() => ({ id: "backend-a" }));

vi.mock("#/contexts/active-backend-context", () => ({
  useActiveBackend: () => ({
    backend: { id: active.id, kind: "local" },
    orgId: null,
  }),
}));

const reactions = (enabled: boolean) => ({
  key: "conversation_reactions",
  label: "Conversation reactions",
  description: "Reactions.",
  default: false,
  enabled,
});

describe("useUpdateExperimentalFeature", () => {
  it("writes the answer to the backend the switch was sent to", async () => {
    let answer: (value: ReturnType<typeof reactions>) => void = () => {};
    vi.spyOn(
      experimentalFeaturesService,
      "updateExperimentalFeature",
    ).mockReturnValue(
      new Promise((resolve) => {
        answer = resolve;
      }),
    );
    const queryClient = new QueryClient();
    queryClient.setQueryData(
      [EXPERIMENTAL_FEATURES_QUERY_KEY, "backend-a"],
      [reactions(false)],
    );
    queryClient.setQueryData(
      [EXPERIMENTAL_FEATURES_QUERY_KEY, "backend-b"],
      [reactions(false)],
    );
    const wrapper = ({ children }: { children: React.ReactNode }) => (
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    );
    const { result, rerender } = renderHook(
      () => useUpdateExperimentalFeature(),
      { wrapper },
    );

    act(() => {
      result.current.mutate({ key: "conversation_reactions", enabled: true });
    });
    // The person switches to backend B before A answers.
    active.id = "backend-b";
    rerender();
    await act(async () => {
      answer(reactions(true));
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(
      queryClient.getQueryData([EXPERIMENTAL_FEATURES_QUERY_KEY, "backend-a"]),
    ).toEqual([reactions(true)]);
    expect(
      queryClient.getQueryData([EXPERIMENTAL_FEATURES_QUERY_KEY, "backend-b"]),
    ).toEqual([reactions(false)]);
  });
});
