import React from "react";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useSubmitOnboarding } from "#/hooks/mutation/use-submit-onboarding";
import { openHands } from "#/api/open-hands-axios";

const mockNavigate = vi.fn();

vi.mock("react-router", () => ({
  useNavigate: () => mockNavigate,
}));

vi.mock("#/api/open-hands-axios", () => ({
  openHands: {
    post: vi.fn(),
  },
}));

vi.mock("#/utils/custom-toast-handlers", () => ({
  displayErrorToast: vi.fn(),
}));

const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });

  return function Wrapper({ children }: { children: React.ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    );
  };
};

describe("useSubmitOnboarding", () => {
  beforeEach(() => {
    mockNavigate.mockClear();
    vi.mocked(openHands.post).mockReset();
    vi.stubGlobal("location", {
      href: "",
      origin: "https://pr-254.staging.all-hands.dev",
      assign: vi.fn(),
    });
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("hard reloads same-origin absolute Canvas redirects", async () => {
    vi.mocked(openHands.post).mockResolvedValue({
      data: {
        redirect_url: "https://pr-254.staging.all-hands.dev/canvas",
      },
    });

    const { result } = renderHook(() => useSubmitOnboarding(), {
      wrapper: createWrapper(),
    });

    result.current.mutate({ selections: {}, returnTo: "/" });

    await waitFor(() => {
      expect(window.location.assign).toHaveBeenCalledWith("/canvas");
    });
    expect(mockNavigate).not.toHaveBeenCalledWith("/canvas", { replace: true });
  });
});
