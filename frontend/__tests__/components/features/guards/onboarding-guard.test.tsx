import { render } from "@testing-library/react";
import { createRoutesStub } from "react-router";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { OnboardingGuard } from "#/components/features/guards/onboarding-guard";

const mockNavigate = vi.fn();

vi.mock("react-router", async (importOriginal) => {
  const original = await importOriginal<typeof import("react-router")>();
  return {
    ...original,
    useNavigate: () => mockNavigate,
  };
});

const mockUseConfig = vi.fn();
const mockUseOnboardingStatus = vi.fn();

vi.mock("#/hooks/query/use-config", () => ({
  useConfig: () => mockUseConfig(),
}));

vi.mock("#/hooks/query/use-onboarding-status", () => ({
  useOnboardingStatus: () => mockUseOnboardingStatus(),
}));

const renderGuardAt = (initialEntry: string) => {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });

  const Stub = createRoutesStub([
    {
      path: "*",
      Component: () => (
        <OnboardingGuard>
          <div data-testid="children" />
        </OnboardingGuard>
      ),
    },
  ]);

  return render(
    <QueryClientProvider client={queryClient}>
      <Stub initialEntries={[initialEntry]} />
    </QueryClientProvider>,
  );
};

describe("OnboardingGuard returnTo preservation", () => {
  beforeEach(() => {
    mockNavigate.mockReset();
    mockUseConfig.mockReturnValue({
      data: { feature_flags: { enable_onboarding: true } },
    });
    mockUseOnboardingStatus.mockReturnValue({
      data: { should_complete_onboarding: true },
      isLoading: false,
    });
  });

  it("preserves the originally requested path as a returnTo query parameter", async () => {
    renderGuardAt("/conversations/abc-123");

    await vi.waitFor(() => {
      expect(mockNavigate).toHaveBeenCalledWith(
        `/onboarding?returnTo=${encodeURIComponent("/conversations/abc-123")}`,
        { replace: true },
      );
    });
  });

  it("preserves search params alongside the path in returnTo", async () => {
    renderGuardAt("/conversations/abc?foo=bar&baz=qux");

    await vi.waitFor(() => {
      expect(mockNavigate).toHaveBeenCalledWith(
        `/onboarding?returnTo=${encodeURIComponent("/conversations/abc?foo=bar&baz=qux")}`,
        { replace: true },
      );
    });
  });

  it("does not double-encode pre-encoded characters in search params", async () => {
    // ``search`` from useLocation() is already percent-encoded. Without
    // decoding it first, ``encodeURIComponent`` would re-encode the ``%``
    // as ``%25``, causing double-encoding (e.g. ``%20`` → ``%2520``).
    // With the fix, ``%20`` in the original search is decoded to a space
    // and then re-encoded once, yielding ``%20`` (not ``%2520``) in the
    // ``returnTo`` parameter.
    renderGuardAt("/conversations/abc?tab=user%20profile");

    await vi.waitFor(() => {
      expect(mockNavigate).toHaveBeenCalledWith(
        `/onboarding?returnTo=${encodeURIComponent("/conversations/abc?tab=user profile")}`,
        { replace: true },
      );
    });
  });

  it("does not append a returnTo when the originally requested path is /", async () => {
    renderGuardAt("/");

    await vi.waitFor(() => {
      expect(mockNavigate).toHaveBeenCalledWith("/onboarding", {
        replace: true,
      });
    });
  });

  it("does not redirect when already on /onboarding", async () => {
    renderGuardAt("/onboarding");

    // Allow the effect to settle.
    await new Promise((resolve) => {
      setTimeout(resolve, 0);
    });

    expect(mockNavigate).not.toHaveBeenCalled();
  });

  it("does not redirect when the enable_onboarding feature flag is off", async () => {
    mockUseConfig.mockReturnValue({
      data: { feature_flags: { enable_onboarding: false } },
    });

    renderGuardAt("/conversations/abc");

    await new Promise((resolve) => {
      setTimeout(resolve, 0);
    });

    expect(mockNavigate).not.toHaveBeenCalled();
  });

  it("does not redirect when onboarding is already complete", async () => {
    mockUseOnboardingStatus.mockReturnValue({
      data: { should_complete_onboarding: false },
      isLoading: false,
    });

    renderGuardAt("/conversations/abc");

    await new Promise((resolve) => {
      setTimeout(resolve, 0);
    });

    expect(mockNavigate).not.toHaveBeenCalled();
  });
});
