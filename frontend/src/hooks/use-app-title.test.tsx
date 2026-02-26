import { renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useParams } from "react-router";
import OptionService from "#/api/option-service/option-service.api";
import { useUserConversation } from "./query/use-user-conversation";
import { useAppTitle } from "./use-app-title";

const renderAppTitleHook = () =>
  renderHook(() => useAppTitle(), {
    wrapper: ({ children }) => (
      <QueryClientProvider client={new QueryClient()}>
        {children}
      </QueryClientProvider>
    ),
  });

vi.mock("./query/use-user-conversation");
vi.mock("react-router", async () => {
  const actual = await vi.importActual("react-router");
  return {
    ...actual,
    useParams: vi.fn(),
  };
});

describe("useAppTitle", () => {
  const getConfigSpy = vi.spyOn(OptionService, "getConfig");
  const mockUseUserConversation = vi.mocked(useUserConversation);
  const mockUseParams = vi.mocked(useParams);

  beforeEach(() => {
    // @ts-expect-error - only returning partial config for test
    mockUseUserConversation.mockReturnValue({ data: null });
    mockUseParams.mockReturnValue({});
  });

  it("should return 'OpenHands' if is OSS and NOT in /conversations", async () => {
    // @ts-expect-error - only returning partial config for test
    getConfigSpy.mockResolvedValue({
      app_mode: "oss",
    });

    const { result } = renderAppTitleHook();

    await waitFor(() => expect(result.current).toBe("OpenHands"));
  });

  it("should return 'OpenHands Cloud' if is SaaS and NOT in /conversations", async () => {
    // @ts-expect-error - only returning partial config for test
    getConfigSpy.mockResolvedValue({
      app_mode: "saas",
    });

    const { result } = renderAppTitleHook();

    await waitFor(() => expect(result.current).toBe("OpenHands Cloud"));
  });

  it("should return '{some title} | OpenHands' if is OSS and in /conversations", async () => {
    // @ts-expect-error - only returning partial config for test
    getConfigSpy.mockResolvedValue({ app_mode: "oss" });
    mockUseParams.mockReturnValue({ conversationId: "123" });
    mockUseUserConversation.mockReturnValue({
      // @ts-expect-error - only returning partial config for test
      data: { title: "My Conversation" },
    });

    const { result } = renderAppTitleHook();

    await waitFor(() =>
      expect(result.current).toBe("My Conversation | OpenHands"),
    );
  });

  it("should return '{some title} | OpenHands Cloud' if is SaaS and in /conversations", async () => {
    // @ts-expect-error - only returning partial config for test
    getConfigSpy.mockResolvedValue({ app_mode: "saas" });
    mockUseParams.mockReturnValue({ conversationId: "456" });
    mockUseUserConversation.mockReturnValue({
      // @ts-expect-error - only returning partial config for test
      data: { title: "Another Conversation Title" },
    });

    const { result } = renderAppTitleHook();

    await waitFor(() =>
      expect(result.current).toBe(
        "Another Conversation Title | OpenHands Cloud",
      ),
    );
  });

  it("should return app name while conversation is loading", async () => {
    // @ts-expect-error - only returning partial config for test
    getConfigSpy.mockResolvedValue({ app_mode: "oss" });
    mockUseParams.mockReturnValue({ conversationId: "123" });
    // @ts-expect-error - only returning partial config for test
    mockUseUserConversation.mockReturnValue({ data: undefined });

    const { result } = renderAppTitleHook();

    await waitFor(() => expect(result.current).toBe("OpenHands"));
  });
});
