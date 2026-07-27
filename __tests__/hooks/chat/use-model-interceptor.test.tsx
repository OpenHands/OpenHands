import React from "react";
import { act, renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { beforeEach, describe, expect, it, vi } from "vitest";
import ProfilesService from "#/api/profiles-service/profiles-service.api";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import type { Backend } from "#/api/backend-registry/types";
import { useModelInterceptor } from "#/hooks/chat/use-model-interceptor";
import { useEventStore } from "#/stores/use-event-store";
import { useModelStore } from "#/stores/model-store";

const { mockSwitchAndLog, mockDisplayErrorToast } = vi.hoisted(() => ({
  mockSwitchAndLog: vi.fn(),
  mockDisplayErrorToast: vi.fn(),
}));

vi.mock("#/hooks/mutation/use-switch-llm-profile-and-log", () => ({
  useSwitchLlmProfileAndLog: () => ({
    switchAndLog: mockSwitchAndLog,
    isPending: false,
  }),
}));

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string) => key,
  }),
}));

vi.mock("#/utils/custom-toast-handlers", () => ({
  displayErrorToast: mockDisplayErrorToast,
}));

const CONVERSATION_ID = "conv-1";

const cloudBackend: Backend = {
  id: "prod",
  name: "Production",
  host: "https://app.all-hands.dev",
  apiKey: "bearer-token",
  kind: "cloud",
};

const createUserMessageEvent = (id: string) => ({
  id,
  timestamp: new Date().toISOString(),
  source: "user" as const,
  llm_message: {
    role: "user" as const,
    content: [{ type: "text" as const, text: "User message" }],
  },
  activated_microagents: [],
  extended_content: [],
});

const makeWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });

  return function Wrapper({ children }: { children: React.ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>
        <ActiveBackendProvider>{children}</ActiveBackendProvider>
      </QueryClientProvider>
    );
  };
};

describe("useModelInterceptor", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    window.localStorage.clear();
    vi.spyOn(ProfilesService, "listProfiles").mockResolvedValue({
      profiles: [],
      active_profile: null,
    });
    __resetActiveStoreForTests();
    useEventStore.getState().clearEvents();
    useModelStore.setState({ entriesByConversation: {} });
  });

  it("lists profiles inline for bare /model submissions", async () => {
    vi.mocked(ProfilesService.listProfiles).mockResolvedValueOnce({
      profiles: [
        {
          name: "haiku",
          model: "anthropic/claude-haiku-4-5",
          base_url: null,
          api_key_set: true,
        },
      ],
      active_profile: "haiku",
    });
    useEventStore.getState().addEvent(createUserMessageEvent("message-1"));
    const onSubmit = vi.fn();

    const { result } = renderHook(
      () => useModelInterceptor(CONVERSATION_ID, onSubmit),
      { wrapper: makeWrapper() },
    );

    act(() => result.current(" /model "));

    await waitFor(() => {
      expect(
        useModelStore.getState().entriesByConversation[CONVERSATION_ID],
      ).toEqual([
        expect.objectContaining({
          anchorEventId: "message-1",
          profiles: [expect.objectContaining({ name: "haiku" })],
        }),
      ]);
    });
    expect(onSubmit).not.toHaveBeenCalled();
    expect(mockSwitchAndLog).not.toHaveBeenCalled();
  });

  it("switches profiles for /model <name> submissions", () => {
    const onSubmit = vi.fn();
    const { result } = renderHook(
      () => useModelInterceptor(CONVERSATION_ID, onSubmit),
      { wrapper: makeWrapper() },
    );

    act(() => result.current("/model haiku"));

    expect(mockSwitchAndLog).toHaveBeenCalledWith(CONVERSATION_ID, "haiku");
    expect(onSubmit).not.toHaveBeenCalled();
    expect(ProfilesService.listProfiles).not.toHaveBeenCalled();
  });

  it("falls through for non-model messages", () => {
    const onSubmit = vi.fn();
    const { result } = renderHook(
      () => useModelInterceptor(CONVERSATION_ID, onSubmit),
      { wrapper: makeWrapper() },
    );

    act(() => result.current("hello /model haiku"));

    expect(onSubmit).toHaveBeenCalledWith("hello /model haiku");
    expect(mockSwitchAndLog).not.toHaveBeenCalled();
  });

  it("intercepts on cloud backends (cloud also manages the LLM via profiles)", () => {
    setRegisteredBackends([cloudBackend]);
    setActiveSelection({ backendId: cloudBackend.id });
    const onSubmit = vi.fn();

    const { result } = renderHook(
      () => useModelInterceptor(CONVERSATION_ID, onSubmit),
      { wrapper: makeWrapper() },
    );

    act(() => result.current("/model haiku"));

    expect(mockSwitchAndLog).toHaveBeenCalledWith(CONVERSATION_ID, "haiku");
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("activates the named profile globally even when no conversation is set", () => {
    const onSubmit = vi.fn();
    const { result } = renderHook(() => useModelInterceptor(null, onSubmit), {
      wrapper: makeWrapper(),
    });

    act(() => result.current("/model haiku"));

    // Message is intercepted (never forwarded to the LLM) and the activate
    // call still fires — the home-page `/model NAME` flow.
    expect(onSubmit).not.toHaveBeenCalled();
    expect(mockSwitchAndLog).toHaveBeenCalledWith(null, "haiku");
  });

  it("swallows bare /model on the home page (no conversation to anchor to)", () => {
    const onSubmit = vi.fn();
    const { result } = renderHook(() => useModelInterceptor(null, onSubmit), {
      wrapper: makeWrapper(),
    });

    act(() => result.current("/model"));

    expect(onSubmit).not.toHaveBeenCalled();
    expect(mockSwitchAndLog).not.toHaveBeenCalled();
  });

  it("shows an error toast when listing profiles fails", async () => {
    vi.mocked(ProfilesService.listProfiles).mockRejectedValue(
      new Error("Network error"),
    );
    const onSubmit = vi.fn();

    const { result } = renderHook(
      () => useModelInterceptor(CONVERSATION_ID, onSubmit),
      { wrapper: makeWrapper() },
    );

    act(() => result.current("/model"));

    await waitFor(() => {
      expect(mockDisplayErrorToast).toHaveBeenCalledWith("Network error");
    });
    expect(onSubmit).not.toHaveBeenCalled();
  });
});
