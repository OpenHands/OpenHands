import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import React from "react";
import { act, renderHook, waitFor } from "@testing-library/react";
import { QueryClientProvider } from "@tanstack/react-query";
import { AxiosError, type AxiosResponse } from "axios";

import AutomationService from "#/api/automation-service/automation-service.api";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import type { Backend } from "#/api/backend-registry/types";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import { useHomeAutomationActions } from "#/hooks/use-home-automation-actions";
import { createAgentServerQueryClient } from "#/query-client-config";
import * as telemetry from "#/services/telemetry";
import {
  AutomationRunStatus,
  type Automation,
  type AutomationRun,
} from "#/types/automation";
import * as ToastHandlers from "#/utils/custom-toast-handlers";

vi.mock("#/api/automation-service/automation-service.api", () => ({
  default: {
    dispatchAutomation: vi.fn(),
    cancelAutomationRun: vi.fn(),
    toggleAutomation: vi.fn(),
  },
}));

vi.mock("#/hooks/use-automation-permissions", () => ({
  useAutomationPermissions: () => ({
    canView: true,
    canManage: true,
    isLoading: false,
  }),
  useIsAutomationOwner: () => true,
}));

vi.mock("#/hooks/query/use-settings", () => ({
  useSettings: () => ({ data: { user_consents_to_analytics: false } }),
}));

const localBackend: Backend = {
  id: "local-1",
  name: "Local 1",
  host: "http://localhost:8000",
  apiKey: "session-key",
  kind: "local",
};

const automation: Automation = {
  id: "auto-1",
  name: "Test",
  prompt: "p",
  trigger: { type: "schedule", schedule_human: "Daily" },
  enabled: true,
  repository: "acme/repo",
  model: "daily-profile",
  created_at: "2026-01-01T00:00:00Z",
  updated_at: "2026-01-01T00:00:00Z",
};

const runningRun: AutomationRun = {
  id: "run-1",
  status: AutomationRunStatus.RUNNING,
  conversation_id: null,
  bash_command_id: null,
  error_detail: null,
  started_at: "2026-01-02T00:00:00Z",
  completed_at: null,
};

// The automation service answers a stale id with 404 and a `detail` body.
const notFound = new AxiosError(
  "Request failed with status code 404",
  "ERR_BAD_REQUEST",
  undefined,
  undefined,
  {
    status: 404,
    data: { detail: "Automation not found" },
  } as AxiosResponse,
);

// The app's real client, whose MutationCache toasts unless a mutation opts out.
function makeAppClientWrapper() {
  const queryClient = createAgentServerQueryClient();
  return function Wrapper({ children }: { children: React.ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>
        <ActiveBackendProvider>{children}</ActiveBackendProvider>
      </QueryClientProvider>
    );
  };
}

let errorToast: ReturnType<typeof vi.spyOn>;
let trackEvent: ReturnType<typeof vi.spyOn>;

beforeEach(() => {
  trackEvent = vi.spyOn(telemetry, "trackEvent").mockResolvedValue(undefined);
  errorToast = vi
    .spyOn(ToastHandlers, "displayErrorToast")
    .mockImplementation(() => "toast-id");
  __resetActiveStoreForTests();
  setRegisteredBackends([localBackend]);
  setActiveSelection({ backendId: localBackend.id });
  vi.mocked(AutomationService.dispatchAutomation).mockRejectedValue(notFound);
  vi.mocked(AutomationService.cancelAutomationRun).mockRejectedValue(notFound);
  vi.mocked(AutomationService.toggleAutomation).mockRejectedValue(notFound);
});

afterEach(() => {
  errorToast.mockRestore();
  trackEvent.mockRestore();
  __resetActiveStoreForTests();
});

describe("useHomeAutomationActions — error toasts", () => {
  it.each([
    {
      action: "Run now",
      service: () => AutomationService.dispatchAutomation,
      trigger: (actions: ReturnType<typeof useHomeAutomationActions>) =>
        actions.runNow(),
    },
    {
      action: "Turn off",
      service: () => AutomationService.toggleAutomation,
      trigger: (actions: ReturnType<typeof useHomeAutomationActions>) =>
        actions.confirmTurnOff(),
    },
    {
      action: "Cancel run",
      service: () => AutomationService.cancelAutomationRun,
      trigger: (actions: ReturnType<typeof useHomeAutomationActions>) =>
        actions.cancelRun(),
    },
  ])(
    "a failed home $action shows one toast with the API message",
    async ({ service, trigger }) => {
      // Arrange
      const { result } = renderHook(
        () => useHomeAutomationActions(automation, runningRun),
        { wrapper: makeAppClientWrapper() },
      );

      // Act
      act(() => trigger(result.current));

      // Assert
      await waitFor(() => expect(service()).toHaveBeenCalledTimes(1));
      // The global MutationCache handler runs before the caller's onError.
      await waitFor(() =>
        expect(errorToast).toHaveBeenCalledWith("Automation not found"),
      );
      expect(errorToast).toHaveBeenCalledTimes(1);
    },
  );
});
