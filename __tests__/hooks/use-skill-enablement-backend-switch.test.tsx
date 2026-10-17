import React from "react";
import { act, renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, expect, it, vi } from "vitest";
import SettingsService from "#/api/settings-service/settings-service.api";
import {
  ActiveBackendProvider,
  useActiveBackendContext,
} from "#/contexts/active-backend-context";
import {
  getActiveBackend,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import { useSkillEnablement } from "#/hooks/use-skill-enablement";
import { useSettings } from "#/hooks/query/use-settings";
import { DEFAULT_SETTINGS } from "#/services/settings";
import { CATALOG_SKILL_NAMES } from "#/utils/skill-enablement";
import type { Settings, SkillInfo } from "#/types/settings";
import { displayErrorToast } from "#/utils/custom-toast-handlers";

vi.mock("#/hooks/use-tracking", () => ({
  useTracking: () => ({ trackMcpConfigUpdated: vi.fn() }),
}));
vi.mock("#/utils/custom-toast-handlers", () => ({
  displayErrorToast: vi.fn(),
}));
afterEach(() => {
  vi.restoreAllMocks();
  vi.clearAllMocks();
});

it.each([
  { transition: "backend", oldCompletes: false },
  { transition: "backend", oldCompletes: true },
  { transition: "organization", oldCompletes: true },
  { transition: "roundtrip", oldCompletes: true },
] as const)(
  "preserves the current $transition rollback baseline (old save completes: $oldCompletes)",
  async ({ transition, oldCompletes }) => {
    const [changedName, untouchedName] = CATALOG_SKILL_NAMES;
    const skill = (name: string) => ({ name }) as SkillInfo;
    const settingsA: Settings = {
      ...DEFAULT_SETTINGS,
      enabled_skills:
        transition === "roundtrip"
          ? [changedName, untouchedName]
          : [untouchedName],
      disabled_skills: [],
    };
    const settingsB: Settings = {
      ...settingsA,
      enabled_skills: [changedName, untouchedName],
    };
    setRegisteredBackends([
      {
        id: "backend-A",
        name: "A",
        host: "http://127.0.0.1:10001",
        apiKey: "",
        kind: transition === "organization" ? "cloud" : "local",
      },
      {
        id: "backend-B",
        name: "B",
        host: "http://127.0.0.1:10002",
        apiKey: "",
        kind: "local",
      },
    ]);
    setActiveSelection({
      backendId: "backend-A",
      orgId: transition === "organization" ? "org-A" : null,
    });
    const reads = new Set<string>();
    let finishRefetch: ((value: Settings) => void) | undefined;
    vi.spyOn(SettingsService, "getSettings").mockImplementation(() => {
      const { backend, orgId } = getActiveBackend();
      const identity = JSON.stringify([backend.id, orgId]);
      if (reads.has(identity))
        return new Promise((resolve) => {
          finishRefetch = resolve;
        });
      reads.add(identity);
      return Promise.resolve(
        backend.id === "backend-B" || orgId === "org-B" ? settingsB : settingsA,
      );
    });
    let completeOldWrite!: () => void;
    const save = vi
      .spyOn(SettingsService, "saveSettings")
      .mockImplementationOnce(
        () =>
          new Promise((resolve) => {
            completeOldWrite = () => resolve(true);
          }),
      )
      .mockRejectedValueOnce(new Error("Current settings write failed"));
    const queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
        mutations: { retry: false },
      },
    });
    const wrapper = ({ children }: { children: React.ReactNode }) => (
      <QueryClientProvider client={queryClient}>
        <ActiveBackendProvider>{children}</ActiveBackendProvider>
      </QueryClientProvider>
    );
    const mounted = renderHook(
      () => ({
        context: useActiveBackendContext(),
        settings: useSettings(),
        ...useSkillEnablement(),
      }),
      { wrapper },
    );
    try {
      await waitFor(() =>
        expect(mounted.result.current.settings.isSuccess).toBe(true),
      );
      act(() => mounted.result.current.setEnabled(untouchedName, false));
      await waitFor(() => expect(save).toHaveBeenCalledTimes(1));
      act(() =>
        mounted.result.current.context.setActive(
          transition === "organization" ? "backend-A" : "backend-B",
          transition === "organization" ? "org-B" : null,
        ),
      );
      await waitFor(() =>
        expect(mounted.result.current.settings.data?.enabled_skills).toEqual(
          settingsB.enabled_skills,
        ),
      );
      await waitFor(() =>
        expect(mounted.result.current.isEnabled(skill(untouchedName))).toBe(
          true,
        ),
      );
      if (transition === "roundtrip") {
        // Returning to the same cached owner needs a generation check too.
        act(() => mounted.result.current.context.setActive("backend-A"));
        await act(async () => Promise.resolve());
      }
      if (oldCompletes) {
        act(() => completeOldWrite());
        await waitFor(() => expect(finishRefetch).toBeDefined());
      }
      // Hold GET so it cannot hide a contaminated rollback baseline.
      act(() => mounted.result.current.setEnabled(changedName, false));
      await waitFor(() => expect(displayErrorToast).toHaveBeenCalledTimes(1));
      expect(mounted.result.current.isEnabled(skill(untouchedName))).toBe(true);
      expect(mounted.result.current.isEnabled(skill(changedName))).toBe(true);
      expect(save).toHaveBeenCalledTimes(2);
    } finally {
      if (finishRefetch) await act(async () => finishRefetch!(settingsB));
      mounted.unmount();
      queryClient.clear();
      if (!oldCompletes) completeOldWrite();
    }
  },
);
