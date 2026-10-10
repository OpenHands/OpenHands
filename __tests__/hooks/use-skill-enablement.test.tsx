import React from "react";
import { act, renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import SettingsService from "#/api/settings-service/settings-service.api";
import { useSkillEnablement } from "#/hooks/use-skill-enablement";
import { useSettings } from "#/hooks/query/use-settings";
import { SETTINGS_QUERY_KEYS } from "#/hooks/query/query-keys";
import { DEFAULT_SETTINGS } from "#/services/settings";
import { CATALOG_SKILL_NAMES } from "#/utils/skill-enablement";
import type { Settings, SkillInfo } from "#/types/settings";
import { displayErrorToast } from "#/utils/custom-toast-handlers";

const active = vi.hoisted(() => ({
  backend: { id: "test-backend", kind: "local" },
  orgId: null,
}));
vi.mock("#/contexts/active-backend-context", () => ({
  useActiveBackend: () => active,
}));
vi.mock("#/hooks/use-tracking", () => ({
  useTracking: () => ({ trackMcpConfigUpdated: vi.fn() }),
}));
vi.mock("#/utils/custom-toast-handlers", () => ({
  displayErrorToast: vi.fn(),
}));

describe("useSkillEnablement", () => {
  let queryClient: QueryClient;
  let settings: Settings;
  const skill = { name: CATALOG_SKILL_NAMES[0] } as SkillInfo;
  const otherSkill = { name: "project-skill" } as SkillInfo;

  beforeEach(() => {
    active.backend.kind = "local";
    settings = {
      ...DEFAULT_SETTINGS,
      enabled_skills: [],
      disabled_skills: [skill.name],
    };
    queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
        mutations: { retry: false },
      },
    });
    vi.spyOn(SettingsService, "getSettings").mockImplementation(
      async () => settings,
    );
  });

  afterEach(() => {
    queryClient.clear();
    vi.restoreAllMocks();
    vi.clearAllMocks();
  });

  function wrapper({ children }: { children: React.ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    );
  }

  it.each(["local", "cloud"])(
    "preserves the last successful %s save when refetch is pending and the next save fails",
    async (kind) => {
      active.backend.kind = kind;
      const secondSkill = { name: CATALOG_SKILL_NAMES[1] } as SkillInfo;
      settings = {
        ...settings,
        enabled_skills: [skill.name, secondSkill.name],
        disabled_skills: [],
      };
      let finishRefetch!: (value: Settings) => void;
      vi.mocked(SettingsService.getSettings)
        .mockResolvedValueOnce(settings)
        .mockImplementationOnce(
          () =>
            new Promise((resolve) => {
              finishRefetch = resolve;
            }),
        );
      const save = vi
        .spyOn(SettingsService, "saveSettings")
        .mockImplementationOnce(async (update) => {
          settings = { ...settings, ...update };
          return true;
        })
        .mockRejectedValueOnce(new Error("Second save failed"));
      const { result } = renderHook(
        () => {
          const query = useSettings();
          return { ...useSkillEnablement(), settingsLoaded: query.isSuccess };
        },
        { wrapper },
      );
      await waitFor(() => expect(result.current.settingsLoaded).toBe(true));

      act(() => result.current.setEnabled(skill.name, false));
      // PATCH has succeeded, but the real mutation hook is awaiting GET.
      await waitFor(() =>
        expect(SettingsService.getSettings).toHaveBeenCalledTimes(2),
      );
      act(() => result.current.setEnabled(secondSkill.name, false));

      await waitFor(() => expect(displayErrorToast).toHaveBeenCalledTimes(1));
      expect(result.current.isEnabled(skill)).toBe(false);
      expect(result.current.isEnabled(secondSkill)).toBe(true);
      expect(save).toHaveBeenCalledTimes(2);
      await act(async () => finishRefetch(settings));
    },
  );

  it("rolls back to a successful overlapping save without waiting for refetch", async () => {
    const secondSkill = { name: CATALOG_SKILL_NAMES[1] } as SkillInfo;
    settings = {
      ...settings,
      enabled_skills: [skill.name, secondSkill.name],
      disabled_skills: [],
    };
    let finishSave!: () => void;
    let rejectSecondSave!: (error: Error) => void;
    let finishRefetch!: (value: Settings) => void;
    vi.mocked(SettingsService.getSettings)
      .mockResolvedValueOnce(settings)
      .mockImplementationOnce(
        () =>
          new Promise((resolve) => {
            finishRefetch = resolve;
          }),
      );
    const save = vi
      .spyOn(SettingsService, "saveSettings")
      .mockImplementationOnce(
        (update) =>
          new Promise((resolve) => {
            finishSave = () => {
              settings = { ...settings, ...update };
              resolve(true);
            };
          }),
      )
      .mockImplementationOnce(
        () =>
          new Promise((_, reject) => {
            rejectSecondSave = reject;
          }),
      );
    const { result } = renderHook(
      () => {
        const query = useSettings();
        return { ...useSkillEnablement(), settingsLoaded: query.isSuccess };
      },
      { wrapper },
    );
    await waitFor(() => expect(result.current.settingsLoaded).toBe(true));
    act(() => result.current.setEnabled(skill.name, false));
    await waitFor(() => expect(save).toHaveBeenCalledTimes(1));
    act(() => result.current.setEnabled(secondSkill.name, false));
    await waitFor(() => expect(save).toHaveBeenCalledTimes(2));
    act(() => finishSave());
    await waitFor(() =>
      expect(SettingsService.getSettings).toHaveBeenCalledTimes(2),
    );

    act(() => rejectSecondSave(new Error("Second save failed")));

    await waitFor(() => expect(displayErrorToast).toHaveBeenCalledTimes(1));
    expect(result.current.isEnabled(skill)).toBe(false);
    expect(result.current.isEnabled(secondSkill)).toBe(true);
    await act(async () => finishRefetch(settings));
  });

  it("does not roll back refreshed settings when an older save fails", async () => {
    let rejectSave!: (error: Error) => void;
    const save = vi.spyOn(SettingsService, "saveSettings").mockImplementation(
      () =>
        new Promise((_, reject) => {
          rejectSave = reject;
        }),
    );
    const { result } = renderHook(
      () => {
        const query = useSettings();
        return { ...useSkillEnablement(), settingsLoaded: query.isSuccess };
      },
      { wrapper },
    );
    await waitFor(() => expect(result.current.settingsLoaded).toBe(true));
    act(() => result.current.setEnabled(skill.name, true));
    await waitFor(() => expect(save).toHaveBeenCalledTimes(1));

    settings = {
      ...settings,
      enabled_skills: [skill.name],
      disabled_skills: [],
    };
    await act(async () => {
      await queryClient.invalidateQueries({
        queryKey: SETTINGS_QUERY_KEYS.all,
      });
    });
    act(() => rejectSave(new Error("Earlier save failed")));

    await waitFor(() => expect(displayErrorToast).toHaveBeenCalledTimes(1));
    expect(result.current.isEnabled(skill)).toBe(true);
    expect(save).toHaveBeenCalledTimes(1);
  });

  it.each(["local", "cloud"])(
    "rolls back a failed %s save and allows a later retry",
    async (kind) => {
      active.backend.kind = kind;
      let rejectSave!: (error: Error) => void;
      const save = vi
        .spyOn(SettingsService, "saveSettings")
        .mockImplementationOnce(
          () =>
            new Promise((_, reject) => {
              rejectSave = reject;
            }),
        )
        .mockImplementation(async (update) => {
          settings = { ...settings, ...update };
          return true;
        });
      const { result } = renderHook(
        () => {
          const query = useSettings();
          return { ...useSkillEnablement(), settingsLoaded: query.isSuccess };
        },
        { wrapper },
      );
      await waitFor(() => expect(result.current.settingsLoaded).toBe(true));
      expect(result.current.isEnabled(skill)).toBe(false);
      expect(save).not.toHaveBeenCalled();

      act(() => result.current.setEnabled(skill.name, true));
      await waitFor(() => expect(save).toHaveBeenCalledTimes(1));
      expect(result.current.isEnabled(skill)).toBe(true);
      act(() => rejectSave(new Error("Save failed")));

      await waitFor(() => expect(result.current.isEnabled(skill)).toBe(false));
      expect(displayErrorToast).toHaveBeenCalledTimes(1);
      // A subsequent edit must not carry the failed enablement into its payload.
      act(() => result.current.setEnabled(otherSkill.name, false));
      await waitFor(() => expect(save).toHaveBeenCalledTimes(2));
      expect(save.mock.calls[1][0].disabled_skills).toContain(skill.name);
      await waitFor(() =>
        expect(SettingsService.getSettings).toHaveBeenCalledTimes(2),
      );

      act(() => result.current.setEnabled(skill.name, true));
      await waitFor(() => expect(save).toHaveBeenCalledTimes(3));
      await waitFor(() =>
        expect(settings.disabled_skills).not.toContain(skill.name),
      );
      expect(result.current.isEnabled(skill)).toBe(true);
      expect(save.mock.calls[2][0]).toEqual(
        kind === "cloud"
          ? { disabled_skills: [otherSkill.name] }
          : {
              enabled_skills: [skill.name],
              disabled_skills: [otherSkill.name],
            },
      );
    },
  );
});
