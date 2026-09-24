import { renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { DEFAULT_ENABLED_SKILL_NAMES } from "@openhands/extensions/skills";
import { useMigrateEnabledSkills } from "#/hooks/use-migrate-enabled-skills";

const useSettingsMock = vi.fn();
const useActiveBackendMock = vi.fn();
const saveSettingsMock = vi.fn();

vi.mock("#/hooks/query/use-settings", () => ({
  useSettings: () => useSettingsMock(),
}));
vi.mock("#/contexts/active-backend-context", () => ({
  useActiveBackend: () => useActiveBackendMock(),
}));
vi.mock("#/hooks/mutation/use-save-settings", () => ({
  useSaveSettings: () => ({ mutate: saveSettingsMock }),
}));

function settings(overrides: Record<string, unknown> = {}) {
  return { data: { ...overrides }, isLoading: false, isError: false };
}

describe("useMigrateEnabledSkills", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useActiveBackendMock.mockReturnValue({
      backend: { kind: "local", id: "b1" },
      orgId: null,
    });
  });

  it("writes the curated default for a workspace that has never chosen", async () => {
    useSettingsMock.mockReturnValue(settings({ disabled_skills: [] }));

    renderHook(() => useMigrateEnabledSkills());

    await waitFor(() =>
      expect(saveSettingsMock).toHaveBeenCalledWith({
        enabled_skills: [...DEFAULT_ENABLED_SKILL_NAMES],
        disabled_skills: [],
      }),
    );
  });

  it("preserves what an existing workspace had switched on", async () => {
    useSettingsMock.mockReturnValue(settings({ disabled_skills: ["deno"] }));

    renderHook(() => useMigrateEnabledSkills());

    await waitFor(() => expect(saveSettingsMock).toHaveBeenCalled());
    const [payload] = saveSettingsMock.mock.calls[0];
    // The old default was everything-on, so a workspace that only turned off
    // `deno` must not lose the rest of the catalog to the curated set.
    expect(payload.enabled_skills).toContain("add-javadoc");
    expect(payload.enabled_skills).not.toContain("deno");
  });

  it("leaves an already migrated workspace alone", async () => {
    useSettingsMock.mockReturnValue(
      settings({ enabled_skills: [], disabled_skills: [] }),
    );

    renderHook(() => useMigrateEnabledSkills());

    await waitFor(() => expect(saveSettingsMock).not.toHaveBeenCalled());
  });

  it("runs at most once while the save and refetch settle", async () => {
    useSettingsMock.mockReturnValue(settings({ disabled_skills: [] }));

    const { rerender } = renderHook(() => useMigrateEnabledSkills());
    rerender();
    rerender();

    await waitFor(() => expect(saveSettingsMock).toHaveBeenCalledTimes(1));
  });

  it("skips cloud, which never reads enabled_skills", async () => {
    useActiveBackendMock.mockReturnValue({
      backend: { kind: "cloud", id: "cloud" },
      orgId: null,
    });
    useSettingsMock.mockReturnValue(settings({ disabled_skills: [] }));

    renderHook(() => useMigrateEnabledSkills());

    await waitFor(() => expect(saveSettingsMock).not.toHaveBeenCalled());
  });

  it("does not persist a default over settings that failed to load", async () => {
    useSettingsMock.mockReturnValue({
      data: undefined,
      isLoading: false,
      isError: true,
    });

    renderHook(() => useMigrateEnabledSkills());

    await waitFor(() => expect(saveSettingsMock).not.toHaveBeenCalled());
  });
});
