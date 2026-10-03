import { renderHook, act, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useSkillEnablement } from "#/hooks/use-skill-enablement";
import type { SkillInfo } from "#/types/settings";

const useSettingsMock = vi.fn();
const useActiveBackendMock = vi.fn();
const saveSettingsMock = vi.fn();
const isPendingMock = vi.fn();
const tMock = vi.fn((key: string) => key);

vi.mock("#/hooks/query/use-settings", () => ({
  useSettings: () => useSettingsMock(),
}));
vi.mock("#/contexts/active-backend-context", () => ({
  useActiveBackend: () => useActiveBackendMock(),
}));
vi.mock("#/hooks/mutation/use-save-settings", () => ({
  useSaveSettings: () => ({
    mutate: saveSettingsMock,
    isPending: isPendingMock(),
  }),
}));
vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: tMock }),
}));

function settings(overrides: Record<string, unknown> = {}) {
  return {
    data: { enabled_skills: undefined, disabled_skills: [], ...overrides },
    isLoading: false,
    isError: false,
  };
}

const SKILL: SkillInfo = {
  name: "add-skill",
  type: "repo",
  source: null,
};

describe("useSkillEnablement", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useActiveBackendMock.mockReturnValue({
      backend: { kind: "local", id: "b1" },
      orgId: null,
    });
    isPendingMock.mockReturnValue(false);
  });

  it("does not revert a toggle when a stale refetch arrives during a save (#15694)", async () => {
    // Start with the skill enabled (not in the disabled list).
    useSettingsMock.mockReturnValue(settings({ disabled_skills: [] }));

    const { result, rerender } = renderHook(() => useSkillEnablement());

    // Wait for initial hydration — the skill should be enabled.
    await waitFor(() => expect(result.current.isEnabled(SKILL)).toBe(true));

    // User toggles the skill off.
    act(() => {
      result.current.setEnabled("add-skill", false);
    });

    // The save should have been triggered and the skill disabled locally.
    expect(saveSettingsMock).toHaveBeenCalled();
    expect(result.current.isEnabled(SKILL)).toBe(false);

    // Simulate a stale refetch arriving while the save is still in flight:
    // isPending = true, settings still show the OLD state (skill enabled).
    isPendingMock.mockReturnValue(true);
    useSettingsMock.mockReturnValue(settings({ disabled_skills: [] }));

    rerender();

    // The toggle must NOT be reverted — the guard skips hydration during save.
    expect(result.current.isEnabled(SKILL)).toBe(false);
  });

  it("hydrates from settings once the save settles", async () => {
    // Start with the skill enabled.
    useSettingsMock.mockReturnValue(settings({ disabled_skills: [] }));

    const { result, rerender } = renderHook(() => useSkillEnablement());

    await waitFor(() => expect(result.current.isEnabled(SKILL)).toBe(true));

    // Toggle off.
    act(() => {
      result.current.setEnabled("add-skill", false);
    });
    expect(result.current.isEnabled(SKILL)).toBe(false);

    // Save completes, refetch returns the updated state (skill disabled).
    isPendingMock.mockReturnValue(false);
    useSettingsMock.mockReturnValue(
      settings({ disabled_skills: ["add-skill"] }),
    );

    rerender();

    // Hydration fires with correct data — skill stays disabled.
    await waitFor(() => expect(result.current.isEnabled(SKILL)).toBe(false));
  });
});
