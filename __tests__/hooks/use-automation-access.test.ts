import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderHook } from "@testing-library/react";
import { useAutomationAccess } from "#/hooks/use-automation-access";
import type { Automation } from "#/types/automation";

const permissions = vi.hoisted(() => ({
  canManage: false,
  isOwner: false,
}));

vi.mock("#/hooks/use-automation-permissions", () => ({
  useAutomationPermissions: () => ({ canManage: permissions.canManage }),
  useIsAutomationOwner: () => permissions.isOwner,
}));

const base = {
  id: "auto-1",
  name: "Nightly",
  enabled: true,
  user_id: "user-1",
} as Automation;

beforeEach(() => {
  permissions.canManage = false;
  permissions.isOwner = false;
});

describe("useAutomationAccess", () => {
  it("fails closed while the caller's permissions are loading", () => {
    // /me unresolved: no manage permission and the owner check is false.
    const { result } = renderHook(() => useAutomationAccess(base));

    expect(result.current).toEqual({
      canManage: false,
      canEdit: false,
      canToggle: false,
    });
  });

  it("treats an automation without run_as_kind as running as a person", () => {
    permissions.canManage = true;

    const { result } = renderHook(() => useAutomationAccess(base));

    // A manager may manage but not edit someone else's automation that runs as a person.
    expect(result.current.canManage).toBe(true);
    expect(result.current.canEdit).toBe(false);
  });

  it("lets a manager edit and turn on a automation that runs as a service account", () => {
    permissions.canManage = true;

    const { result } = renderHook(() =>
      useAutomationAccess({ ...base, run_as_kind: "service", enabled: false }),
    );

    expect(result.current).toEqual({
      canManage: true,
      canEdit: true,
      canToggle: true,
    });
  });

  it("keeps a non-manager out of a automation that runs as a service account", () => {
    const { result } = renderHook(() =>
      useAutomationAccess({ ...base, run_as_kind: "service" }),
    );

    expect(result.current).toEqual({
      canManage: false,
      canEdit: false,
      canToggle: false,
    });
  });
});
