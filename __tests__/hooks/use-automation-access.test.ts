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

  it("treats an automation without owner_kind as user-owned", () => {
    permissions.canManage = true;

    const { result } = renderHook(() => useAutomationAccess(base));

    // A manager may manage but not edit someone else's user-owned automation.
    expect(result.current.canManage).toBe(true);
    expect(result.current.canEdit).toBe(false);
  });

  it("lets a manager edit and turn on a service-owned automation", () => {
    permissions.canManage = true;

    const { result } = renderHook(() =>
      useAutomationAccess({ ...base, owner_kind: "service", enabled: false }),
    );

    expect(result.current).toEqual({
      canManage: true,
      canEdit: true,
      canToggle: true,
    });
  });

  it("keeps a non-manager out of a service-owned automation", () => {
    const { result } = renderHook(() =>
      useAutomationAccess({ ...base, owner_kind: "service" }),
    );

    expect(result.current).toEqual({
      canManage: false,
      canEdit: false,
      canToggle: false,
    });
  });
});
