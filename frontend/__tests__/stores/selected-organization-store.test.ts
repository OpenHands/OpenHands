import { act, renderHook } from "@testing-library/react";
import { describe, expect, it, beforeEach } from "vitest";
import { useSelectedOrganizationStore } from "#/stores/selected-organization-store";

describe("useSelectedOrganizationStore", () => {
  beforeEach(() => {
    localStorage.clear();
  });

  it("should have null as initial orgId", () => {
    const { result } = renderHook(() => useSelectedOrganizationStore());
    expect(result.current.orgId).toBeNull();
  });

  it("should update orgId when setOrgId is called", () => {
    const { result } = renderHook(() => useSelectedOrganizationStore());

    act(() => {
      result.current.setOrgId("org-123");
    });

    expect(result.current.orgId).toBe("org-123");
  });

  it("should allow setting orgId to null", () => {
    const { result } = renderHook(() => useSelectedOrganizationStore());

    act(() => {
      result.current.setOrgId("org-123");
    });

    expect(result.current.orgId).toBe("org-123");

    act(() => {
      result.current.setOrgId(null);
    });

    expect(result.current.orgId).toBeNull();
  });

  it("should persist orgId to localStorage", () => {
    const { result } = renderHook(() => useSelectedOrganizationStore());

    act(() => {
      result.current.setOrgId("org-456");
    });

    const stored = localStorage.getItem("selected-organization-id");
    expect(stored).toBeTruthy();

    const parsed = JSON.parse(stored!);
    expect(parsed.state.orgId).toBe("org-456");
  });

  it("should persist state across hook calls", () => {
    const { result: result1 } = renderHook(() =>
      useSelectedOrganizationStore(),
    );

    act(() => {
      result1.current.setOrgId("org-789");
    });

    const stored = localStorage.getItem("selected-organization-id");
    expect(stored).toBeTruthy();

    const { result: result2 } = renderHook(() =>
      useSelectedOrganizationStore(),
    );

    expect(result2.current.orgId).toBe("org-789");
  });

  it("should share state across multiple hook instances", () => {
    const { result: result1 } = renderHook(() =>
      useSelectedOrganizationStore(),
    );
    const { result: result2 } = renderHook(() =>
      useSelectedOrganizationStore(),
    );

    act(() => {
      result1.current.setOrgId("shared-org");
    });

    expect(result2.current.orgId).toBe("shared-org");
  });
});
