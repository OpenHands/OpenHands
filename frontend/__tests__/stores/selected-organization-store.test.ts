import { act, renderHook } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { useSelectedOrganizationStore } from "#/stores/selected-organization-store";

describe("useSelectedOrganizationStore", () => {
  it("should have null as initial organizationId", () => {
    const { result } = renderHook(() => useSelectedOrganizationStore());
    expect(result.current.organizationId).toBeNull();
  });

  it("should update organizationId when setOrganizationId is called", () => {
    const { result } = renderHook(() => useSelectedOrganizationStore());

    act(() => {
      result.current.setOrganizationId("org-123");
    });

    expect(result.current.organizationId).toBe("org-123");
  });

  it("should allow setting organizationId to null", () => {
    const { result } = renderHook(() => useSelectedOrganizationStore());

    act(() => {
      result.current.setOrganizationId("org-123");
    });

    expect(result.current.organizationId).toBe("org-123");

    act(() => {
      result.current.setOrganizationId(null);
    });

    expect(result.current.organizationId).toBeNull();
  });

  it("should share state across multiple hook instances", () => {
    const { result: result1 } = renderHook(() =>
      useSelectedOrganizationStore(),
    );
    const { result: result2 } = renderHook(() =>
      useSelectedOrganizationStore(),
    );

    act(() => {
      result1.current.setOrganizationId("shared-organization");
    });

    expect(result2.current.organizationId).toBe("shared-organization");
  });
});
