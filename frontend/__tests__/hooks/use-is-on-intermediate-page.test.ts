import { renderHook } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

// Unmock the hook so we can test the real implementation
vi.unmock("#/hooks/use-is-on-intermediate-page");

const useLocationMock = vi.fn();

vi.mock("react-router", async () => {
  const actual = await vi.importActual("react-router");
  return {
    ...actual,
    useLocation: useLocationMock,
  };
});

// Import after mock setup
const { useIsOnIntermediatePage } = await import(
  "#/hooks/use-is-on-intermediate-page"
);

describe("useIsOnIntermediatePage", () => {
  describe("returns true for intermediate pages", () => {
    it("should return true when on /accept-tos page", () => {
      useLocationMock.mockReturnValue({ pathname: "/accept-tos" });
      const { result } = renderHook(() => useIsOnIntermediatePage());
      expect(result.current).toBe(true);
    });

    it("should return true when on /onboarding page", () => {
      useLocationMock.mockReturnValue({ pathname: "/onboarding" });
      const { result } = renderHook(() => useIsOnIntermediatePage());
      expect(result.current).toBe(true);
    });
  });

  describe("returns false for non-intermediate pages", () => {
    it("should return false when on root page", () => {
      useLocationMock.mockReturnValue({ pathname: "/" });
      const { result } = renderHook(() => useIsOnIntermediatePage());
      expect(result.current).toBe(false);
    });

    it("should return false when on /settings page", () => {
      useLocationMock.mockReturnValue({ pathname: "/settings" });
      const { result } = renderHook(() => useIsOnIntermediatePage());
      expect(result.current).toBe(false);
    });
  });

  describe("handles edge cases", () => {
    it("should return false for paths containing intermediate page names", () => {
      useLocationMock.mockReturnValue({ pathname: "/accept-tos-extra" });
      const { result } = renderHook(() => useIsOnIntermediatePage());
      expect(result.current).toBe(false);
    });

    it("should return false for paths with intermediate page names as subpaths", () => {
      useLocationMock.mockReturnValue({ pathname: "/settings/accept-tos" });
      const { result } = renderHook(() => useIsOnIntermediatePage());
      expect(result.current).toBe(false);
    });
  });
});
