import { renderHook } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { useOnboardingVisibility } from "./use-onboarding-visibility";

type HookProps = {
  scopeKey: string;
  eligible: boolean;
  dismissed: boolean;
};

function renderVisibility(initialProps: HookProps) {
  return renderHook((props: HookProps) => useOnboardingVisibility(props), {
    initialProps,
  });
}

describe("useOnboardingVisibility", () => {
  it("keeps an opened flow visible until it is dismissed", () => {
    const { result, rerender } = renderVisibility({
      scopeKey: "root",
      eligible: true,
      dismissed: false,
    });

    expect(result.current).toBe(true);

    rerender({ scopeKey: "root", eligible: false, dismissed: false });
    expect(result.current).toBe(true);

    rerender({ scopeKey: "root", eligible: true, dismissed: true });
    expect(result.current).toBe(false);

    rerender({ scopeKey: "root", eligible: false, dismissed: false });
    expect(result.current).toBe(false);
  });

  it("does not carry a latch into a different scope", () => {
    const { result, rerender } = renderVisibility({
      scopeKey: "backend-a",
      eligible: true,
      dismissed: false,
    });

    expect(result.current).toBe(true);

    rerender({
      scopeKey: "backend-b",
      eligible: false,
      dismissed: false,
    });
    expect(result.current).toBe(false);
  });

  it("opens when an existing scope becomes eligible", () => {
    const { result, rerender } = renderVisibility({
      scopeKey: "backend-a",
      eligible: false,
      dismissed: false,
    });

    expect(result.current).toBe(false);

    rerender({ scopeKey: "backend-a", eligible: true, dismissed: false });
    expect(result.current).toBe(true);
  });
});
