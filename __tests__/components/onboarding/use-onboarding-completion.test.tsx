import React from "react";
import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
  ONBOARDING_COMPLETED_STORAGE_KEY,
  useOnboardingCompletion,
} from "#/components/features/onboarding/use-onboarding-completion";

beforeEach(() => {
  window.localStorage.clear();
});
afterEach(() => {
  window.localStorage.clear();
});

describe("useOnboardingCompletion", () => {
  it("starts incomplete when the localStorage flag is missing", () => {
    const { result } = renderHook(() => useOnboardingCompletion());
    expect(result.current.isCompleted).toBe(false);
  });

  it("starts complete when the localStorage flag is set", () => {
    window.localStorage.setItem(ONBOARDING_COMPLETED_STORAGE_KEY, "1");
    const { result } = renderHook(() => useOnboardingCompletion());
    expect(result.current.isCompleted).toBe(true);
  });

  it("persists completion to localStorage and flips the flag", () => {
    const { result } = renderHook(() => useOnboardingCompletion());
    expect(result.current.isCompleted).toBe(false);

    act(() => {
      result.current.markCompleted();
    });

    expect(result.current.isCompleted).toBe(true);
    expect(
      window.localStorage.getItem(ONBOARDING_COMPLETED_STORAGE_KEY),
    ).not.toBeNull();
  });

  it("syncs across tabs via the storage event", () => {
    const { result } = renderHook(() => useOnboardingCompletion());
    expect(result.current.isCompleted).toBe(false);

    act(() => {
      window.localStorage.setItem(ONBOARDING_COMPLETED_STORAGE_KEY, "1");
      window.dispatchEvent(
        new StorageEvent("storage", {
          key: ONBOARDING_COMPLETED_STORAGE_KEY,
          newValue: "1",
        }),
      );
    });

    expect(result.current.isCompleted).toBe(true);
  });
});
