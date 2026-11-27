import { renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useGoalInterceptor } from "./use-goal-interceptor";

const startGoal = vi.fn();
const displayErrorToast = vi.fn();

vi.mock("#/hooks/mutation/conversation-mutation-utils", () => ({
  startGoal: (...args: unknown[]) => startGoal(...args),
}));
vi.mock("#/utils/custom-toast-handlers", () => ({
  displayErrorToast: (...args: unknown[]) => displayErrorToast(...args),
}));
vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: (key: string) => key }),
}));

const CONV = "conv-1";

const setup = (conversationId: string | null) => {
  const onSubmit = vi.fn();
  const { result } = renderHook(() =>
    useGoalInterceptor(conversationId, onSubmit),
  );
  return { intercept: result.current, onSubmit };
};

describe("useGoalInterceptor", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    startGoal.mockResolvedValue(undefined);
  });

  it("passes a non-goal message straight through to onSubmit", () => {
    const { intercept, onSubmit } = setup(CONV);
    intercept("hello there");
    expect(onSubmit).toHaveBeenCalledWith("hello there");
    expect(startGoal).not.toHaveBeenCalled();
  });

  it("toasts and starts nothing for a bare /goal (no objective)", () => {
    const { intercept, onSubmit } = setup(CONV);
    intercept("/goal");
    expect(displayErrorToast).toHaveBeenCalledWith("GOAL$OBJECTIVE_REQUIRED");
    expect(startGoal).not.toHaveBeenCalled();
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("toasts for a flag-only /goal with no objective", () => {
    const { intercept } = setup(CONV);
    intercept("/goal --max 5");
    expect(displayErrorToast).toHaveBeenCalledWith("GOAL$OBJECTIVE_REQUIRED");
    expect(startGoal).not.toHaveBeenCalled();
  });

  it("starts a goal loop with the objective", () => {
    const { intercept } = setup(CONV);
    intercept("/goal build a hello world script");
    expect(startGoal).toHaveBeenCalledWith(CONV, {
      objective: "build a hello world script",
    });
    expect(displayErrorToast).not.toHaveBeenCalled();
  });

  it("parses a leading --max flag into max_iterations", () => {
    const { intercept } = setup(CONV);
    intercept("/goal --max 3 build it");
    expect(startGoal).toHaveBeenCalledWith(CONV, {
      objective: "build it",
      max_iterations: 3,
    });
  });

  it("passes through (no toast, no loop) when there is no conversation", () => {
    const { intercept, onSubmit } = setup(null);
    intercept("/goal");
    expect(onSubmit).toHaveBeenCalledWith("/goal");
    expect(startGoal).not.toHaveBeenCalled();
    expect(displayErrorToast).not.toHaveBeenCalled();
  });
});
