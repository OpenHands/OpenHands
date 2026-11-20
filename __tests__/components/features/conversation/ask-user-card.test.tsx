import { beforeEach, describe, expect, it, vi } from "vitest";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "test-utils";
import { AskUserCard } from "#/components/features/conversation/ask-user-card";
import {
  type AskUserDecisionRequest,
  useAskUserStore,
} from "#/stores/ask-user-store";
import * as askUserService from "#/services/ask-user";

describe("AskUserCard Component", () => {
  const completeSpy = vi.spyOn(askUserService, "completeAskUserDecision");

  beforeEach(() => {
    window.localStorage.clear();
    useAskUserStore.getState().clear();
    completeSpy.mockClear();
  });

  const baseRequest: AskUserDecisionRequest = {
    conversationId: "conv-1",
    toolCallId: "call-1",
    question: "Do you want to proceed with database migration?",
    options: [
      {
        label: "Apply migration now",
        description: "Executes Alembic migrations in sandbox",
      },
      {
        label: "Skip and test manually",
        description: "Bypasses automatic execution",
      },
    ],
    recommended: "Apply migration now",
    selectedAnswer: null,
  };

  it("renders question, option choices, and recommended indicator", () => {
    renderWithProviders(<AskUserCard request={baseRequest} />);

    expect(screen.getByTestId("ask-user-card")).toBeInTheDocument();
    expect(
      screen.getByText("Do you want to proceed with database migration?"),
    ).toBeInTheDocument();
    expect(screen.getByText("Apply migration now")).toBeInTheDocument();
    expect(
      screen.getByText("Executes Alembic migrations in sandbox"),
    ).toBeInTheDocument();
    expect(screen.getByText("Skip and test manually")).toBeInTheDocument();
  });

  it("allows selecting an option and submitting the decision", async () => {
    const user = userEvent.setup();
    renderWithProviders(<AskUserCard request={baseRequest} />);

    const skipOption = screen.getByTestId("ask-user-option-1");
    await user.click(skipOption);

    const submitBtn = screen.getByTestId("ask-user-submit");
    expect(submitBtn).not.toBeDisabled();

    await user.click(submitBtn);
    expect(completeSpy).toHaveBeenCalledWith(
      baseRequest,
      "Skip and test manually",
    );
  });

  it("allows typing a custom answer and submitting it", async () => {
    const user = userEvent.setup();
    renderWithProviders(<AskUserCard request={baseRequest} />);

    const customInput = screen.getByTestId("ask-user-custom-input");
    await user.type(customInput, "Run migration with --dry-run first");

    const submitBtn = screen.getByTestId("ask-user-submit");
    await user.click(submitBtn);

    expect(completeSpy).toHaveBeenCalledWith(
      baseRequest,
      "Run migration with --dry-run first",
    );
  });

  it("renders settled state when request has already been answered", () => {
    const answeredRequest: AskUserDecisionRequest = {
      ...baseRequest,
      selectedAnswer: "Apply migration now",
    };

    renderWithProviders(<AskUserCard request={answeredRequest} />);

    expect(screen.getByTestId("ask-user-card-settled")).toBeInTheDocument();
    expect(screen.queryByTestId("ask-user-submit")).not.toBeInTheDocument();
    expect(screen.getByText("Apply migration now")).toBeInTheDocument();
  });
});
