import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import {
  AutomationDisableFeedbackModal,
  type AutomationDisableFeedback,
} from "#/components/features/automations/automation-disable-feedback-modal";
import { I18nKey } from "#/i18n/declaration";

vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: (key: string) => key }),
}));

function renderModal() {
  const onSubmit = vi.fn<(feedback: AutomationDisableFeedback) => void>();
  const onDismiss = vi.fn();

  render(
    <AutomationDisableFeedbackModal
      onSubmit={onSubmit}
      onDismiss={onDismiss}
    />,
  );

  return { onSubmit, onDismiss };
}

describe("AutomationDisableFeedbackModal", () => {
  it("offers every standard disablement reason", () => {
    renderModal();

    expect(screen.getAllByRole("radio")).toHaveLength(7);
    expect(
      screen.getByRole("radio", {
        name: I18nKey.AUTOMATIONS$DISABLE_REASON_NO_LONGER_NEEDED,
      }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("radio", {
        name: I18nKey.AUTOMATIONS$DISABLE_REASON_UNRELIABLE,
      }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("radio", {
        name: I18nKey.AUTOMATIONS$DISABLE_REASON_MISCONFIGURED,
      }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("radio", {
        name: I18nKey.AUTOMATIONS$DISABLE_REASON_TOO_NOISY,
      }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("radio", {
        name: I18nKey.AUTOMATIONS$DISABLE_REASON_TOO_EXPENSIVE,
      }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("radio", {
        name: I18nKey.AUTOMATIONS$DISABLE_REASON_LOW_QUALITY,
      }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("radio", {
        name: I18nKey.AUTOMATIONS$DISABLE_REASON_OTHER,
      }),
    ).toBeInTheDocument();
  });

  it("submits a structured reason with optional free-text context", async () => {
    const user = userEvent.setup();
    const { onSubmit } = renderModal();

    const submit = screen.getByTestId("submit-automation-disable-feedback");
    expect(submit).toBeDisabled();

    await user.click(
      screen.getByRole("radio", {
        name: I18nKey.AUTOMATIONS$DISABLE_REASON_OTHER,
      }),
    );
    await user.type(
      screen.getByLabelText(I18nKey.AUTOMATIONS$DISABLE_FEEDBACK_DETAILS_LABEL),
      "The notifications duplicate another workflow.",
    );
    await user.click(submit);

    expect(onSubmit).toHaveBeenCalledWith({
      reason: "other",
      details: "The notifications duplicate another workflow.",
    });
  });

  it("lets the user dismiss feedback without submitting it", async () => {
    const user = userEvent.setup();
    const { onSubmit, onDismiss } = renderModal();

    await user.click(screen.getByTestId("skip-automation-disable-feedback"));

    expect(onDismiss).toHaveBeenCalledTimes(1);
    expect(onSubmit).not.toHaveBeenCalled();
  });
});
