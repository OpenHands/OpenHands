import { afterEach, describe, expect, it, vi } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "test-utils";
import { LikertScale } from "#/components/features/feedback/likert-scale";
import { I18nKey } from "#/i18n/declaration";

// Mock the mutation hook
vi.mock("#/hooks/mutation/use-submit-conversation-feedback", () => ({
  useSubmitConversationFeedback: () => ({
    mutate: vi.fn(),
  }),
}));

describe("LikertScale", () => {
  const user = userEvent.setup();

  afterEach(() => {
    vi.clearAllMocks();
  });

  it("should render with proper localized text for rating prompt", () => {
    renderWithProviders(<LikertScale eventId={1} />);

    // Check that the rating prompt is displayed with proper translation key
    expect(
      screen.getByText(I18nKey.FEEDBACK$RATE_AGENT_PERFORMANCE),
    ).toBeInTheDocument();
  });

  it("should show localized feedback reasons when rating is 3 or below", async () => {
    renderWithProviders(<LikertScale eventId={1} />);

    // Click on a rating of 3 (which should show reasons)
    const threeStarButton = screen.getAllByRole("button")[2]; // 3rd button (rating 3)
    await user.click(threeStarButton);

    // Wait for reasons to appear
    await waitFor(() => {
      expect(
        screen.getByText(I18nKey.FEEDBACK$SELECT_REASON),
      ).toBeInTheDocument();
    });

    // Check that all feedback reasons are properly localized
    expect(
      screen.getByText(I18nKey.FEEDBACK$REASON_MISUNDERSTOOD_INSTRUCTION),
    ).toBeInTheDocument();
    expect(
      screen.getByText(I18nKey.FEEDBACK$REASON_FORGOT_CONTEXT),
    ).toBeInTheDocument();
    expect(
      screen.getByText(I18nKey.FEEDBACK$REASON_UNNECESSARY_CHANGES),
    ).toBeInTheDocument();
    expect(
      screen.getByText(I18nKey.FEEDBACK$REASON_SHOULD_ASK_FIRST),
    ).toBeInTheDocument();
    expect(
      screen.getByText(I18nKey.FEEDBACK$REASON_DIDNT_FINISH_JOB),
    ).toBeInTheDocument();
    expect(screen.getByText(I18nKey.FEEDBACK$REASON_OTHER)).toBeInTheDocument();
  });

  it("should show countdown message with proper localization", async () => {
    renderWithProviders(<LikertScale eventId={1} />);

    // Click on a rating of 2 (which should show reasons and countdown)
    const twoStarButton = screen.getAllByRole("button")[1]; // 2nd button (rating 2)
    await user.click(twoStarButton);

    // Wait for countdown to appear
    await waitFor(() => {
      expect(
        screen.getByText(I18nKey.FEEDBACK$SELECT_REASON_COUNTDOWN),
      ).toBeInTheDocument();
    });
  });

  it("should show thank you message after submission", () => {
    renderWithProviders(
      <LikertScale eventId={1} initiallySubmitted={true} initialRating={4} />,
    );

    // Check that thank you message is displayed with proper translation key
    expect(
      screen.getByText(I18nKey.FEEDBACK$THANK_YOU_FOR_FEEDBACK),
    ).toBeInTheDocument();
  });

  it("should render all 5 star rating buttons", () => {
    renderWithProviders(<LikertScale eventId={1} />);

    // Check that all 5 star buttons are rendered
    const starButtons = screen.getAllByRole("button");
    expect(starButtons).toHaveLength(5);

    // Check that each button has proper aria-label
    for (let i = 1; i <= 5; i++) {
      expect(screen.getByLabelText(`Rate ${i} stars`)).toBeInTheDocument();
    }
  });

  it("should not show reasons for ratings above 3", async () => {
    renderWithProviders(<LikertScale eventId={1} />);

    // Click on a rating of 5 (which should NOT show reasons)
    const fiveStarButton = screen.getAllByRole("button")[4]; // 5th button (rating 5)
    await user.click(fiveStarButton);

    // Wait a bit to ensure reasons don't appear
    await waitFor(() => {
      expect(
        screen.queryByText(I18nKey.FEEDBACK$SELECT_REASON),
      ).not.toBeInTheDocument();
    });
  });
});
