import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "test-utils";
import { AddCreditsModal } from "#/components/features/org/add-credits-modal";
import BillingService from "#/api/billing-service/billing-service.api";

vi.mock("react-i18next", async (importOriginal) => ({
  ...(await importOriginal<typeof import("react-i18next")>()),
  useTranslation: () => ({
    t: (key: string) => key,
    i18n: {
      changeLanguage: vi.fn(),
    },
  }),
}));

describe("AddCreditsModal", () => {
  const onCloseMock = vi.fn();

  const renderModal = () => {
    const user = userEvent.setup();
    renderWithProviders(<AddCreditsModal onClose={onCloseMock} />);
    return { user };
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe("Rendering", () => {
    it("should render the form with correct elements", () => {
      renderModal();

      expect(screen.getByTestId("add-credits-form")).toBeInTheDocument();
      expect(screen.getByTestId("amount-input")).toBeInTheDocument();
      expect(screen.getByRole("button", { name: /ORG\$NEXT/i })).toBeInTheDocument();
    });

    it("should display the title", () => {
      renderModal();

      expect(screen.getByText("ORG$ADD_CREDITS")).toBeInTheDocument();
    });
  });

  describe("Button State Management", () => {
    it("should enable submit button initially when modal opens", () => {
      renderModal();

      const nextButton = screen.getByRole("button", { name: /ORG\$NEXT/i });
      expect(nextButton).not.toBeDisabled();
    });

    it("should enable submit button when input contains invalid value", async () => {
      const { user } = renderModal();
      const amountInput = screen.getByTestId("amount-input");
      const nextButton = screen.getByRole("button", { name: /ORG\$NEXT/i });

      await user.type(amountInput, "-50");

      expect(nextButton).not.toBeDisabled();
    });

    it("should enable submit button when input contains valid value", async () => {
      const { user } = renderModal();
      const amountInput = screen.getByTestId("amount-input");
      const nextButton = screen.getByRole("button", { name: /ORG\$NEXT/i });

      await user.type(amountInput, "100");

      expect(nextButton).not.toBeDisabled();
    });

    it("should enable submit button after validation error is shown", async () => {
      const { user } = renderModal();
      const amountInput = screen.getByTestId("amount-input");
      const nextButton = screen.getByRole("button", { name: /ORG\$NEXT/i });

      await user.type(amountInput, "9");
      await user.click(nextButton);

      await waitFor(() => {
        expect(screen.getByTestId("amount-error")).toBeInTheDocument();
      });

      expect(nextButton).not.toBeDisabled();
    });
  });

  describe("Input Attributes & Placeholder", () => {
    it("should have min attribute set to 10", () => {
      renderModal();

      const amountInput = screen.getByTestId("amount-input");
      expect(amountInput).toHaveAttribute("min", "10");
    });

    it("should have max attribute set to 25000", () => {
      renderModal();

      const amountInput = screen.getByTestId("amount-input");
      expect(amountInput).toHaveAttribute("max", "25000");
    });

    it("should have step attribute set to 1", () => {
      renderModal();

      const amountInput = screen.getByTestId("amount-input");
      expect(amountInput).toHaveAttribute("step", "1");
    });
  });

  describe("Error Message Display", () => {
    it("should not display error message initially when modal opens", () => {
      renderModal();

      const errorMessage = screen.queryByTestId("amount-error");
      expect(errorMessage).not.toBeInTheDocument();
    });

    it("should display error message after submitting amount above maximum", async () => {
      const { user } = renderModal();
      const amountInput = screen.getByTestId("amount-input");
      const nextButton = screen.getByRole("button", { name: /ORG\$NEXT/i });

      await user.type(amountInput, "25001");
      await user.click(nextButton);

      await waitFor(() => {
        const errorMessage = screen.getByTestId("amount-error");
        expect(errorMessage).toHaveTextContent("PAYMENT$ERROR_MAXIMUM_AMOUNT");
      });
    });

    it("should display error message after submitting decimal value", async () => {
      const { user } = renderModal();
      const amountInput = screen.getByTestId("amount-input");
      const nextButton = screen.getByRole("button", { name: /ORG\$NEXT/i });

      await user.type(amountInput, "50.5");
      await user.click(nextButton);

      await waitFor(() => {
        const errorMessage = screen.getByTestId("amount-error");
        expect(errorMessage).toHaveTextContent("PAYMENT$ERROR_MUST_BE_WHOLE_NUMBER");
      });
    });

    it("should display error message after submitting amount below minimum", async () => {
      const { user } = renderModal();
      const amountInput = screen.getByTestId("amount-input");
      const nextButton = screen.getByRole("button", { name: /ORG\$NEXT/i });

      await user.type(amountInput, "9");
      await user.click(nextButton);

      await waitFor(() => {
        const errorMessage = screen.getByTestId("amount-error");
        expect(errorMessage).toHaveTextContent("PAYMENT$ERROR_MINIMUM_AMOUNT");
      });
    });

    it("should display error message after submitting negative amount", async () => {
      const { user } = renderModal();
      const amountInput = screen.getByTestId("amount-input");
      const nextButton = screen.getByRole("button", { name: /ORG\$NEXT/i });

      await user.type(amountInput, "-50");
      await user.click(nextButton);

      await waitFor(() => {
        const errorMessage = screen.getByTestId("amount-error");
        expect(errorMessage).toHaveTextContent("PAYMENT$ERROR_NEGATIVE_AMOUNT");
      });
    });

    it("should replace error message when submitting different invalid value", async () => {
      const { user } = renderModal();
      const amountInput = screen.getByTestId("amount-input");
      const nextButton = screen.getByRole("button", { name: /ORG\$NEXT/i });

      await user.type(amountInput, "9");
      await user.click(nextButton);

      await waitFor(() => {
        const errorMessage = screen.getByTestId("amount-error");
        expect(errorMessage).toHaveTextContent("PAYMENT$ERROR_MINIMUM_AMOUNT");
      });

      await user.clear(amountInput);
      await user.type(amountInput, "25001");
      await user.click(nextButton);

      await waitFor(() => {
        const errorMessage = screen.getByTestId("amount-error");
        expect(errorMessage).toHaveTextContent("PAYMENT$ERROR_MAXIMUM_AMOUNT");
      });
    });
  });

  describe("Form Submission Behavior", () => {
    it("should prevent submission when amount is invalid", async () => {
      const createCheckoutSessionSpy = vi.spyOn(
        BillingService,
        "createCheckoutSession",
      );
      const { user } = renderModal();
      const amountInput = screen.getByTestId("amount-input");
      const nextButton = screen.getByRole("button", { name: /ORG\$NEXT/i });

      await user.type(amountInput, "9");
      await user.click(nextButton);

      expect(createCheckoutSessionSpy).not.toHaveBeenCalled();
      await waitFor(() => {
        const errorMessage = screen.getByTestId("amount-error");
        expect(errorMessage).toHaveTextContent("PAYMENT$ERROR_MINIMUM_AMOUNT");
      });
    });

    it("should call createCheckoutSession with correct amount when valid", async () => {
      const createCheckoutSessionSpy = vi.spyOn(
        BillingService,
        "createCheckoutSession",
      );
      const { user } = renderModal();
      const amountInput = screen.getByTestId("amount-input");
      const nextButton = screen.getByRole("button", { name: /ORG\$NEXT/i });

      await user.type(amountInput, "1000");
      await user.click(nextButton);

      expect(createCheckoutSessionSpy).toHaveBeenCalledWith(1000);
      const errorMessage = screen.queryByTestId("amount-error");
      expect(errorMessage).not.toBeInTheDocument();
    });

    it("should not call createCheckoutSession when validation fails", async () => {
      const createCheckoutSessionSpy = vi.spyOn(
        BillingService,
        "createCheckoutSession",
      );
      const { user } = renderModal();
      const amountInput = screen.getByTestId("amount-input");
      const nextButton = screen.getByRole("button", { name: /ORG\$NEXT/i });

      await user.type(amountInput, "-50");
      await user.click(nextButton);

      expect(createCheckoutSessionSpy).not.toHaveBeenCalled();
      await waitFor(() => {
        const errorMessage = screen.getByTestId("amount-error");
        expect(errorMessage).toHaveTextContent("PAYMENT$ERROR_NEGATIVE_AMOUNT");
      });
    });

    it("should close modal on successful submission", async () => {
      vi.spyOn(BillingService, "createCheckoutSession").mockResolvedValue(
        "https://checkout.stripe.com/test-session",
      );

      const { user } = renderModal();
      const amountInput = screen.getByTestId("amount-input");
      const nextButton = screen.getByRole("button", { name: /ORG\$NEXT/i });

      await user.type(amountInput, "1000");
      await user.click(nextButton);

      await waitFor(() => {
        expect(onCloseMock).toHaveBeenCalled();
      });
    });

    it("should allow API call when validation passes and clear any previous errors", async () => {
      const createCheckoutSessionSpy = vi.spyOn(
        BillingService,
        "createCheckoutSession",
      );

      const { user } = renderModal();
      const amountInput = screen.getByTestId("amount-input");
      const nextButton = screen.getByRole("button", { name: /ORG\$NEXT/i });

      // First submit invalid value
      await user.type(amountInput, "9");
      await user.click(nextButton);

      await waitFor(() => {
        expect(screen.getByTestId("amount-error")).toBeInTheDocument();
      });

      // Then submit valid value
      await user.clear(amountInput);
      await user.type(amountInput, "100");
      await user.click(nextButton);

      expect(createCheckoutSessionSpy).toHaveBeenCalledWith(100);
      const errorMessage = screen.queryByTestId("amount-error");
      expect(errorMessage).not.toBeInTheDocument();
    });
  });

  describe("Edge Cases", () => {
    it("should handle zero value correctly", async () => {
      const { user } = renderModal();
      const amountInput = screen.getByTestId("amount-input");
      const nextButton = screen.getByRole("button", { name: /ORG\$NEXT/i });

      await user.type(amountInput, "0");
      await user.click(nextButton);

      await waitFor(() => {
        const errorMessage = screen.getByTestId("amount-error");
        expect(errorMessage).toHaveTextContent("PAYMENT$ERROR_MINIMUM_AMOUNT");
      });
    });

    it("should handle whitespace-only input correctly", async () => {
      const createCheckoutSessionSpy = vi.spyOn(
        BillingService,
        "createCheckoutSession",
      );
      const { user } = renderModal();
      const amountInput = screen.getByTestId("amount-input");
      const nextButton = screen.getByRole("button", { name: /ORG\$NEXT/i });

      // Number inputs typically don't accept spaces, but test the behavior
      await user.type(amountInput, "   ");
      await user.click(nextButton);

      // Should not call API (empty/invalid input)
      expect(createCheckoutSessionSpy).not.toHaveBeenCalled();
    });
  });

  describe("Modal Interaction", () => {
    it("should call onClose when cancel button is clicked", async () => {
      const { user } = renderModal();

      const cancelButton = screen.getByRole("button", { name: /close/i });
      await user.click(cancelButton);

      expect(onCloseMock).toHaveBeenCalledOnce();
    });
  });
});
