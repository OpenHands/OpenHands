import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, test, vi } from "vitest";
import BillingService from "#/api/billing-service/billing-service.api";
import OptionService from "#/api/option-service/option-service.api";
import { PaymentForm } from "#/components/features/payment/payment-form";
import { renderWithProviders } from "../../../../test-utils";

// Mock the stripe checkout hook to avoid JSDOM navigation issues
const mockMutate = vi.fn().mockResolvedValue(undefined);
vi.mock("#/hooks/mutation/stripe/use-create-stripe-checkout-session", () => ({
  useCreateStripeCheckoutSession: () => ({
    mutate: mockMutate,
    mutateAsync: vi.fn().mockResolvedValue(undefined),
    isPending: false,
  }),
}));

describe("PaymentForm", () => {
  const getBalanceSpy = vi.spyOn(BillingService, "getBalance");
  const createCheckoutSessionSpy = vi.spyOn(
    BillingService,
    "createCheckoutSession",
  );
  const getConfigSpy = vi.spyOn(OptionService, "getConfig");

  const renderPaymentForm = () => renderWithProviders(<PaymentForm />);

  beforeEach(() => {
    // useBalance hook will return the balance only if the app_mode is "saas" and the billing feature is enabled
    // @ts-expect-error - partial mock for testing
    getConfigSpy.mockResolvedValue({
      app_mode: "saas",
      posthog_client_key: "456",
      feature_flags: {
        enable_billing: true,
        hide_llm_settings: false,
        enable_jira: false,
        enable_jira_dc: false,
        enable_linear: false,
      },
    });
  });

  afterEach(() => {
    vi.clearAllMocks();
    mockMutate.mockClear();
  });

  it("should render the users current balance", async () => {
    getBalanceSpy.mockResolvedValue("100.50");
    renderPaymentForm();

    await waitFor(() => {
      const balance = screen.getByTestId("user-balance");
      expect(balance).toHaveTextContent("$100.50");
    });
  });

  it("should render the users current balance to two decimal places", async () => {
    getBalanceSpy.mockResolvedValue("100");
    renderPaymentForm();

    await waitFor(() => {
      const balance = screen.getByTestId("user-balance");
      expect(balance).toHaveTextContent("$100.00");
    });
  });

  test("the user can top-up a specific amount", async () => {
    const user = userEvent.setup();
    renderPaymentForm();

    const topUpInput = await screen.findByTestId("top-up-input");
    await user.type(topUpInput, "50");

    const topUpButton = screen.getByText("PAYMENT$ADD_CREDIT");
    await user.click(topUpButton);

    expect(mockMutate).toHaveBeenCalledWith({ amount: 50 });
  });

  it("should only accept integer values", async () => {
    const user = userEvent.setup();
    renderPaymentForm();

    const topUpInput = await screen.findByTestId("top-up-input");
    await user.type(topUpInput, "50");

    const topUpButton = screen.getByText("PAYMENT$ADD_CREDIT");
    await user.click(topUpButton);

    expect(mockMutate).toHaveBeenCalledWith({ amount: 50 });
  });

  it("should disable the top-up button if the user enters an invalid amount", async () => {
    const user = userEvent.setup();
    renderPaymentForm();

    const topUpButton = screen.getByText("PAYMENT$ADD_CREDIT");
    expect(topUpButton).toBeDisabled();

    const topUpInput = await screen.findByTestId("top-up-input");
    await user.type(topUpInput, "  ");

    expect(topUpButton).toBeDisabled();
  });

  it("should disable the top-up button after submission", async () => {
    const user = userEvent.setup();
    renderPaymentForm();

    const topUpInput = await screen.findByTestId("top-up-input");
    await user.type(topUpInput, "50");

    const topUpButton = screen.getByText("PAYMENT$ADD_CREDIT");
    await user.click(topUpButton);

    expect(topUpButton).toBeDisabled();
  });

  describe("prevent submission if", () => {
    test("user enters a negative amount", async () => {
      const user = userEvent.setup();
      renderPaymentForm();

      const topUpInput = await screen.findByTestId("top-up-input");
      await user.type(topUpInput, "-50");

      const topUpButton = screen.getByText("PAYMENT$ADD_CREDIT");
      await user.click(topUpButton);

      expect(mockMutate).not.toHaveBeenCalled();
    });

    test("user enters an empty string", async () => {
      const user = userEvent.setup();
      renderPaymentForm();

      const topUpInput = await screen.findByTestId("top-up-input");
      await user.type(topUpInput, "     ");

      const topUpButton = screen.getByText("PAYMENT$ADD_CREDIT");
      await user.click(topUpButton);

      expect(mockMutate).not.toHaveBeenCalled();
    });

    test("user enters a non-numeric value", async () => {
      const user = userEvent.setup();
      renderPaymentForm();

      // With type="number", the browser would prevent non-numeric input,
      // but we'll test the validation logic anyway
      const topUpInput = await screen.findByTestId("top-up-input");
      await user.type(topUpInput, "abc");

      const topUpButton = screen.getByText("PAYMENT$ADD_CREDIT");
      await user.click(topUpButton);

      expect(mockMutate).not.toHaveBeenCalled();
    });

    test("user enters less than the minimum amount", async () => {
      const user = userEvent.setup();
      renderPaymentForm();

      const topUpInput = await screen.findByTestId("top-up-input");
      await user.type(topUpInput, "9"); // test assumes the minimum is 10

      const topUpButton = screen.getByText("PAYMENT$ADD_CREDIT");
      await user.click(topUpButton);

      expect(mockMutate).not.toHaveBeenCalled();
    });

    test("user enters a decimal value", async () => {
      const user = userEvent.setup();
      renderPaymentForm();

      // With step="1", the browser would validate this, but we'll test our validation logic
      const topUpInput = await screen.findByTestId("top-up-input");
      await user.type(topUpInput, "50.5");

      const topUpButton = screen.getByText("PAYMENT$ADD_CREDIT");
      await user.click(topUpButton);

      expect(mockMutate).not.toHaveBeenCalled();
    });
  });
});
