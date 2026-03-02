import { openHands } from "../open-hands-axios";

/**
 * Billing Service API - Handles all billing-related API endpoints
 */
class BillingService {
  /**
   * Create a Stripe checkout session for credit purchase
   * @param amount The amount to charge in dollars
   * @returns The redirect URL for the checkout session
   */
  static async createCheckoutSession(amount: number): Promise<string> {
    const { data } = await openHands.post(
      "/api/billing/create-checkout-session",
      {
        amount,
      },
    );
    return data.redirect_url;
  }

  /**
   * Create a customer setup session for payment method management
   * @returns The redirect URL for the customer setup session
   */
  static async createBillingSessionResponse(): Promise<string> {
    const { data } = await openHands.post(
      "/api/billing/create-customer-setup-session",
    );
    return data.redirect_url;
  }

  /**
   * Get the user's current credit balance
   * @returns The user's credit balance as a string
   */
  static async getBalance(): Promise<string> {
    const { data } = await openHands.get<{ credits: string }>(
      "/api/billing/credits",
    );
    return data.credits;
  }
}

export default BillingService;
