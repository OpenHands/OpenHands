import { openHands } from "../open-hands-axios";
import {
  ResendEmailVerificationParams,
  ResendEmailVerificationResponse,
} from "./email.types";

/**
 * Email Service API - Handles all email-related API endpoints
 */
export const emailService = {
  /**
   * Resend email verification to the user's registered email address
   * @param userId - Optional user ID to send verification email for
   * @param isAuthFlow - Whether this is part of the authentication flow
   * @returns The response message indicating the email was sent
   */
  resendEmailVerification: async ({
    userId,
    isAuthFlow,
  }: ResendEmailVerificationParams): Promise<ResendEmailVerificationResponse> => {
    const body: { user_id?: string; is_auth_flow?: boolean } = {};
    if (userId) {
      body.user_id = userId;
    }
    if (isAuthFlow !== undefined) {
      body.is_auth_flow = isAuthFlow;
    }
    const { data } = await openHands.put<ResendEmailVerificationResponse>(
      "/api/email/resend",
      body,
      { withCredentials: true },
    );
    return data;
  },
};
