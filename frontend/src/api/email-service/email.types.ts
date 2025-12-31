export interface ResendEmailVerificationParams {
  userId?: string | null;
  isAuthFlow?: boolean;
}

export interface ResendEmailVerificationResponse {
  message: string;
}
