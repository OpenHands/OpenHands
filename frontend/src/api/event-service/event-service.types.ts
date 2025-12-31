export interface ConfirmationResponseRequest {
  accept: boolean;
  reason?: string;
}

export interface ConfirmationResponseResponse {
  success: boolean;
}
