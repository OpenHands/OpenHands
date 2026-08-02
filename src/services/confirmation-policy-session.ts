import type { ConfirmationPolicyBase } from "@openhands/typescript-client";

// Mirrors the CLI's process-local confirmation preference: selecting a policy
// updates the running conversation and seeds later local conversations, but it
// does not rewrite the user's persisted verification settings.
let confirmationPolicy: ConfirmationPolicyBase | null = null;

export const getSessionConfirmationPolicy = () => confirmationPolicy;

export const setSessionConfirmationPolicy = (
  policy: ConfirmationPolicyBase | null,
) => {
  confirmationPolicy = policy;
};
