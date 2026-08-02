import type { ConfirmationPolicyBase } from "@openhands/typescript-client";
import type { Backend } from "#/api/backend-registry/types";

export interface ConfirmationPolicySessionScope {
  backendId: string;
  connectionRevision: number;
}

// Mirrors the CLI's process-local confirmation preference: selecting a policy
// updates the running conversation and seeds later local conversations on the
// same backend connection, but it does not rewrite persisted verification
// settings or leak to another backend/credential revision.
const confirmationPolicies = new Map<string, ConfirmationPolicyBase>();

const getScopeKey = ({
  backendId,
  connectionRevision,
}: ConfirmationPolicySessionScope) =>
  JSON.stringify([backendId, connectionRevision]);

export const getConfirmationPolicySessionScope = (
  backend: Pick<Backend, "id" | "connectionRevision">,
): ConfirmationPolicySessionScope => ({
  backendId: backend.id,
  connectionRevision: backend.connectionRevision ?? 0,
});

export const getSessionConfirmationPolicy = (
  scope: ConfirmationPolicySessionScope,
) => confirmationPolicies.get(getScopeKey(scope)) ?? null;

export const setSessionConfirmationPolicy = (
  scope: ConfirmationPolicySessionScope,
  policy: ConfirmationPolicyBase,
) => {
  confirmationPolicies.set(getScopeKey(scope), policy);
};

export const clearSessionConfirmationPolicies = () => {
  confirmationPolicies.clear();
};
