/**
 * Custom secret with name, value, and optional description.
 * Used for creating/updating secrets via PUT /api/settings/secrets.
 */
export type CustomSecret = {
  name: string;
  value: string;
  description?: string;
};

/**
 * Custom secret metadata without the secret value.
 * Used for listing secrets via GET /api/settings/secrets.
 */
export type CustomSecretScope = "personal" | "organization";

export type CustomSecretWithoutValue = Omit<CustomSecret, "value"> & {
  scope?: CustomSecretScope;
  source_name?: string | null;
  writable?: boolean;
  deletable?: boolean;
};
