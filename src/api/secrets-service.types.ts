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
export type CustomSecretWithoutValue = Omit<CustomSecret, "value"> & {
  /**
   * Whether the secret is personal or shared with the organization.
   * Only the Cloud App-API search endpoint reports this; personal secrets
   * keep their stored name, while a colliding org-shared secret is listed
   * under a display-only suffixed name (e.g. TOKEN_2) that the personal
   * delete/update endpoints cannot resolve. Local agent-server listings
   * omit the field and every entry is personal.
   */
  scope?: "personal" | "organization";
};
