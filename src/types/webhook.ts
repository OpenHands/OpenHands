/** A custom (non-built-in) webhook source, org-scoped. */
export interface CustomWebhook {
  id: string;
  org_id: string;
  name: string;
  source: string;
  webhook_url: string;
  event_key_expr: string;
  signature_header: string;
  enabled: boolean;
  created_at: string;
  updated_at: string;
}

/**
 * Response for webhook creation. `webhook_secret` is only present when the
 * server generated it (the caller didn't supply their own) — it is shown
 * exactly once and never echoed back afterward.
 */
export interface CustomWebhookCreateResponse extends CustomWebhook {
  webhook_secret?: string | null;
}

/** Response for secret rotation. Also shown exactly once. */
export interface CustomWebhookSecretResponse {
  webhook_secret: string;
}

export interface CustomWebhookListResponse {
  webhooks: CustomWebhook[];
  total: number;
}

export interface CreateWebhookRequest {
  name: string;
  source: string;
  event_key_expr?: string;
  signature_header?: string;
  webhook_secret?: string;
}

export interface UpdateWebhookRequest {
  name?: string;
  event_key_expr?: string;
  signature_header?: string;
  enabled?: boolean;
}
