export interface AutomationWebhook {
  id: string;
  name: string;
  source: string;
  webhook_url: string;
  enabled: boolean;
  event_key_expr: string;
  signature_header: string;
  signature_scheme: string;
  created_at: string;
  updated_at: string;
}

export interface AutomationWebhooksResponse {
  webhooks: AutomationWebhook[];
  total: number;
}

export interface CreateAutomationWebhookRequest {
  name: string;
  source: string;
  event_key_expr?: string;
  signature_header?: string;
  signature_scheme?: "hmac_sha256_hex";
  webhook_secret?: string;
}

export interface CreatedAutomationWebhook extends AutomationWebhook {
  /** Present only when the server generated the signing secret. */
  webhook_secret?: string;
}
