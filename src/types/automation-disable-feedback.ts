export type AutomationDisableReason =
  | "no_longer_needed"
  | "unreliable"
  | "misconfigured"
  | "too_noisy"
  | "too_expensive"
  | "low_quality"
  | "other";

export interface AutomationDisableFeedback {
  reason: AutomationDisableReason;
  details?: string;
}

export interface AutomationDisableAnalyticsContext {
  backendKind: BackendKind;
  automationId: string;
  automationType: string;
  automationSource?: string;
  automationTemplateId?: string;
  disablementId: string;
}
import type { BackendKind } from "#/api/backend-registry/types";
