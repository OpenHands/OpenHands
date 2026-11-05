import type { AutomationRepository } from "#/types/automation";

export type AutomationSetupKind = "prompt" | "plugin" | "custom";
export type AutomationSetupTriggerKind = "cron" | "event";
export type AutomationSetupFrequency =
  "once" | "hourly" | "daily" | "weekdays" | "weekly" | "custom";
export type AutomationSetupFieldUpdateSource = "agent" | "user";

export interface AutomationSetupFormValues {
  kind: AutomationSetupKind;
  name: string;
  /** Service account id selected for a new automation; empty means the caller. */
  runAsUserId: string;
  prompt: string;
  repository: string;
  pluginSource: string;
  pluginRef: string;
  pluginList: string;
  customCode: string;
  entrypoint: string;
  setupScriptPath: string;
  setupScript: string;
  triggerKind: AutomationSetupTriggerKind;
  frequency: AutomationSetupFrequency;
  time: string;
  scheduleDateTime: string;
  timezone: string;
  weekday: string;
  customSchedule: string;
  eventSource: string;
  eventKey: string;
  eventFilter: string;
  model: string;
  agentProfileId: string;
  showTimeout: boolean;
  timeoutSeconds: string;
}

export type AutomationSetupField = keyof AutomationSetupFormValues;
export type AutomationSetupFormPatch = Partial<AutomationSetupFormValues>;

export interface AutomationSetupFieldMetadata {
  updatedBy: AutomationSetupFieldUpdateSource;
  updatedAt: string;
  userDirty: boolean;
}

export interface AutomationSetupPatchResult {
  applied: AutomationSetupField[];
  skipped: AutomationSetupField[];
  duplicate: boolean;
}

export interface AutomationSetupDraft {
  prompt: string;
  kind: AutomationSetupKind;
  plugins?: string[];
  editingAutomationId?: string;
  serverDraftId?: string;
  materializedAutomationId?: string | null;
  existingCustomTarballPath?: string;
  /** Repositories the edited automation clones, as the service stores them. */
  existingRepositories?: AutomationRepository[];
  /** Whether the edited automation's repositories can be changed in place. */
  canUpdateRepositories?: boolean;
  /**
   * Fields of the edited event trigger that the form does not edit (such as
   * `destination`), sent back unchanged so a save does not reset them.
   */
  existingEventTrigger?: Record<string, unknown>;
  form?: AutomationSetupFormPatch;
  fieldMetadata?: Partial<
    Record<AutomationSetupField, AutomationSetupFieldMetadata>
  >;
  appliedAgentEventIds?: string[];
}
