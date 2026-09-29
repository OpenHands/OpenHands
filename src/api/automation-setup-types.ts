export type AutomationSetupKind = "prompt" | "plugin" | "custom";
export type AutomationSetupTriggerKind = "cron" | "event";
export type AutomationSetupFrequency =
  | "once"
  | "hourly"
  | "daily"
  | "weekdays"
  | "weekly"
  | "custom";
export type AutomationSetupFieldUpdateSource = "agent" | "user";

export interface AutomationSetupFormValues {
  kind: AutomationSetupKind;
  name: string;
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
  form?: AutomationSetupFormPatch;
  serverDraftId?: string;
  materializedAutomationId?: string | null;
  fieldMetadata?: Partial<
    Record<AutomationSetupField, AutomationSetupFieldMetadata>
  >;
  appliedAgentEventIds?: string[];
}
