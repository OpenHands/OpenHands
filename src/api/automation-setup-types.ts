export type AutomationSetupKind = "prompt" | "plugin" | "custom";

export interface AutomationSetupDraft {
  prompt: string;
  kind: AutomationSetupKind;
  plugins?: string[];
}
