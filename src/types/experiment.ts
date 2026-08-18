import type { AutomationTrigger } from "./automation";

/** A plugin to load, matching the backend's PluginSource shape. */
export interface ExperimentPluginSource {
  source: string;
  ref?: string;
  repo_path?: string;
}

/** One A/B test arm: its own plugin set, an optional model override, and a relative weight. */
export interface ExperimentVariant {
  name: string;
  weight: number;
  model?: string;
  plugins: ExperimentPluginSource[];
}

/**
 * Body for creating a plugin automation with A/B experiment variants.
 * Mirrors the backend's CreatePluginAutomationRequest with `variants` set
 * (mutually exclusive with a flat `plugins` list).
 */
export interface CreateExperimentAutomationRequest {
  name: string;
  prompt: string;
  trigger: AutomationTrigger;
  variants: ExperimentVariant[];
  experiment_id: string;
  model?: string;
  timeout?: number;
  keep_alive?: boolean;
  enabled?: boolean;
}
