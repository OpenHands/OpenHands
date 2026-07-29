import { create } from "zustand";

export interface MetricsState {
  cost: number | null;
  max_budget_per_task: number | null;
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    cache_read_tokens: number;
    cache_write_tokens: number;
    context_window: number;
    per_turn_token: number;
  } | null;
}

export interface MetricsStore extends MetricsState {
  setMetrics: (metrics: MetricsState) => void;
}

const useMetricsStore = create<MetricsStore>((set) => ({
  cost: null,
  max_budget_per_task: null,
  usage: null,
  setMetrics: (metrics) => set(metrics),
}));

export default useMetricsStore;
