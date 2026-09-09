import {
  FEATURE_DEV_STATUSES,
  FEATURE_DEV_TICKET_STATUSES,
} from "./feature-developer-constants";

export type FeatureDevStatus = (typeof FEATURE_DEV_STATUSES)[number];
export type FeatureDevTicketStatus =
  (typeof FEATURE_DEV_TICKET_STATUSES)[number];

export interface FeatureDevTicket {
  id: string;
  run_id: string;
  card_id: string | null;
  title: string;
  status: FeatureDevTicketStatus;
  branch_name: string | null;
  session_id: string | null;
  estimate_usd: number | null;
  actual_usd: number | null;
  error: string | null;
  started_at: string | null;
  finished_at: string | null;
  position: number;
}

export interface FeatureDevRun {
  id: string;
  project_id: string;
  spec_text: string;
  board_id: string | null;
  status: FeatureDevStatus;
  current_ticket_index: number;
  total_estimate_usd: number;
  total_actual_usd: number;
  max_concurrent_agents: number;
  continue_on_failure: boolean;
  created_at: string;
  updated_at: string;
  tickets: FeatureDevTicket[];
}

export interface StartFeatureDevRunPayload {
  project_id: string;
  spec_text: string;
  max_concurrent_agents?: number;
  continue_on_failure?: boolean;
}

export interface FeatureDevReport {
  markdown: string;
}
