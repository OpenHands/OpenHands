import { CHANNEL_DIRECTIONS, CHANNEL_STATES } from "./channel-constants";

export type ChannelDirection = (typeof CHANNEL_DIRECTIONS)[number];
export type ChannelState = (typeof CHANNEL_STATES)[number];

export interface ChannelRoutingRule {
  pattern: string;
  project_id?: string;
  loop?: string;
  card_type?: string;
}

export interface ChannelStatus {
  state: ChannelState;
  mode?: string | null;
  last_error?: string | null;
  metrics?: Record<string, number>;
}

export interface ChannelConfigSummary {
  mode?: string;
  has_bot_token?: boolean;
  has_app_token?: boolean;
  webhook_host?: string;
  routing_rules?: ChannelRoutingRule[];
  cost_cap?: number | null;
  human_only?: boolean;
}

export interface ChannelRecord {
  id: string;
  type: string;
  status: ChannelStatus;
  config: ChannelConfigSummary;
}

export interface ChannelMessage {
  id: string;
  channel_id: string;
  direction: ChannelDirection;
  correlation_id: string;
  session_id: string | null;
  acked: boolean;
  created_at: string;
  source: string;
  channel_ref: string;
  author: string;
  text: string;
  thread_ref: string;
  timestamp: string;
  hold_for_human?: boolean;
  estimated_cost?: number;
}

export interface ChannelMessagePage {
  items: ChannelMessage[];
  limit: number;
  offset: number;
}

export interface ChannelConfigPayload {
  routing_rules?: ChannelRoutingRule[];
  cost_cap?: number | null;
  human_only?: boolean;
  bot_user_id?: string;
}

export interface ListMessagesFilters {
  channel?: string;
  direction?: ChannelDirection;
  correlation_id?: string;
  limit?: number;
  offset?: number;
}
