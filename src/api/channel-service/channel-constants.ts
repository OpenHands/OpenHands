export const CHANNELS_PATH = "/channels";
export const CHANNELS_API_PATH = "/api/channels";
export const CHANNELS_MESSAGES_PATH = `${CHANNELS_API_PATH}/messages`;
export const SESSION_API_KEY_HEADER = "X-Session-API-Key";

export const CHANNEL_DIRECTIONS = ["inbound", "outbound"] as const;
export const CHANNEL_STATES = [
  "stopped",
  "running",
  "unconfigured",
  "error",
] as const;

export function channelStartPath(channelId: string): string {
  return `${CHANNELS_API_PATH}/${encodeURIComponent(channelId)}/start`;
}

export function channelStopPath(channelId: string): string {
  return `${CHANNELS_API_PATH}/${encodeURIComponent(channelId)}/stop`;
}

export function channelConfigPath(channelId: string): string {
  return `${CHANNELS_API_PATH}/${encodeURIComponent(channelId)}/config`;
}

export function channelDetailPath(channelId: string): string {
  return `${CHANNELS_API_PATH}/${encodeURIComponent(channelId)}`;
}
