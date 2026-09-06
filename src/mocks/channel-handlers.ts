import { http, HttpResponse } from "msw";
import {
  CHANNELS_API_PATH,
  CHANNELS_MESSAGES_PATH,
} from "#/api/channel-service/channel-constants";
import type {
  ChannelConfigPayload,
  ChannelMessage,
  ChannelRecord,
} from "#/api/channel-service/channel-types";

let channels: ChannelRecord[] = [];
let messages: ChannelMessage[] = [];
let nextId = 1;

function now(): string {
  return new Date().toISOString();
}

function id(prefix: string): string {
  nextId += 1;
  return `${prefix}-${nextId}`;
}

export function resetChannelMockData() {
  channels = [
    {
      id: "slack",
      type: "slack",
      status: { state: "stopped", mode: "sockets", metrics: { posted: 0 } },
      config: {
        mode: "sockets",
        has_bot_token: true,
        has_app_token: true,
        routing_rules: [],
        human_only: false,
        cost_cap: null,
      },
    },
  ];
  messages = [];
  nextId = 1;
}

resetChannelMockData();

function channelOr404(idValue: string) {
  return channels.find((channel) => channel.id === idValue) ?? null;
}

export const CHANNEL_HANDLERS = [
  http.get(CHANNELS_API_PATH, () => HttpResponse.json(channels)),
  http.get(CHANNELS_MESSAGES_PATH, ({ request }) => {
    const url = new URL(request.url);
    const channel = url.searchParams.get("channel");
    const direction = url.searchParams.get("direction");
    const correlationId = url.searchParams.get("correlation_id");
    const limit = Number(url.searchParams.get("limit") || 50);
    const offset = Number(url.searchParams.get("offset") || 0);
    const items = messages.filter((row) => {
      if (channel && row.channel_id !== channel) return false;
      if (direction && row.direction !== direction) return false;
      if (correlationId && row.correlation_id !== correlationId) return false;
      return true;
    });
    return HttpResponse.json({
      items: items.slice(offset, offset + limit),
      limit,
      offset,
    });
  }),
  http.get(`${CHANNELS_API_PATH}/:id`, ({ params }) => {
    const found = channelOr404(String(params.id));
    if (!found) {
      return HttpResponse.json({ error: "not found" }, { status: 404 });
    }
    return HttpResponse.json(found);
  }),
  http.post(`${CHANNELS_API_PATH}/:id/start`, ({ params }) => {
    const found = channelOr404(String(params.id));
    if (!found) {
      return HttpResponse.json({ error: "not found" }, { status: 404 });
    }
    if (found.status.state !== "unconfigured") {
      found.status.state = "running";
    }
    return HttpResponse.json(found);
  }),
  http.post(`${CHANNELS_API_PATH}/:id/stop`, ({ params }) => {
    const found = channelOr404(String(params.id));
    if (!found) {
      return HttpResponse.json({ error: "not found" }, { status: 404 });
    }
    if (found.status.state !== "unconfigured") {
      found.status.state = "stopped";
    }
    return HttpResponse.json(found);
  }),
  http.put(`${CHANNELS_API_PATH}/:id/config`, async ({ params, request }) => {
    const found = channelOr404(String(params.id));
    if (!found) {
      return HttpResponse.json({ error: "not found" }, { status: 404 });
    }
    const payload = (await request.json()) as ChannelConfigPayload;
    found.config = {
      ...found.config,
      routing_rules: payload.routing_rules ?? found.config.routing_rules,
      cost_cap: payload.cost_cap ?? found.config.cost_cap,
      human_only: payload.human_only ?? found.config.human_only,
    };
    return HttpResponse.json(found);
  }),
];

export function seedChannelMessage(
  partial: Partial<ChannelMessage> = {},
): ChannelMessage {
  const created: ChannelMessage = {
    id: partial.id ?? id("msg"),
    channel_id: partial.channel_id ?? "slack",
    direction: partial.direction ?? "inbound",
    correlation_id: partial.correlation_id ?? id("cid"),
    session_id: partial.session_id ?? "sess-1",
    acked: partial.acked ?? true,
    created_at: partial.created_at ?? now(),
    source: partial.source ?? "slack",
    channel_ref: partial.channel_ref ?? "C1",
    author: partial.author ?? "U1",
    text: partial.text ?? "hello",
    thread_ref: partial.thread_ref ?? "t-1",
    timestamp: partial.timestamp ?? now(),
  };
  messages.unshift(created);
  return created;
}
