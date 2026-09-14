import { AgentServerClient } from "@openhands/typescript-client/clients";
import { getAgentServerClientOptions } from "./agent-server-client-options";
import {
  OPENAI_SUBSCRIPTION_DEVICE_POLL_PATH,
  OPENAI_SUBSCRIPTION_DEVICE_START_PATH,
  OPENAI_SUBSCRIPTION_LOGOUT_PATH,
  OPENAI_SUBSCRIPTION_MODELS_PATH,
  OPENAI_SUBSCRIPTION_STATUS_PATH,
  OPENAI_SUBSCRIPTION_VENDOR,
} from "#/constants/llm-subscription";

type RawSubscriptionStatus = Record<string, unknown>;
type RawDeviceStart = Record<string, unknown>;

export interface LLMSubscriptionStatus {
  vendor: typeof OPENAI_SUBSCRIPTION_VENDOR;
  connected: boolean;
  accountEmail: string | null;
  expiresAt: string | number | null;
}

export interface LLMSubscriptionDeviceChallenge {
  deviceCode: string;
  userCode: string;
  verificationUri: string;
  verificationUriComplete: string | null;
  expiresAt: string | number | null;
  intervalSeconds: number | null;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

const readString = (
  value: Record<string, unknown>,
  keys: string[],
): string | null => {
  for (const key of keys) {
    const candidate = value[key];
    if (typeof candidate === "string" && candidate.trim().length > 0) {
      return candidate.trim();
    }
  }
  return null;
};

const readNumber = (
  value: Record<string, unknown>,
  keys: string[],
): number | null => {
  for (const key of keys) {
    const candidate = value[key];
    if (typeof candidate === "number" && Number.isFinite(candidate)) {
      return candidate;
    }
  }
  return null;
};

const readBoolean = (
  value: Record<string, unknown>,
  keys: string[],
): boolean => {
  for (const key of keys) {
    const candidate = value[key];
    if (typeof candidate === "boolean") {
      return candidate;
    }
  }
  return false;
};

interface SubscriptionRequestOptions {
  method?: "GET" | "POST";
  jsonBody?: unknown;
}

async function requestSubscriptionEndpoint(
  path: string,
  options: SubscriptionRequestOptions = {},
): Promise<unknown> {
  const { host, apiKey } = getAgentServerClientOptions();
  const client = new AgentServerClient({ host, apiKey });
  try {
    return await client.request<unknown>({
      method: options.method ?? "GET",
      path,
      body: options.jsonBody,
      responseType: "json",
    });
  } finally {
    client.close();
  }
}

function normalizeModels(raw: unknown): string[] {
  if (Array.isArray(raw)) {
    return raw.filter((item): item is string => typeof item === "string");
  }
  if (isRecord(raw) && Array.isArray(raw.models)) {
    return raw.models.filter(
      (item): item is string => typeof item === "string",
    );
  }
  return [];
}

function normalizeStatus(raw: RawSubscriptionStatus): LLMSubscriptionStatus {
  return {
    vendor: OPENAI_SUBSCRIPTION_VENDOR,
    connected: readBoolean(raw, ["connected", "authenticated", "is_connected"]),
    accountEmail: readString(raw, ["account_email", "email", "account"]),
    expiresAt:
      readString(raw, ["expires_at", "expiresAt"]) ??
      readNumber(raw, ["expires_at", "expiresAt"]),
  };
}

function normalizeDeviceChallenge(
  raw: RawDeviceStart,
): LLMSubscriptionDeviceChallenge {
  const deviceCode = readString(raw, ["device_code", "deviceCode"]);
  const userCode = readString(raw, ["user_code", "userCode"]);
  const verificationUri = readString(raw, [
    "verification_uri",
    "verificationUri",
    "verification_url",
    "verificationUrl",
  ]);

  if (!deviceCode || !userCode || !verificationUri) {
    throw new Error("Subscription device login response is incomplete");
  }

  return {
    deviceCode,
    userCode,
    verificationUri,
    verificationUriComplete: readString(raw, [
      "verification_uri_complete",
      "verificationUriComplete",
      "verification_url_complete",
      "verificationUrlComplete",
    ]),
    expiresAt:
      readString(raw, ["expires_at", "expiresAt"]) ??
      readNumber(raw, ["expires_at", "expiresAt", "expires_in", "expiresIn"]),
    intervalSeconds: readNumber(raw, [
      "interval",
      "interval_seconds",
      "intervalSeconds",
    ]),
  };
}

class LLMSubscriptionService {
  static async getOpenAIModels(): Promise<string[]> {
    const response = await requestSubscriptionEndpoint(
      OPENAI_SUBSCRIPTION_MODELS_PATH,
    );
    return normalizeModels(response);
  }

  static async getOpenAIStatus(): Promise<LLMSubscriptionStatus> {
    const response = await requestSubscriptionEndpoint(
      OPENAI_SUBSCRIPTION_STATUS_PATH,
    );
    return normalizeStatus(response as RawSubscriptionStatus);
  }

  static async startOpenAIDeviceLogin(): Promise<LLMSubscriptionDeviceChallenge> {
    const response = await requestSubscriptionEndpoint(
      OPENAI_SUBSCRIPTION_DEVICE_START_PATH,
      { method: "POST" },
    );
    return normalizeDeviceChallenge(response as RawDeviceStart);
  }

  static async pollOpenAIDeviceLogin(
    deviceCode: string,
  ): Promise<LLMSubscriptionStatus> {
    const response = await requestSubscriptionEndpoint(
      OPENAI_SUBSCRIPTION_DEVICE_POLL_PATH,
      {
        method: "POST",
        jsonBody: { device_code: deviceCode },
      },
    );
    return normalizeStatus(response as RawSubscriptionStatus);
  }

  static async logoutOpenAI(): Promise<LLMSubscriptionStatus> {
    const response = await requestSubscriptionEndpoint(
      OPENAI_SUBSCRIPTION_LOGOUT_PATH,
      { method: "POST" },
    );
    return normalizeStatus(response as RawSubscriptionStatus);
  }
}

export default LLMSubscriptionService;
