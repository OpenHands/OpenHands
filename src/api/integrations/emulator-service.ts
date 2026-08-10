import axios, { isAxiosError } from "axios";
import { NoBackendAvailableError } from "#/api/agent-server-client-options";
import { getEffectiveLocalBackend } from "#/api/backend-registry/active-store";

export const EMULATOR_PROXY_BASE = "/api/emulator";

export type EmulatorStatus = {
  ready: boolean;
  starting: boolean;
  unavailable: boolean;
  url: string;
  detail?: string;
};

export class EmulatorRequestError extends Error {
  readonly status: number;
  readonly unavailable: boolean;

  constructor(
    message: string,
    options: { status: number; unavailable?: boolean },
  ) {
    super(message);
    this.name = "EmulatorRequestError";
    this.status = options.status;
    this.unavailable = Boolean(options.unavailable);
  }
}

function unavailableStatus(detail?: string): EmulatorStatus {
  return {
    ready: false,
    starting: false,
    unavailable: true,
    url: `${EMULATOR_PROXY_BASE}/`,
    detail,
  };
}

async function emulatorRequest<T>(
  method: "GET" | "POST",
  path: string,
): Promise<T> {
  const backend = getEffectiveLocalBackend();
  if (!backend) {
    throw new NoBackendAvailableError();
  }

  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  const url = `${backend.host.replace(/\/+$/, "")}${EMULATOR_PROXY_BASE}${normalizedPath}`;
  const apiKey = backend.apiKey?.trim();

  try {
    const response = await axios.request<T>({
      method,
      url,
      headers: {
        Accept: "application/json",
        ...(apiKey ? { "X-Session-API-Key": apiKey } : {}),
      },
      withCredentials: true,
      validateStatus: () => true,
    });

    if (response.status === 404) {
      throw new EmulatorRequestError(
        "Emulator proxy is not available on this backend.",
        { status: 404, unavailable: true },
      );
    }

    if (response.status === 401) {
      throw new EmulatorRequestError("Missing or invalid session API key", {
        status: 401,
      });
    }

    const data = response.data as EmulatorStatus & { detail?: string };
    if (response.status >= 400) {
      throw new EmulatorRequestError(
        typeof data?.detail === "string" && data.detail.trim()
          ? data.detail
          : `Emulator request failed (${response.status})`,
        {
          status: response.status,
          unavailable:
            Boolean(data?.unavailable) ||
            response.status === 503 ||
            response.status === 501,
        },
      );
    }

    return response.data;
  } catch (err) {
    if (err instanceof EmulatorRequestError) {
      throw err;
    }
    if (isAxiosError(err)) {
      const status = err.response?.status ?? 0;
      throw new EmulatorRequestError(err.message || "Emulator request failed", {
        status,
        unavailable: status === 404 || status === 503,
      });
    }
    throw err;
  }
}

export class EmulatorService {
  static async getStatus(): Promise<EmulatorStatus> {
    try {
      return await emulatorRequest<EmulatorStatus>("GET", "/status");
    } catch (err) {
      if (err instanceof EmulatorRequestError && err.unavailable) {
        return unavailableStatus(err.message);
      }
      throw err;
    }
  }

  static start(): Promise<EmulatorStatus> {
    return emulatorRequest<EmulatorStatus>("POST", "/start");
  }

  /** Same-origin iframe path after /start sets the auth cookie. */
  static iframePath(): string {
    return `${EMULATOR_PROXY_BASE}/?autoconnect=1&reconnect=1&path=api/emulator/websockify`;
  }
}
