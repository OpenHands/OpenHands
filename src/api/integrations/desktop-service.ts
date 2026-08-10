import axios, { isAxiosError } from "axios";
import { NoBackendAvailableError } from "#/api/agent-server-client-options";
import { getEffectiveLocalBackend } from "#/api/backend-registry/active-store";

export const DESKTOP_PROXY_BASE = "/api/desktop";

export type DesktopStatus = {
  ready: boolean;
  starting: boolean;
  unavailable: boolean;
  url: string;
  detail?: string;
};

export class DesktopRequestError extends Error {
  readonly status: number;
  readonly unavailable: boolean;

  constructor(
    message: string,
    options: { status: number; unavailable?: boolean },
  ) {
    super(message);
    this.name = "DesktopRequestError";
    this.status = options.status;
    this.unavailable = Boolean(options.unavailable);
  }
}

function unavailableStatus(detail?: string): DesktopStatus {
  return {
    ready: false,
    starting: false,
    unavailable: true,
    url: `${DESKTOP_PROXY_BASE}/index.html`,
    detail,
  };
}

async function desktopRequest<T>(
  method: "GET" | "POST",
  path: string,
): Promise<T> {
  const backend = getEffectiveLocalBackend();
  if (!backend) {
    throw new NoBackendAvailableError();
  }

  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  const url = `${backend.host.replace(/\/+$/, "")}${DESKTOP_PROXY_BASE}${normalizedPath}`;
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
      // Published images without the Desktop proxy fall through to the
      // agent-server, which returns 404 HTML/JSON.
      throw new DesktopRequestError(
        "Desktop proxy is not available on this backend. Rebuild the agent-canvas Docker image.",
        { status: 404, unavailable: true },
      );
    }

    if (response.status === 401) {
      throw new DesktopRequestError("Missing or invalid session API key", {
        status: 401,
      });
    }

    const data = response.data as DesktopStatus & { detail?: string };
    if (response.status >= 400) {
      throw new DesktopRequestError(
        typeof data?.detail === "string" && data.detail.trim()
          ? data.detail
          : `Desktop request failed (${response.status})`,
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
    if (err instanceof DesktopRequestError) {
      throw err;
    }
    if (isAxiosError(err)) {
      const status = err.response?.status ?? 0;
      throw new DesktopRequestError(
        err.message || "Desktop request failed",
        {
          status,
          unavailable: status === 404 || status === 503,
        },
      );
    }
    throw err;
  }
}

export class DesktopService {
  static async getStatus(): Promise<DesktopStatus> {
    try {
      return await desktopRequest<DesktopStatus>("GET", "/status");
    } catch (err) {
      if (err instanceof DesktopRequestError && err.unavailable) {
        return unavailableStatus(err.message);
      }
      throw err;
    }
  }

  static start(): Promise<DesktopStatus> {
    return desktopRequest<DesktopStatus>("POST", "/start");
  }

  static stop(): Promise<{ ready: boolean }> {
    return desktopRequest<{ ready: boolean }>("POST", "/stop");
  }

  /** Same-origin iframe path after /start sets the auth cookie. */
  static iframePath(): string {
    return `${DESKTOP_PROXY_BASE}/index.html?autoconnect=1&reconnect=1&resize=remote&path=api/desktop/websockify`;
  }
}
