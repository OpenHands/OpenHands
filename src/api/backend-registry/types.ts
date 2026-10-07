export type BackendKind = "local" | "cloud";
export type BackendAuthMode = "api-key" | "cookie";

export interface Backend {
  id: string;
  name: string;
  host: string;
  apiKey: string;
  kind: BackendKind;
  authMode?: BackendAuthMode;
  /** Changes whenever connection credentials change, invalidating keyed data. */
  connectionRevision?: number;
  /**
   * Set when `host` reaches a DigitalOcean MARS session — its public ingress
   * URL, or the loopback end of a port-forward tunnel. The far end is an
   * ordinary agent-server, so the backend stays `kind: "local"`; this only
   * lets the Managed Agents screen and connection lifecycle recognise it.
   */
  marsSessionId?: string;
  /** The MARS agent config the session was launched from, when known. */
  marsConfigId?: string;
}

export interface BackendSelection {
  backendId: string;
  orgId?: string | null;
}

export interface ResolvedActiveBackend {
  backend: Backend;
  orgId: string | null;
}
