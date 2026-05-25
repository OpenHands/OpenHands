import { openHands } from "../open-hands-axios";

/**
 * One row returned by ``GET /api/v1/databricks/models``.
 *
 * Mirrors the ``ModelPickerEntry`` dataclass in the openhands-sdk
 * Databricks provider (discovery.py) — kept intentionally in sync so the
 * same merged curated + discovered list powers both the web picker and the
 * CLI picker.
 */
export interface DatabricksModelEntry {
  qualified_name: string;
  name: string;
  family: "openai" | "openai_responses" | "anthropic" | "gemini";
  source: "curated" | "discovered" | "curated+discovered";
  endpoint_type: string | null;
  ready: boolean;
  recommended: boolean;
}

export interface DatabricksModelsResponse {
  entries: DatabricksModelEntry[];
  /**
   * ``curated`` when the backend could not reach the workspace (no auth or
   * discovery failed — curated-only response). ``curated+discovered`` when
   * the live workspace endpoints were successfully merged in.
   */
  source: "curated" | "curated+discovered" | "unavailable";
  host: string | null;
  reason?: string;
}

/**
 * Databricks two-tier model picker API client.
 *
 * See ``openhands/app_server/auth/databricks_models_routes.py`` for the
 * contract. This hits the authenticated V1 endpoint — it relies on the
 * user's browser session (cookies) for auth; no extra headers needed.
 */
class DatabricksModelsService {
  /**
   * Fetch the merged curated + discovered model list.
   *
   * @param host  Optional explicit workspace URL (e.g. when the user is
   *              still editing their base URL and we want to preview what
   *              that host exposes). If omitted, the backend falls back to
   *              the user's stored ``llm_base_url`` and finally the
   *              ``DATABRICKS_HOST`` env.
   * @param includeDiscovered  Set to ``false`` to hard-skip the live
   *              discovery probe (curated-only).
   */
  static async list(
    host?: string,
    includeDiscovered: boolean = true,
  ): Promise<DatabricksModelsResponse> {
    const params: Record<string, string> = {};
    if (host) params.host = host;
    if (!includeDiscovered) params.include_discovered = "false";

    const { data } = await openHands.get<DatabricksModelsResponse>(
      "/api/v1/databricks/models",
      { params },
    );
    return data;
  }
}

export default DatabricksModelsService;
