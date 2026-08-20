import type { MCPTestResponse } from "@openhands/typescript-client";
import { http, HttpResponse } from "msw";

/**
 * MSW handlers for the MCP API.
 *
 * Currently only the pre-flight connectivity check (`POST /api/mcp/test`)
 * needs a mock — the install/save flow in `InstallServerModal` and
 * `CustomServerEditor` calls it before persisting the new server via the
 * existing settings PATCH. In mock mode there is no real MCP server to
 * connect to, so we return a deterministic success response so the install
 * flow can complete in `npm run dev:mock`.
 */
const MOCK_MCP_TEST_SUCCESS: MCPTestResponse = {
  ok: true,
  tools: [],
};

export const MCP_HANDLERS = [
  http.post("*/api/mcp/test", async () =>
    HttpResponse.json(MOCK_MCP_TEST_SUCCESS),
  ),
];
