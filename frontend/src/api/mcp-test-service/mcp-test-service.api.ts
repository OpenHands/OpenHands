import { openHands } from "../open-hands-axios";
import {
  McpServerHealthResponse,
  McpServerTestRun,
  McpServerTestRunPage,
  StartMcpServerTestResponse,
} from "#/types/mcp-test";

class McpTestService {
  static async startTest(
    serverId: string,
  ): Promise<StartMcpServerTestResponse> {
    const { data } = await openHands.post<StartMcpServerTestResponse>(
      `/api/v1/settings/mcp/servers/${encodeURIComponent(serverId)}/test`,
    );
    return data;
  }

  static async getTestRun(testId: string): Promise<McpServerTestRun> {
    const { data } = await openHands.get<McpServerTestRun>(
      `/api/v1/settings/mcp/test-runs/${testId}`,
    );
    return data;
  }

  static async getServerHealth(
    serverId: string,
  ): Promise<McpServerHealthResponse> {
    const { data } = await openHands.get<McpServerHealthResponse>(
      `/api/v1/settings/mcp/servers/${encodeURIComponent(serverId)}/health`,
    );
    return data;
  }

  static async listServerTestRuns(
    serverId: string,
    pageId?: string,
    limit = 20,
  ): Promise<McpServerTestRunPage> {
    const { data } = await openHands.get<McpServerTestRunPage>(
      `/api/v1/settings/mcp/servers/${encodeURIComponent(serverId)}/test-runs`,
      { params: { page_id: pageId, limit } },
    );
    return data;
  }
}

export default McpTestService;
