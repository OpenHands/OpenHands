import { useQuery } from "@tanstack/react-query";

import McpTestService from "#/api/mcp-test-service/mcp-test-service.api";
import { McpServerTestRunStatus } from "#/types/mcp-test";

const TERMINAL_STATUSES: McpServerTestRunStatus[] = [
  "succeeded",
  "failed",
  "cancelled",
];

export function useMcpTestRun(testId: string | null, enabled = true) {
  return useQuery({
    queryKey: ["mcp-test-run", testId],
    enabled: enabled && !!testId,
    queryFn: () => McpTestService.getTestRun(testId!),
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      if (status && TERMINAL_STATUSES.includes(status)) {
        return false;
      }
      return 2000;
    },
  });
}
