import React from "react";
import { render, screen, waitFor } from "@testing-library/react";
import { QueryClientProvider } from "@tanstack/react-query";
import { HttpError } from "@openhands/typescript-client";
import { ServerClient } from "@openhands/typescript-client/clients";
import { afterEach, expect, it, vi } from "vitest";
import type { Backend } from "#/api/backend-registry/types";
import { BackendVersion } from "#/components/features/backends/backend-version";
import { createAgentServerQueryClient } from "#/query-client-config";
import * as ToastHandlers from "#/utils/custom-toast-handlers";

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, options: { version: string }) => options.version,
  }),
}));

afterEach(() => vi.restoreAllMocks());

it("keeps failed version probes quiet and shows the version after recovery", async () => {
  const backend: Backend = {
    id: "local-version-test",
    name: "Local",
    kind: "local",
    host: "http://127.0.0.1:18160",
    apiKey: "fixture-key",
  };
  const probe = vi
    .spyOn(ServerClient.prototype, "getServerInfo")
    .mockRejectedValue(
      new HttpError(503, "Service Unavailable", {
        detail: "Version probe unavailable",
      }),
    );
  const toast = vi.spyOn(ToastHandlers, "displayErrorToast");
  const client = createAgentServerQueryClient();
  render(
    <QueryClientProvider client={client}>
      <BackendVersion backend={backend} />
    </QueryClientProvider>,
  );
  await waitFor(() =>
    expect(client.getQueryCache().getAll()[0]?.state.status).toBe("error"),
  );
  expect(toast).not.toHaveBeenCalled();
  expect(
    screen.queryByTestId("manage-backends-version-Local"),
  ).not.toBeInTheDocument();
  probe.mockResolvedValue({ version: "1.54.0" } as Awaited<
    ReturnType<ServerClient["getServerInfo"]>
  >);
  await client.refetchQueries();
  expect(
    await screen.findByTestId("manage-backends-version-Local"),
  ).toHaveTextContent("1.54.0");
  client.clear();
});
