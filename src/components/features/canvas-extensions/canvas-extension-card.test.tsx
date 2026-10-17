import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import CanvasExtensionsService from "#/api/canvas-extensions-service";
import { I18nKey } from "#/i18n/declaration";
import type { InstalledCanvasExtensionInfo } from "#/types/canvas-extension";
import { CanvasExtensionCard } from "./canvas-extension-card";

const extension: InstalledCanvasExtensionInfo = {
  name: "demo",
  version: "1",
  enabled: true,
  source: "/demo",
  installed_at: "now",
  install_path: "/demo",
  manifest: {
    schema_version: 1,
    name: "demo",
    version: "1",
    entrypoint: "index.js",
    backend: { secret_refs: ["DEMO_API_KEY"] },
  },
};
const stopped = {
  name: "demo",
  state: "stopped",
  revision: "rev",
  prepared_revision: "rev",
  secret_refs: ["DEMO_API_KEY"],
  secret_grant_revision: null,
};
function show() {
  return render(
    <QueryClientProvider client={new QueryClient()}>
      <CanvasExtensionCard
        extension={extension}
        isBusy={false}
        onToggle={vi.fn()}
        onRefresh={vi.fn()}
        onUninstall={vi.fn()}
      />
    </QueryClientProvider>,
  );
}
afterEach(() => vi.restoreAllMocks());

it("only offers revocation for approved secrets, never manual lifecycle buttons", async () => {
  vi.spyOn(CanvasExtensionsService, "backendStatus").mockResolvedValue({
    ...stopped,
    secret_grant_revision: "rev",
  });
  const action = vi
    .spyOn(CanvasExtensionsService, "backendAction")
    .mockResolvedValue(stopped);
  show();
  const revoke = await screen.findByRole("button", {
    name: I18nKey.SETTINGS$APPS_BACKEND_REVOKE,
  });
  expect(
    screen.queryByRole("button", { name: I18nKey.SETTINGS$APPS_BACKEND_START }),
  ).not.toBeInTheDocument();
  expect(
    screen.queryByRole("button", {
      name: I18nKey.SETTINGS$APPS_BACKEND_PREPARE,
    }),
  ).not.toBeInTheDocument();
  fireEvent.click(revoke);
  await act(async () => {});
  expect(action).toHaveBeenCalled();
  expect(
    screen.queryByRole("button", {
      name: I18nKey.SETTINGS$APPS_BACKEND_REVOKE,
    }),
  ).not.toBeInTheDocument();
});

it("does not apply a late status response after the controls unmount", async () => {
  let resolve!: (value: typeof stopped) => void;
  vi.spyOn(CanvasExtensionsService, "backendStatus").mockReturnValue(
    new Promise((done) => {
      resolve = done;
    }),
  );
  const { unmount } = show();
  unmount();
  await act(async () => resolve(stopped));
  expect(
    screen.queryByRole("button", { name: I18nKey.SETTINGS$APPS_BACKEND_GRANT }),
  ).not.toBeInTheDocument();
});
