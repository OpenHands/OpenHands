import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter, Route, Routes } from "react-router";

import AutomationService from "#/api/automation-service/automation-service.api";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import AutomationWebhooksScreen from "#/routes/automation-webhooks";
import type { Backend } from "#/api/backend-registry/types";
import type { CustomWebhook } from "#/types/webhook";

vi.mock("#/api/automation-service/automation-service.api", () => ({
  default: {
    listWebhooks: vi.fn(),
    createWebhook: vi.fn(),
    updateWebhook: vi.fn(),
    deleteWebhook: vi.fn(),
    rotateWebhookSecret: vi.fn(),
  },
}));

const localBackend: Backend = {
  id: "local-1",
  name: "Local 1",
  host: "http://localhost:8000",
  apiKey: "session-key",
  kind: "local",
};

const existingWebhook: CustomWebhook = {
  id: "wh-1",
  org_id: "org-1",
  name: "Stripe events",
  source: "stripe",
  webhook_url: "/api/automation/v1/events/org-1/stripe",
  event_key_expr: "type",
  signature_header: "Stripe-Signature",
  enabled: true,
  created_at: "2026-01-01T00:00:00Z",
  updated_at: "2026-01-01T00:00:00Z",
};

function renderScreen() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <ActiveBackendProvider>
        <MemoryRouter initialEntries={["/automations/webhooks"]}>
          <Routes>
            <Route
              path="/automations/webhooks"
              element={<AutomationWebhooksScreen />}
            />
          </Routes>
        </MemoryRouter>
      </ActiveBackendProvider>
    </QueryClientProvider>,
  );
}

beforeEach(() => {
  __resetActiveStoreForTests();
  setRegisteredBackends([localBackend]);
  setActiveSelection({ backendId: localBackend.id });
  vi.mocked(AutomationService.listWebhooks).mockReset();
  vi.mocked(AutomationService.createWebhook).mockReset();
  vi.mocked(AutomationService.deleteWebhook).mockReset();
  vi.mocked(AutomationService.rotateWebhookSecret).mockReset();
});

afterEach(() => {
  __resetActiveStoreForTests();
});

describe("AutomationWebhooksScreen", () => {
  it("shows the empty state when there are no webhooks", async () => {
    vi.mocked(AutomationService.listWebhooks).mockResolvedValue({
      webhooks: [],
      total: 0,
    });

    renderScreen();

    expect(await screen.findByTestId("webhooks-empty")).toBeInTheDocument();
  });

  it("lists existing webhook sources", async () => {
    vi.mocked(AutomationService.listWebhooks).mockResolvedValue({
      webhooks: [existingWebhook],
      total: 1,
    });

    renderScreen();

    expect(await screen.findByText("Stripe events")).toBeInTheDocument();
    expect(screen.getByText("stripe")).toBeInTheDocument();
  });

  it("creates a webhook and reveals a system-generated secret exactly once", async () => {
    vi.mocked(AutomationService.listWebhooks).mockResolvedValue({
      webhooks: [],
      total: 0,
    });
    vi.mocked(AutomationService.createWebhook).mockResolvedValue({
      ...existingWebhook,
      id: "wh-2",
      name: "New source",
      source: "linear",
      webhook_secret: "whsec_generated123",
    });

    renderScreen();
    await screen.findByTestId("webhooks-empty");

    fireEvent.click(screen.getByTestId("add-webhook-button"));

    fireEvent.change(screen.getByTestId("webhook-name-input"), {
      target: { value: "New source" },
    });
    fireEvent.change(screen.getByTestId("webhook-source-input"), {
      target: { value: "linear" },
    });

    fireEvent.click(screen.getByTestId("webhook-submit-button"));

    await waitFor(() => {
      expect(AutomationService.createWebhook).toHaveBeenCalledWith(
        expect.objectContaining({ name: "New source", source: "linear" }),
      );
    });

    expect(
      await screen.findByTestId("webhook-secret-reveal"),
    ).toBeInTheDocument();
    expect(screen.getByTestId("webhook-secret-value")).toHaveTextContent(
      "whsec_generated123",
    );
  });

  it("does not reveal a secret when the caller supplied their own", async () => {
    vi.mocked(AutomationService.listWebhooks).mockResolvedValue({
      webhooks: [],
      total: 0,
    });
    vi.mocked(AutomationService.createWebhook).mockResolvedValue({
      ...existingWebhook,
      id: "wh-3",
      name: "Own secret source",
      source: "custom-src",
      webhook_secret: null,
    });

    renderScreen();
    await screen.findByTestId("webhooks-empty");
    fireEvent.click(screen.getByTestId("add-webhook-button"));
    fireEvent.change(screen.getByTestId("webhook-name-input"), {
      target: { value: "Own secret source" },
    });
    fireEvent.change(screen.getByTestId("webhook-source-input"), {
      target: { value: "custom-src" },
    });
    fireEvent.click(screen.getByTestId("webhook-submit-button"));

    await waitFor(() => {
      expect(AutomationService.createWebhook).toHaveBeenCalled();
    });

    expect(
      screen.queryByTestId("webhook-secret-reveal"),
    ).not.toBeInTheDocument();
  });

  it("rotates a webhook's secret and reveals the new value once", async () => {
    vi.mocked(AutomationService.listWebhooks).mockResolvedValue({
      webhooks: [existingWebhook],
      total: 1,
    });
    vi.mocked(AutomationService.rotateWebhookSecret).mockResolvedValue({
      webhook_secret: "whsec_rotated456",
    });

    renderScreen();
    await screen.findByText("Stripe events");

    fireEvent.click(screen.getByTestId("rotate-webhook-secret-button"));

    await waitFor(() => {
      expect(AutomationService.rotateWebhookSecret).toHaveBeenCalledWith(
        "wh-1",
      );
    });
    expect(
      await screen.findByTestId("webhook-secret-reveal"),
    ).toBeInTheDocument();
    expect(screen.getByTestId("webhook-secret-value")).toHaveTextContent(
      "whsec_rotated456",
    );
  });

  it("deletes a webhook after confirmation", async () => {
    vi.mocked(AutomationService.listWebhooks).mockResolvedValue({
      webhooks: [existingWebhook],
      total: 1,
    });
    vi.mocked(AutomationService.deleteWebhook).mockResolvedValue(undefined);

    renderScreen();
    await screen.findByText("Stripe events");

    fireEvent.click(screen.getByTestId("delete-webhook-button"));

    fireEvent.click(await screen.findByTestId("confirm-button"));

    await waitFor(() => {
      expect(AutomationService.deleteWebhook).toHaveBeenCalledWith("wh-1");
    });
  });
});
