import { describe, expect, it, beforeEach } from "vitest";
import React from "react";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ModelSelector } from "#/components/shared/modals/settings/model-selector";
import { server } from "#/mocks/node";

describe("ModelSelector — OpenHands provider display", () => {
  let providersCount = 0;
  let verifiedCount = 0;
  let modelsCount = 0;

  beforeEach(() => {
    providersCount = 0;
    verifiedCount = 0;
    modelsCount = 0;
    // Use "*" prefix to match both relative paths and absolute URLs (e.g.,
    // http://127.0.0.1:8000/api/...) when VITE_BACKEND_BASE_URL is configured.
    server.use(
      http.get("*/api/llm/providers", () => {
        providersCount += 1;
        return HttpResponse.json({
          providers: ["openhands", "anthropic", "openai"],
        });
      }),
      http.get("*/api/llm/models/verified", () => {
        verifiedCount += 1;
        return HttpResponse.json({
          models: {
            openhands: [
              "claude-opus-4-7",
              "glm-5.2",
              "deepseek-v4-flash",
              "minimax-m2.7",
            ],
            anthropic: ["claude-opus-4-5-20251101"],
          },
        });
      }),
      http.get("*/api/llm/models", () => {
        modelsCount += 1;
        return HttpResponse.json({ models: [] });
      }),
    );
  });

  function renderWithQuery(ui: React.ReactElement) {
    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });
    return render(
      <QueryClientProvider client={queryClient}>{ui}</QueryClientProvider>,
    );
  }

  it("shows OpenHands immediately for current openhands/<m> settings", async () => {
    renderWithQuery(<ModelSelector currentModel="openhands/claude-opus-4-7" />);

    await waitFor(() => {
      expect(screen.getByLabelText("LLM$PROVIDER")).toHaveValue("OpenHands");
    });

    expect(providersCount).toBe(1);
    expect(verifiedCount).toBe(1);
    expect(modelsCount).toBe(1);
  });

  it("makes clear which OpenHands models are free", async () => {
    const user = userEvent.setup();
    renderWithQuery(<ModelSelector currentModel="openhands/glm-5.2" />);

    await waitFor(() => {
      expect(
        screen.getByTestId("openhands-free-models-note"),
      ).toHaveTextContent("openhands/glm-5.2");
    });
    expect(screen.getByTestId("openhands-free-models-note")).toHaveTextContent(
      "openhands/deepseek-v4-flash",
    );
    expect(screen.getByTestId("openhands-free-models-note")).toHaveTextContent(
      "openhands/minimax-m2.7",
    );
    expect(screen.getByTestId("selected-free-model-badge")).toHaveTextContent(
      "Free",
    );

    await user.click(screen.getByLabelText("LLM$MODEL"));

    expect(screen.getAllByText("Free")).toHaveLength(4);
    expect(screen.getByLabelText("LLM$MODEL")).toHaveValue("glm-5.2");

    await user.click(screen.getByText("minimax-m2.7"));

    expect(screen.getByLabelText("LLM$MODEL")).toHaveValue("minimax-m2.7");
    expect(screen.getByTestId("selected-free-model-badge")).toHaveTextContent(
      "Free",
    );
  });
});
