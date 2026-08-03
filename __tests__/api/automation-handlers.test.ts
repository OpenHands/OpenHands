import { http, HttpResponse } from "msw";
import { setupServer } from "msw/node";
import { describe, it, expect, beforeAll, afterAll, afterEach } from "vitest";
import capabilitiesFixture from "@openhands/extensions/testing/automations/capabilities.json";
import prReviewerFixture from "@openhands/extensions/testing/automations/github-pr-reviewer.json";
import repoMonitorFixture from "@openhands/extensions/testing/automations/github-repo-monitor.json";
import {
  AUTOMATION_HANDLERS,
  resetAutomationMockData,
} from "#/mocks/automation-handlers";
import { MOCK_AUTOMATIONS_RESPONSE } from "#/mocks/automations.mock";

interface PreflightExchange {
  request: { method: string; path: string; body: unknown };
  response: { status: number; body: unknown };
}

interface FixtureBundle {
  automationId: string;
  scenarios: { id: string; preflight?: PreflightExchange }[];
}

/**
 * Every preflight exchange the published contract fixtures record. The mock
 * backend must reproduce each one, so the setup flow exercised against it is
 * exercised against the reference contract.
 */
const PREFLIGHT_EXCHANGES = (
  [prReviewerFixture, repoMonitorFixture] as FixtureBundle[]
).flatMap((bundle) =>
  bundle.scenarios.flatMap((scenario) =>
    scenario.preflight
      ? [
          {
            name: `${bundle.automationId}/${scenario.id}`,
            exchange: scenario.preflight,
          },
        ]
      : [],
  ),
);

const server = setupServer(...AUTOMATION_HANDLERS);

describe("Automation MSW Handlers", () => {
  beforeAll(() => server.listen({ onUnhandledRequest: "error" }));
  afterAll(() => server.close());
  afterEach(() => {
    server.resetHandlers();
    resetAutomationMockData();
  });

  describe("GET /api/automation/v1", () => {
    it("returns paginated automations list", async () => {
      const res = await fetch("/api/automation/v1?limit=10&offset=0");
      const data = await res.json();

      expect(res.status).toBe(200);
      expect(data.automations).toHaveLength(
        MOCK_AUTOMATIONS_RESPONSE.automations.length,
      );
      expect(data.total).toBe(MOCK_AUTOMATIONS_RESPONSE.total);
    });

    it("respects pagination parameters", async () => {
      const res = await fetch("/api/automation/v1?limit=2&offset=2");
      const data = await res.json();

      expect(res.status).toBe(200);
      expect(data.automations).toHaveLength(2);
      expect(data.automations[0].name).toBe("Docs Sync on Push");
    });
  });

  describe("GET /api/automation/v1/capabilities", () => {
    it("answers with the contract fixtures' supported deployment", async () => {
      // Act
      const res = await fetch("/api/automation/v1/capabilities");
      const data = await res.json();

      // Assert
      expect({ status: res.status, body: data }).toEqual({
        status: 200,
        body: capabilitiesFixture.responses.supported.body,
      });
    });
  });

  describe("POST /api/automation/v1/validate", () => {
    it.each(PREFLIGHT_EXCHANGES)(
      "reproduces the $name exchange from the contract fixtures",
      async ({ exchange }) => {
        // Act
        const res = await fetch(`/api/automation${exchange.request.path}`, {
          method: exchange.request.method,
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(exchange.request.body),
        });
        const data = await res.json();

        // Assert
        expect({ status: res.status, body: data }).toEqual({
          status: exchange.response.status,
          body: exchange.response.body,
        });
      },
    );

    it("rejects an every-minute schedule below the deployment minimum", async () => {
      // Arrange
      const draft = {
        trigger: { type: "cron", schedule: "* * * * *", timezone: "UTC" },
      };

      // Act
      const res = await fetch("/api/automation/v1/validate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          automationId: "github-pr-reviewer",
          endpoint: "/v1/preset/prompt",
          draft,
        }),
      });
      const data = await res.json();

      // Assert
      expect(data).toEqual({
        valid: false,
        errors: [
          {
            field: "trigger.schedule",
            code: "interval_too_short",
            message: "Minimum interval for this deployment is 5 minutes.",
          },
        ],
      });
    });

    it("passes a schedule shape its cron reader does not model", async () => {
      // Arrange — "0 0 31 2 *" can never fire, but only the real service can
      // say so; the fixtures record that rejection at the create stage.
      const draft = {
        trigger: { type: "cron", schedule: "0 0 31 2 *", timezone: "UTC" },
      };

      // Act
      const res = await fetch("/api/automation/v1/validate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          automationId: "github-pr-reviewer",
          endpoint: "/v1/preset/prompt",
          draft,
        }),
      });
      const data = await res.json();

      // Assert
      expect(data).toEqual({ valid: true, errors: [] });
    });
  });

  describe("GET /api/automation/v1/:id", () => {
    it("returns a single automation by id", async () => {
      const id = MOCK_AUTOMATIONS_RESPONSE.automations[0].id;
      const res = await fetch(`/api/automation/v1/${id}`);
      const data = await res.json();

      expect(res.status).toBe(200);
      expect(data.id).toBe(id);
      expect(data.name).toBe("PR Triage Digest");
    });

    it("returns 404 for non-existent automation", async () => {
      const res = await fetch("/api/automation/v1/non-existent-id");

      expect(res.status).toBe(404);
    });
  });

  describe("PATCH /api/automation/v1/:id", () => {
    it("updates an automation", async () => {
      const id = MOCK_AUTOMATIONS_RESPONSE.automations[0].id;
      const res = await fetch(`/api/automation/v1/${id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enabled: false }),
      });
      const data = await res.json();

      expect(res.status).toBe(200);
      expect(data.enabled).toBe(false);
    });

    it("returns 404 for non-existent automation", async () => {
      const res = await fetch("/api/automation/v1/non-existent-id", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enabled: false }),
      });

      expect(res.status).toBe(404);
    });
  });

  describe("DELETE /api/automation/v1/:id", () => {
    it("deletes an automation", async () => {
      const id = MOCK_AUTOMATIONS_RESPONSE.automations[0].id;
      const res = await fetch(`/api/automation/v1/${id}`, {
        method: "DELETE",
      });

      expect(res.status).toBe(204);

      // Verify it's actually deleted
      const getRes = await fetch(`/api/automation/v1/${id}`);
      expect(getRes.status).toBe(404);
    });

    it("returns 404 for non-existent automation", async () => {
      const res = await fetch("/api/automation/v1/non-existent-id", {
        method: "DELETE",
      });

      expect(res.status).toBe(404);
    });
  });

  describe("POST /api/automation/v1/:id/dispatch", () => {
    it("creates a pending run for an existing automation", async () => {
      const id = MOCK_AUTOMATIONS_RESPONSE.automations[0].id;
      const res = await fetch(`/api/automation/v1/${id}/dispatch`, {
        method: "POST",
      });
      const data = await res.json();

      expect(res.status).toBe(201);
      expect(data.status).toBe("PENDING");
    });

    it("returns 404 for non-existent automation", async () => {
      const res = await fetch("/api/automation/v1/non-existent-id/dispatch", {
        method: "POST",
      });

      expect(res.status).toBe(404);
    });
  });

  describe("GET /api/automation/v1/:id/runs", () => {
    it("returns automation runs", async () => {
      const id = MOCK_AUTOMATIONS_RESPONSE.automations[0].id;
      const res = await fetch(`/api/automation/v1/${id}/runs?limit=5&offset=0`);
      const data = await res.json();

      expect(res.status).toBe(200);
      expect(data.runs).toBeInstanceOf(Array);
      expect(data.total).toBeGreaterThanOrEqual(0);
    });

    it("returns 404 for non-existent automation", async () => {
      const res = await fetch("/api/automation/v1/non-existent-id/runs");

      expect(res.status).toBe(404);
    });
  });
});
