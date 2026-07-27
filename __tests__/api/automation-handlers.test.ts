import { http, HttpResponse } from "msw";
import { setupServer } from "msw/node";
import { describe, it, expect, beforeAll, afterAll, afterEach } from "vitest";
import {
  AUTOMATION_HANDLERS,
  resetAutomationMockData,
} from "#/mocks/automation-handlers";
import { MOCK_AUTOMATIONS_RESPONSE } from "#/mocks/automations.mock";

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
