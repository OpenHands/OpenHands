// @vitest-environment node
import { afterEach, describe, expect, it, vi } from "vitest";

import { renderReport } from "../../tests/e2e/mock-llm/scripts/render-mock-llm-report.mjs";
import {
  buildCommentBody,
  findMatchingJobComments,
  upsertJobComment,
} from "../../tests/e2e/mock-llm/scripts/upsert-pr-comment.mjs";

const MOCK_MARKER = "<!-- agent-canvas-mock-llm-e2e-report -->";

describe("mock-LLM E2E reporting", () => {
  it("keeps the summary visible while hiding the results table in details", () => {
    const report = renderReport({
      tests: [
        {
          title: "conversations/mock-llm-conversation.spec.ts › runs command",
          file: "conversations/mock-llm-conversation.spec.ts",
          status: "passed",
          durationMs: 1250,
          retryCount: 0,
          error: "",
        },
      ],
      workflowUrl:
        "https://github.com/OpenHands/agent-canvas/actions/runs/28000401257",
      commit: "82c9e1d04d62961e14742e20a4237ecd6db20ff0",
      artifactUrl:
        "https://github.com/OpenHands/agent-canvas/actions/runs/28000401257/artifacts/7811037798",
      title: "Mock-LLM Docker E2E Test Results",
      newFiles: [],
      markerMeta: null,
    });

    const detailsIndex = report.indexOf("<details>");
    const tableIndex = report.indexOf("| Status | Test | Duration |");
    const visibleSummary = report.slice(0, detailsIndex);

    expect(visibleSummary).toContain("## ✅ Mock-LLM Docker E2E Test Results");
    expect(visibleSummary).toContain("**1/1 passed**");
    expect(visibleSummary).toContain("Commit: `82c9e1d0`");
    expect(visibleSummary).toContain("[Workflow run]");
    expect(visibleSummary).toContain("[Test artifacts]");
    expect(visibleSummary).not.toContain("| Status | Test | Duration |");
    expect(visibleSummary).not.toContain("Posted by the Mock-LLM E2E workflow");
    expect(report).toContain("<summary>Details</summary>");
    expect(detailsIndex).toBeGreaterThan(-1);
    expect(tableIndex).toBeGreaterThan(detailsIndex);
  });

  it("marks new comments and finds older comments for the same job", () => {
    const body = "## ✅ Mock-LLM E2E Tests\n\n**60/60 passed**";

    expect(buildCommentBody(body, MOCK_MARKER)).toBe(`${MOCK_MARKER}\n${body}`);
    expect(buildCommentBody(`${MOCK_MARKER}\n${body}`, MOCK_MARKER)).toBe(
      `${MOCK_MARKER}\n${body}`,
    );

    const matching = findMatchingJobComments(
      [
        {
          id: 1,
          body: `${MOCK_MARKER}\n## ✅ Mock-LLM E2E Tests`,
          user: { login: "someone", type: "User" },
        },
        {
          id: 2,
          body: "## ✅ Mock-LLM E2E Tests\n\nolder unmarked body",
          user: { login: "github-actions[bot]", type: "Bot" },
        },
        {
          id: 3,
          body: "## ✅ Mock-LLM E2E Tests\n\nhuman mention",
          user: { login: "maintainer", type: "User" },
        },
        {
          id: 4,
          body: "## ✅ Mock-LLM Docker E2E Test Results\n\nother job",
          user: { login: "github-actions[bot]", type: "Bot" },
        },
      ],
      { marker: MOCK_MARKER, legacyTitle: "Mock-LLM E2E Tests" },
    );

    expect(matching.map((comment: { id: number }) => comment.id)).toEqual([
      1, 2,
    ]);
  });

  it("edits the existing report in place instead of reposting it", async () => {
    const requests: { method: string; path: string }[] = [];
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockImplementation(async (url, init) => {
        const method = init?.method ?? "GET";
        const path =
          new URL(String(url)).pathname + new URL(String(url)).search;
        requests.push({ method, path });

        if (method === "GET") {
          return new Response(
            JSON.stringify(
              path.includes("page=1")
                ? [
                    { id: 11, body: "unrelated", user: { type: "User" } },
                    {
                      id: 22,
                      body: `${MOCK_MARKER}\n## ✅ Mock-LLM E2E Tests`,
                      user: { login: "github-actions[bot]", type: "Bot" },
                    },
                  ]
                : [],
            ),
            { status: 200 },
          );
        }
        return new Response(JSON.stringify({ id: 22 }), { status: 200 });
      });

    const result = await upsertJobComment({
      repo: "OpenHands/OpenHands",
      issueNumber: "16521",
      token: "t",
      body: "## ✅ Mock-LLM E2E Tests\n\n**62/62 passed**",
      marker: MOCK_MARKER,
      legacyTitle: "Mock-LLM E2E Tests",
    });

    expect(result).toMatchObject({ updated: true, deleted: 0 });
    expect(requests.filter((r) => r.method === "PATCH")).toEqual([
      {
        method: "PATCH",
        path: "/repos/OpenHands/OpenHands/issues/comments/22",
      },
    ]);
    expect(requests.some((r) => r.method === "POST")).toBe(false);
    expect(requests.some((r) => r.method === "DELETE")).toBe(false);
    fetchMock.mockRestore();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });
});
