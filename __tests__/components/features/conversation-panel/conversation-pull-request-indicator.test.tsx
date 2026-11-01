import React from "react";
import { render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  GitProviderPullRequestStatusService,
  type ConversationPullRequest,
  type ConversationPullRequestsResult,
} from "#/api/git-provider-pull-request-status-service";
import { ConversationPullRequestIndicator } from "#/components/features/conversation-panel/conversation-card/conversation-pull-request-indicator";

// Echo interpolation values so tests can see what each label describes.
vi.mock("react-i18next", async () => {
  const actual = await vi.importActual("react-i18next");
  return {
    ...actual,
    useTranslation: () => ({
      t: (key: string, values?: Record<string, unknown>) =>
        values ? `${key}(${Object.values(values).join("|")})` : key,
    }),
  };
});

const getConversationPullRequestsSpy = vi.spyOn(
  GitProviderPullRequestStatusService,
  "getConversationPullRequests",
);

function pullRequest(
  overrides: Partial<ConversationPullRequest>,
): ConversationPullRequest {
  return {
    repository: "acme/widgets",
    number: 7,
    title: "Fix login",
    url: "https://github.com/acme/widgets/pull/7",
    lifecycle: "open",
    checks: "passing",
    review: "none",
    mergeable: "clean",
    updatedAt: null,
    unavailableReason: null,
    ...overrides,
  };
}

function ok(pullRequests: ConversationPullRequest[]) {
  return {
    status: "ok",
    source: "tracked",
    omittedCount: 0,
    pullRequests,
  } satisfies ConversationPullRequestsResult;
}

function renderIndicator(prNumbers: number[] | null) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={client}>
      <ConversationPullRequestIndicator
        repository="acme/widgets"
        provider="github"
        prNumbers={prNumbers}
      />
    </QueryClientProvider>,
  );
}

beforeEach(() => {
  getConversationPullRequestsSpy.mockReset();
});

describe("ConversationPullRequestIndicator", () => {
  it("renders nothing and asks the provider nothing when the conversation tracks no PR", async () => {
    // Act
    const { container } = renderIndicator([]);
    // Let any query that wrongly started reach its fetch.
    await new Promise((resolve) => {
      setTimeout(resolve, 0);
    });

    // Assert
    expect(container).toBeEmptyDOMElement();
    expect(getConversationPullRequestsSpy).not.toHaveBeenCalled();
  });

  it("shows the PR's lifecycle and marks failing checks beside it", async () => {
    // Arrange
    getConversationPullRequestsSpy.mockResolvedValue(
      ok([pullRequest({ checks: "failing" })]),
    );

    // Act
    renderIndicator([7]);

    // Assert
    const chip = await screen.findByTestId("conversation-card-pull-requests");
    expect(chip).toHaveTextContent("CONVERSATION$PR_LIFECYCLE_OPEN");
    expect(chip).toHaveAttribute(
      "title",
      "CONVERSATION$PR_INDICATOR_DESCRIPTION(7|acme/widgets|CONVERSATION$PR_LIFECYCLE_OPEN|CONVERSATION$PR_CHECKS_FAILING)",
    );
    expect(
      screen.getByTestId("conversation-card-pull-requests-checks-failing"),
    ).toBeInTheDocument();
  });

  it("summarizes several PRs without hiding a failure or which repository each is in", async () => {
    // Arrange
    getConversationPullRequestsSpy.mockResolvedValue(
      ok([
        pullRequest({ lifecycle: "merged", checks: "passing" }),
        pullRequest({ number: 8, checks: "failing" }),
      ]),
    );

    // Act
    renderIndicator([7, 8]);

    // Assert
    const chip = await screen.findByTestId("conversation-card-pull-requests");
    expect(chip).toHaveTextContent("CONVERSATION$PR_INDICATOR_COUNT(2)");
    const description = chip.getAttribute("title") ?? "";
    expect(description).toContain(
      "(7|acme/widgets|CONVERSATION$PR_LIFECYCLE_MERGED|CONVERSATION$PR_CHECKS_PASSING)",
    );
    expect(description).toContain(
      "(8|acme/widgets|CONVERSATION$PR_LIFECYCLE_OPEN|CONVERSATION$PR_CHECKS_FAILING)",
    );
    expect(
      screen.getByTestId("conversation-card-pull-requests-checks-failing"),
    ).toBeInTheDocument();
  });

  it.each([
    [
      { status: "unavailable", reason: "rate_limited" },
      "CONVERSATION$PR_UNAVAILABLE_RATE_LIMITED",
    ],
    [{ status: "unsupported_provider" }, "CONVERSATION$PR_STATUS_UNSUPPORTED"],
    [
      ok([
        pullRequest({
          lifecycle: "unknown",
          checks: "unknown",
          unavailableReason: "not_found",
        }),
      ]),
      "CONVERSATION$PR_UNAVAILABLE_NOT_FOUND",
    ],
  ] as [ConversationPullRequestsResult, string][])(
    "says the status is unknown rather than hiding the badge (%o)",
    async (result, reason) => {
      // Arrange
      getConversationPullRequestsSpy.mockResolvedValue(result);

      // Act
      renderIndicator([7]);

      // Assert
      const chip = await screen.findByTestId("conversation-card-pull-requests");
      expect(chip).toHaveTextContent("CONVERSATION$PR_INDICATOR_UNKNOWN");
      expect(chip.getAttribute("title")).toContain(reason);
    },
  );
});
