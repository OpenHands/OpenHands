import { describe, expect, it } from "vitest";
import type { PullRequestStatus } from "#/api/git-service/pull-request-status.api";
import {
  ATTENTION_ORDER,
  pullRequestAttention,
} from "#/utils/pull-request-attention";

const status = (extra: Partial<PullRequestStatus>): PullRequestStatus => ({
  repository: "acme/app",
  number: 1,
  title: "Fix",
  url: "https://github.com/acme/app/pull/1",
  state: "open",
  checks: "success",
  ...extra,
});

describe("pullRequestAttention", () => {
  it.each([
    [status({}), "ready"],
    [status({ checks: "none" }), "ready"],
    [status({ checks: "failure" }), "failing"],
    [status({ checks: "failure", draft: true }), "failing"],
    [status({ checks: "pending" }), "checking"],
    [status({ draft: true }), "draft"],
    [status({ state: "merged" }), "merged"],
    [status({ state: "closed", checks: "failure" }), "closed"],
    [undefined, "unknown"],
  ] as const)("%o is %s", (input, expected) => {
    expect(pullRequestAttention(input)).toBe(expected);
  });

  it("puts what needs the user first and what is done last", () => {
    expect(ATTENTION_ORDER.slice(0, 2)).toEqual(["ready", "failing"]);
    expect(ATTENTION_ORDER.slice(-2)).toEqual(["merged", "closed"]);
  });
});
