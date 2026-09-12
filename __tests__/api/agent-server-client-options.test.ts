import {
  assertConversationRuntimeClientSupport,
  supportsConversationRuntimeRoutes,
} from "#/api/agent-server-client-options";
import { describe, expect, it } from "vitest";
import {
  getAgentServerClientOptions,
  getAgentServerHttpClientOptions,
} from "#/api/agent-server-client-options";

const overrides = {
  host: "http://127.0.0.1:9000",
  conversationUrl:
    "http://127.0.0.1:9000/api/conversations/11111111-1111-4111-8111-111111111111",
};

describe("Docker conversation client context", () => {
  it("retains conversation context in typed client options", () => {
    expect(getAgentServerClientOptions(overrides).conversationId).toBe(
      "11111111-1111-4111-8111-111111111111",
    );
  });

  it("retains the same context in HTTP client options", () => {
    expect(getAgentServerHttpClientOptions(overrides)).toHaveProperty(
      "conversationId",
      "11111111-1111-4111-8111-111111111111",
    );
  });
});

describe("explicit conversation context", () => {
  it("retains explicit ID without a URL and prefers it over URL context", () => {
    expect(
      getAgentServerHttpClientOptions({
        ...overrides,
        conversationId: "explicit-id",
      }).conversationId,
    ).toBe("explicit-id");
    expect(
      getAgentServerClientOptions({
        host: overrides.host,
        conversationId: "explicit-id",
      }).conversationId,
    ).toBe("explicit-id");
  });
  it("leaves host discovery unscoped", () => {
    expect(
      getAgentServerClientOptions({ host: overrides.host }),
    ).not.toHaveProperty("conversationId");
  });
  it("retains the full ID from a prefixed URL without query or fragment", () => {
    expect(
      getAgentServerClientOptions({
        ...overrides,
        conversationUrl:
          "http://localhost/runtime/123/api/conversations/conversation-abc?view=files#tab",
      }).conversationId,
    ).toBe("conversation-abc");
  });
});

describe("installed SDK support", () => {
  it("either supports scoped routes or gives an explicit upgrade error", () => {
    if (supportsConversationRuntimeRoutes()) {
      expect(() => assertConversationRuntimeClientSupport()).not.toThrow();
    } else {
      expect(() => assertConversationRuntimeClientSupport()).toThrow(
        "Upgrade Canvas",
      );
    }
  });
});
