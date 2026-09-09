import { describe, expect, it } from "vitest";
import {
  CHATGPT_AUTO_PROFILE_PREFIX,
  canonicalSubscriptionModelKey,
  chatgptAutoProfileName,
  countOffersBySource,
  isChatgptAutoProfileName,
  mergeSubscriptionModelOffers,
  parseCursorAgentModels,
  parseOpenCodeModels,
  subscriptionModelToggleKey,
  subscriptionSourceForConnection,
  type SubscriptionModelOffer,
} from "#/utils/subscription-model-catalog";

describe("canonicalSubscriptionModelKey", () => {
  it("strips known provider prefixes so the same model can match across subscriptions", () => {
    expect(canonicalSubscriptionModelKey("opencode/claude-sonnet-5")).toBe(
      "claude-sonnet-5",
    );
    expect(canonicalSubscriptionModelKey("openai/gpt-5.2")).toBe("gpt-5.2");
    expect(canonicalSubscriptionModelKey("anthropic/claude-opus-5")).toBe(
      "claude-opus-5",
    );
    expect(canonicalSubscriptionModelKey("gpt-5.2")).toBe("gpt-5.2");
  });
});

describe("mergeSubscriptionModelOffers", () => {
  it("groups offers for the same canonical model and keeps each subscription", () => {
    const offers: SubscriptionModelOffer[] = [
      { source: "chatgpt", id: "gpt-5.2", label: "GPT-5.2" },
      { source: "cursor-cli", id: "gpt-5.2", label: "GPT-5.2" },
      { source: "opencode", id: "opencode/gpt-5.2", label: "gpt-5.2" },
      { source: "claude", id: "claude-opus-5", label: "Claude Opus 5" },
    ];

    const merged = mergeSubscriptionModelOffers(offers);
    const gpt = merged.find((row) => row.key === "gpt-5.2");
    expect(gpt?.offers.map((offer) => offer.source)).toEqual([
      "chatgpt",
      "cursor-cli",
      "opencode",
    ]);
    expect(
      merged.find((row) => row.key === "claude-opus-5")?.offers,
    ).toHaveLength(1);
  });
});

describe("countOffersBySource", () => {
  it("counts native models per subscription", () => {
    const offers: SubscriptionModelOffer[] = [
      { source: "chatgpt", id: "gpt-5.2", label: "GPT-5.2" },
      { source: "chatgpt", id: "gpt-5.4", label: "GPT-5.4" },
      { source: "cursor-cli", id: "auto", label: "Auto" },
    ];
    expect(countOffersBySource(offers)).toEqual({
      chatgpt: 2,
      claude: 0,
      "cursor-cli": 1,
      opencode: 0,
    });
  });
});

describe("parseCursorAgentModels", () => {
  it("reads `agent models` id/label lines and skips the heading", () => {
    const stdout = `Available models

auto - Auto (current, default)
gpt-5.2 - GPT-5.2
claude-sonnet-5-thinking-high - Claude Sonnet 5 1M Thinking
`;
    expect(parseCursorAgentModels(stdout)).toEqual([
      { source: "cursor-cli", id: "auto", label: "Auto (current, default)" },
      { source: "cursor-cli", id: "gpt-5.2", label: "GPT-5.2" },
      {
        source: "cursor-cli",
        id: "claude-sonnet-5-thinking-high",
        label: "Claude Sonnet 5 1M Thinking",
      },
    ]);
  });
});

describe("parseOpenCodeModels", () => {
  it("reads one provider/model id per line", () => {
    const stdout = `opencode/claude-sonnet-5
opencode/gpt-5.2
anthropic/claude-opus-4-6
`;
    expect(parseOpenCodeModels(stdout)).toEqual([
      {
        source: "opencode",
        id: "opencode/claude-sonnet-5",
        label: "claude-sonnet-5",
      },
      { source: "opencode", id: "opencode/gpt-5.2", label: "gpt-5.2" },
      {
        source: "opencode",
        id: "anthropic/claude-opus-4-6",
        label: "claude-opus-4-6",
      },
    ]);
  });
});

describe("subscriptionSourceForConnection", () => {
  it("maps host-detected subscription rows, not stored API-key providers", () => {
    expect(
      subscriptionSourceForConnection({
        id: "host:chatgpt",
        provider: "openai",
      }),
    ).toBe("chatgpt");
    expect(
      subscriptionSourceForConnection({
        id: "host:anthropic",
        provider: "anthropic",
      }),
    ).toBe("claude");
    expect(
      subscriptionSourceForConnection({
        id: "host:cursor-cli",
        provider: "cursor-cli",
      }),
    ).toBe("cursor-cli");
    expect(
      subscriptionSourceForConnection({
        id: "conn-1",
        provider: "openai",
      }),
    ).toBeNull();
  });
});

describe("chatgpt auto profile names", () => {
  it("uses a stable prefix so toggles can create and remove them", () => {
    expect(chatgptAutoProfileName("gpt-5.2")).toBe(
      `${CHATGPT_AUTO_PROFILE_PREFIX}gpt-5.2`,
    );
    expect(isChatgptAutoProfileName("sub-chatgpt-gpt-5.2")).toBe(true);
    expect(isChatgptAutoProfileName("my-openai")).toBe(false);
  });
});

describe("subscriptionModelToggleKey", () => {
  it("is unique per subscription even when the model id matches", () => {
    expect(subscriptionModelToggleKey("chatgpt", "gpt-5.2")).toBe(
      "chatgpt:gpt-5.2",
    );
    expect(subscriptionModelToggleKey("cursor-cli", "gpt-5.2")).toBe(
      "cursor-cli:gpt-5.2",
    );
  });
});
