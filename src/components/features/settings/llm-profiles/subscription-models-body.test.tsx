import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { I18nextProvider, initReactI18next } from "react-i18next";
import i18n from "i18next";
import { SubscriptionModelsBody } from "#/components/features/settings/llm-profiles/subscription-models-body";
import {
  mergeSubscriptionModelOffers,
  type SubscriptionModelOffer,
} from "#/utils/subscription-model-catalog";

const testI18n = i18n.createInstance();
void testI18n.use(initReactI18next).init({
  lng: "en",
  fallbackLng: "en",
  ns: ["openhands"],
  defaultNS: "openhands",
  resources: {
    en: {
      openhands: {
        SETTINGS$LLM_AUTH_TYPE_SUBSCRIPTION: "ChatGPT subscription",
        SETTINGS$LLM_AUTH_TYPE_CLAUDE_SUBSCRIPTION: "Claude subscription",
        SETTINGS$SUBSCRIPTION_MODEL_ALSO_ON: "Also in {{sources}}",
        SETTINGS$SUBSCRIPTION_MODEL_TOGGLE: "Include {{model}}",
      },
    },
  },
  interpolation: { escapeValue: false },
});

function renderBody(
  offers: SubscriptionModelOffer[],
  onToggle = vi.fn(),
  enabledIds: string[] | "all" = "all",
) {
  const rows = mergeSubscriptionModelOffers(offers);
  return render(
    <I18nextProvider i18n={testI18n}>
      <SubscriptionModelsBody
        offers={offers}
        rows={rows}
        isOfferEnabled={(offer) =>
          enabledIds === "all" ||
          enabledIds.includes(`${offer.source}:${offer.id}`)
        }
        onToggle={onToggle}
      />
    </I18nextProvider>,
  );
}

describe("SubscriptionModelsBody", () => {
  const offers: SubscriptionModelOffer[] = [
    { source: "chatgpt", id: "gpt-5.2", label: "GPT-5.2" },
    { source: "cursor-cli", id: "gpt-5.2", label: "GPT-5.2" },
    { source: "claude", id: "claude-opus-5", label: "Claude Opus 5" },
  ];

  it("groups models by subscription and notes when another subscription has the same model", () => {
    renderBody(offers);

    expect(
      screen.getByTestId("subscription-model-group-chatgpt"),
    ).toHaveTextContent(/ChatGPT subscription|LLM_AUTH_TYPE_SUBSCRIPTION/);
    expect(
      screen.getByTestId("subscription-model-group-cursor-cli"),
    ).toHaveTextContent("Cursor CLI");
    expect(
      screen.getByTestId("subscription-model-group-claude"),
    ).toHaveTextContent(
      /Claude subscription|LLM_AUTH_TYPE_CLAUDE_SUBSCRIPTION/,
    );
    expect(screen.getAllByText("GPT-5.2")).toHaveLength(2);
    expect(
      screen.getAllByTestId("subscription-model-also-on")[0],
    ).toHaveTextContent(/Also in Cursor CLI|SUBSCRIPTION_MODEL_ALSO_ON/);
  });

  it("toggles a single subscription copy of a shared model", async () => {
    const user = userEvent.setup();
    const onToggle = vi.fn();
    renderBody(offers, onToggle);

    const switches = screen.getAllByRole("switch");
    await user.click(switches[0]);

    expect(onToggle).toHaveBeenCalledWith(offers[0], false);
  });
});
