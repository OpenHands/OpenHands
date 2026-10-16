import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import i18next from "i18next";
import { I18nKey } from "#/i18n/declaration";
import { buildAutomationMetadataPills } from "#/components/features/automations/build-automation-pills";
import type { SkillCardPill } from "#/components/features/skills/skill-card-pill-row";
import type { Automation } from "#/types/automation";
import translationJson from "#/i18n/translation.json";

function buildAutomation(overrides: Partial<Automation> = {}): Automation {
  return {
    id: "automation-1",
    name: "Triage",
    prompt: "Triage the issue.",
    enabled: true,
    trigger: { type: "event", on: "issue.updated", source: "linear" },
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
    ...overrides,
  };
}

function renderPills(pills: SkillCardPill[]) {
  render(
    <div>
      {pills.map((pill) => (
        <span key={pill.id} data-testid={`pill-${pill.id}`}>
          {pill.node}
        </span>
      ))}
    </div>,
  );
}

describe("buildAutomationMetadataPills", () => {
  it("puts the event and source on separate pills", () => {
    const pills = buildAutomationMetadataPills(buildAutomation(), "unused");

    expect(pills.map((pill) => pill.id)).toEqual([
      "event-trigger",
      "event-source",
    ]);

    renderPills(pills);

    expect(screen.getByTestId("pill-event-trigger")).toHaveTextContent(
      "issue.updated",
    );
    expect(screen.getByTestId("pill-event-trigger")).not.toHaveTextContent(
      "linear",
    );
    expect(screen.getByTestId("pill-event-source")).toHaveTextContent("Linear");
    expect(
      screen.getByTestId("pill-event-source").firstElementChild,
    ).toHaveClass("py-0.5");
    expect(screen.getByTestId("automation-source-logo")).toBeInTheDocument();
  });

  it("renders a fallback icon when the source is not in the catalog", () => {
    renderPills(
      buildAutomationMetadataPills(
        buildAutomation({
          trigger: { type: "event", on: "alert.fired", source: "custom-pager" },
        }),
        "unused",
      ),
    );

    expect(screen.getByTestId("pill-event-source")).toHaveTextContent(
      "Custom-Pager",
    );
    expect(screen.getByTestId("automation-source-logo")).toBeInTheDocument();
  });

  it("omits the source pill when the event has no source", () => {
    const pills = buildAutomationMetadataPills(
      buildAutomation({
        trigger: { type: "event", on: "pull_request.opened" },
      }),
      "unused",
    );

    expect(pills.map((pill) => pill.id)).toEqual(["event-trigger"]);
  });

  it("pluralizes RECOMMENDED_AUTOMATIONS$MISSING_CONNECT for single and multiple counts", () => {
    const tOne =
      translationJson.RECOMMENDED_AUTOMATIONS$MISSING_CONNECT_one.en.replace(
        "{{count}}",
        "1",
      );
    const tOther =
      translationJson.RECOMMENDED_AUTOMATIONS$MISSING_CONNECT_other.en.replace(
        "{{count}}",
        "2",
      );
    expect(tOne).toBe("1 MCP to connect before launch");
    expect(tOther).toBe("2 MCPs to connect before launch");
  });

  it("correctly resolves plural forms in i18next", async () => {
    const i18n = i18next.createInstance();
    await i18n.init({
      lng: "en",
      resources: {
        en: {
          openhands: {
            RECOMMENDED_AUTOMATIONS$MISSING_CONNECT_one:
              translationJson.RECOMMENDED_AUTOMATIONS$MISSING_CONNECT_one.en,
            RECOMMENDED_AUTOMATIONS$MISSING_CONNECT_other:
              translationJson.RECOMMENDED_AUTOMATIONS$MISSING_CONNECT_other.en,
            RECOMMENDED_AUTOMATIONS$MISSING_CONNECT:
              translationJson.RECOMMENDED_AUTOMATIONS$MISSING_CONNECT.en,
          },
        },
      },
      defaultNS: "openhands",
    });

    expect(
      i18n.t(I18nKey.RECOMMENDED_AUTOMATIONS$MISSING_CONNECT, { count: 1 }),
    ).toBe("1 MCP to connect before launch");
    expect(
      i18n.t(I18nKey.RECOMMENDED_AUTOMATIONS$MISSING_CONNECT, { count: 2 }),
    ).toBe("2 MCPs to connect before launch");
  });
});
