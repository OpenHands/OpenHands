import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { SlashCommandMenu } from "#/components/features/chat/components/slash-command-menu";
import { SlashCommandMessages } from "#/components/features/chat/slash-command-messages";
import type { SlashCommandItem } from "#/types/slash-command";
import { BUILT_IN_COMMANDS, HELP_COMMAND } from "#/utils/constants";

vi.mock("react-i18next", async (importOriginal) => {
  const actual = await importOriginal<typeof import("react-i18next")>();
  const definitions = await import("#/i18n/translation.json");
  const translations = definitions.default as Record<
    string,
    Record<string, string>
  >;
  return {
    ...actual,
    useTranslation: () => ({
      t: (key: string) => translations[key]?.de ?? key,
      i18n: { language: "de", exists: () => true },
    }),
  };
});

describe("slash-command localization", () => {
  it("uses identical localized built-in copy in autocomplete and /help", () => {
    Element.prototype.scrollIntoView = vi.fn();
    const help = BUILT_IN_COMMANDS.find(
      (item) => item.command === HELP_COMMAND,
    );
    expect(help).toBeDefined();
    const dynamicSkill: SlashCommandItem = {
      command: "/review",
      skill: {
        name: "review",
        type: "agentskills",
        source: "project",
        description: "Backend supplied description",
        triggers: ["/review"],
      },
    };
    const commands = [help!, dynamicSkill];

    render(
      <>
        <SlashCommandMenu
          items={commands}
          selectedIndex={0}
          onSelect={vi.fn()}
        />
        <SlashCommandMessages
          outputScopeId="conversation-1"
          outputs={[
            {
              id: "help-output",
              kind: "help",
              status: "ready",
              invocationOrder: 0,
              timelineBoundaryEventId: null,
              commands,
            },
          ]}
        />
      </>,
    );

    expect(screen.getAllByText("Verfügbare Befehle anzeigen")).toHaveLength(2);
    expect(screen.queryByText("Display available commands")).toBeNull();
    expect(screen.getByText("Backend supplied description")).toBeVisible();
  });
});
