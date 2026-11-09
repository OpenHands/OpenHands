import type { TFunction } from "i18next";
import { describe, expect, it } from "vitest";
import { I18nKey } from "#/i18n/declaration";
import type { Automation } from "#/types/automation";
import { buildAutomationFlowDiagram } from "#/utils/automation-flow-diagram";

// Echo the key and any interpolated values, so assertions name what was asked for.
const t = ((key: string, options?: Record<string, string>) =>
  options ? `${key}[${Object.values(options).join("|")}]` : key) as TFunction;

const base: Automation = {
  id: "auto-1",
  name: "Nightly",
  prompt: null,
  trigger: { type: "cron", schedule: "0 2 * * *" },
  enabled: true,
  created_at: "2026-01-01T00:00:00Z",
  updated_at: "2026-01-01T00:00:00Z",
};

describe("buildAutomationFlowDiagram", () => {
  it("draws a scheduled prompt automation as schedule, agent, result", () => {
    // Act
    const diagram = buildAutomationFlowDiagram(
      { ...base, prompt: "Summarize the week." },
      t,
    );

    // Assert
    expect(diagram.split("\n")[0]).toBe("flowchart LR");
    expect(diagram).toContain(
      `${I18nKey.AUTOMATIONS$FLOW$SCHEDULE}[0 2 * * *]`,
    );
    expect(diagram).toContain(I18nKey.AUTOMATIONS$FLOW$RUN_PROMPT);
    expect(diagram).toContain(I18nKey.AUTOMATIONS$FLOW$RESULT);
    expect(diagram).not.toContain(I18nKey.AUTOMATIONS$FLOW$MATCHES_FILTER);
  });

  it("adds the filter decision for a filtered event and names the command", () => {
    // Act
    const diagram = buildAutomationFlowDiagram(
      {
        ...base,
        // Short values: the echoing `t` already spends most of the label.
        entrypoint: "./run.sh",
        trigger: {
          type: "event",
          source: "gh",
          on: ["push", "pr"],
          filter: "repository.full_name == 'a/b'",
        },
      },
      t,
    );

    // Assert
    expect(diagram).toContain(
      `${I18nKey.AUTOMATIONS$FLOW$EVENT}[gh] (push, pr)`,
    );
    expect(diagram).toContain(I18nKey.AUTOMATIONS$FLOW$MATCHES_FILTER);
    expect(diagram).toContain(I18nKey.AUTOMATIONS$FLOW$IGNORED);
    expect(diagram).toContain(
      `${I18nKey.AUTOMATIONS$FLOW$RUN_COMMAND}[./run.sh]`,
    );
  });

  it("shows a continued conversation instead of a new run", () => {
    // Act
    const diagram = buildAutomationFlowDiagram(
      {
        ...base,
        prompt: "p",
        trigger: {
          type: "event",
          source: "slack",
          destination: "continue_conversation",
        },
      },
      t,
    );

    // Assert
    expect(diagram).toContain(I18nKey.AUTOMATIONS$FLOW$CONTINUE);
    expect(diagram).not.toContain(I18nKey.AUTOMATIONS$FLOW$RUN_PROMPT);
  });

  it("keeps automation-supplied text from breaking a label", () => {
    // Act
    const diagram = buildAutomationFlowDiagram(
      { ...base, entrypoint: 'sh -c "run"\n--> X[injected]' },
      t,
    );

    // Assert: still four lines, and the command's quotes are entity-encoded
    // inside its own quoted label rather than closing it.
    const lines = diagram.split("\n");
    expect(lines).toHaveLength(4);
    const work = lines.find((line) => line.includes("W["))!;
    expect(work).toContain("#quot;run#quot;");
    expect(work.match(/"/g)).toHaveLength(2);
  });
});
