import type { TFunction } from "i18next";
import { I18nKey } from "#/i18n/declaration";
import type { Automation } from "#/types/automation";

const MAX_LABEL_LENGTH = 48;

/**
 * A quoted Mermaid label. Values come from the automation (schedules, event
 * names, commands), so quotes and line breaks are neutralized and long values
 * shortened to keep the node on one line.
 */
function label(text: string): string {
  const oneLine = text.replace(/\s+/g, " ").trim();
  const short =
    oneLine.length > MAX_LABEL_LENGTH
      ? `${oneLine.slice(0, MAX_LABEL_LENGTH - 1)}…`
      : oneLine;
  return `"${short.replace(/"/g, "#quot;")}"`;
}

function triggerLabel(automation: Automation, t: TFunction): string {
  const { trigger } = automation;
  if (trigger.type === "event") {
    const source = t(I18nKey.AUTOMATIONS$FLOW$EVENT, {
      source: trigger.source ?? "",
    });
    const on = Array.isArray(trigger.on) ? trigger.on.join(", ") : trigger.on;
    return on && on !== "*" ? `${source} (${on})` : source;
  }
  return t(I18nKey.AUTOMATIONS$FLOW$SCHEDULE, {
    schedule: trigger.schedule_human ?? trigger.schedule ?? "",
  });
}

function workLabel(automation: Automation, t: TFunction): string {
  if (automation.trigger.destination === "continue_conversation") {
    return t(I18nKey.AUTOMATIONS$FLOW$CONTINUE);
  }
  if (automation.prompt) return t(I18nKey.AUTOMATIONS$FLOW$RUN_PROMPT);
  return t(I18nKey.AUTOMATIONS$FLOW$RUN_COMMAND, {
    command: automation.entrypoint ?? "",
  });
}

/**
 * A left-to-right Mermaid flowchart of how an automation runs, built from its
 * own settings: what starts it, the event filter when there is one, what runs,
 * and where the result goes. Used when neither the automation nor a template
 * describes it, so every automation has a diagram.
 */
export function buildAutomationFlowDiagram(
  automation: Automation,
  t: TFunction,
): string {
  const lines = [
    "flowchart LR",
    `  A([${label(triggerLabel(automation, t))}])`,
  ];
  const hasFilter =
    automation.trigger.type === "event" && Boolean(automation.trigger.filter);
  if (hasFilter) {
    lines.push(
      `  A --> F{{${label(t(I18nKey.AUTOMATIONS$FLOW$MATCHES_FILTER))}}}`,
      `  F -.->|${label(t(I18nKey.AUTOMATIONS$FLOW$NO))}| X[${label(t(I18nKey.AUTOMATIONS$FLOW$IGNORED))}]`,
      `  F --> W[${label(workLabel(automation, t))}]`,
    );
  } else {
    lines.push(`  A --> W[${label(workLabel(automation, t))}]`);
  }
  lines.push(`  W --> R[${label(t(I18nKey.AUTOMATIONS$FLOW$RESULT))}]`);
  return lines.join("\n");
}
