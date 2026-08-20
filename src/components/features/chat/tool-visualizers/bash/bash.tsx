import React from "react";
import { useTranslation } from "react-i18next";
import { SecurityRisk } from "#/types/agent-server/core";
import { I18nKey } from "#/i18n/declaration";
import { defineVisualizer } from "../define";
import { textFromContent } from "../text-content";
import { CodeBlock } from "../primitives/code-block";
import { OutputPane } from "../primitives/output-pane";

/**
 * Bash / terminal visualizer. The action card shows the command (plus a risk
 * warning for HIGH/MEDIUM actions); the observation card shows the command and
 * its output (with an exit-code badge). Both the command and the output carry a
 * hover copy button. Covers both the `execute_bash` and `terminal` tools, which
 * carry the same `command` / `content` / `exit_code` fields.
 */
export const bashVisualizer = defineVisualizer({
  actionKinds: ["ExecuteBashAction", "TerminalAction"],
  observationKinds: ["ExecuteBashObservation", "TerminalObservation"],
  Body: function BashBody({ action, observation }) {
    const { t } = useTranslation("openhands");
    const command =
      observation?.observation.command ?? action?.action.command ?? "";
    const risk = action?.security_risk;

    return (
      <div className="flex flex-col gap-2">
        {command && <CodeBlock code={command} language="bash" />}
        {(risk === SecurityRisk.HIGH || risk === SecurityRisk.MEDIUM) && (
          <span className="text-xs text-status-fail-text">
            {t(
              risk === SecurityRisk.HIGH
                ? I18nKey.SECURITY$HIGH_RISK
                : I18nKey.SECURITY$MEDIUM_RISK,
            )}
          </span>
        )}
        {observation && (
          <OutputPane
            output={textFromContent(observation.observation.content)}
            exitCode={observation.observation.exit_code}
          />
        )}
      </div>
    );
  },
});
