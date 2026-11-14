import React from "react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { defineVisualizer } from "../define";
import { textFromContent } from "../text-content";
import { KeyValueGrid } from "../primitives/key-value-grid";
import { OutputPane } from "../primitives/output-pane";

/**
 * MCP tool visualizer. The observation card names the MCP tool that ran and
 * shows the text the agent received, with errors surfaced as error text rather
 * than folded into the result body. MCP observations carry no exit code, so
 * none is displayed.
 */
export const mcpVisualizer = defineVisualizer({
  actionKinds: ["MCPToolAction"],
  observationKinds: ["MCPToolObservation"],
  Body: function MCPToolBody({ observation }) {
    const { t } = useTranslation("openhands");
    const obs = observation?.observation;

    const toolName = obs?.tool_name ?? "";
    const output = obs ? textFromContent(obs.content) : "";

    return (
      <div className="flex flex-col gap-2">
        {toolName && (
          <KeyValueGrid
            rows={[{ label: t(I18nKey.OBSERVATION$TOOL), value: toolName }]}
          />
        )}
        {obs &&
          (obs.is_error ? (
            <span className="whitespace-pre-wrap text-xs text-danger">
              {t(I18nKey.COMMON$ERROR)}: {output}
            </span>
          ) : (
            <OutputPane output={output} copy />
          ))}
      </div>
    );
  },
});
