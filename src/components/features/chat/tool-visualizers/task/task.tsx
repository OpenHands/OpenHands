import React from "react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { MarkdownRenderer } from "#/components/features/markdown/markdown-renderer";
import { CopyableContentWrapper } from "#/components/shared/buttons/copyable-content-wrapper";
import { defineVisualizer } from "../define";
import { textFromContent } from "../text-content";
import { KeyValueGrid } from "../primitives/key-value-grid";

/**
 * Labelled markdown block with a hover copy button, used for both the query the
 * parent agent sent and the result the subagent returned.
 */
function MarkdownSection({
  label,
  text,
  isError = false,
}: {
  label: string;
  text: string;
  isError?: boolean;
}) {
  return (
    <div className="flex flex-col gap-1">
      <span className="text-xs text-muted">{label}</span>
      <CopyableContentWrapper text={text}>
        <div
          className={`overflow-auto rounded border border-surface-raised bg-surface-raised p-2 text-xs ${
            isError ? "text-danger" : "text-foreground"
          }`}
        >
          <MarkdownRenderer includeStandard includeHeadings>
            {text}
          </MarkdownRenderer>
        </div>
      </CopyableContentWrapper>
    </div>
  );
}

/**
 * Task visualizer for the `task` tool, which delegates work to a spawned
 * subagent. The action card shows which subagent is being run and the query
 * (the prompt the parent agent sent); the observation card adds the task id and
 * the subagent's returned result. Until the observation arrives, the card just
 * shows the query, so an in-flight delegation is still legible. Both the query
 * and the result are rendered as markdown with a copy button.
 */
export const taskVisualizer = defineVisualizer({
  actionKinds: ["TaskAction"],
  observationKinds: ["TaskObservation"],
  Body: function TaskBody({ action, observation }) {
    const { t } = useTranslation("openhands");
    const act = action?.action;
    const obs = observation?.observation;

    const subagent = obs?.subagent ?? act?.subagent_type ?? "";
    const taskId = obs?.task_id;
    const query = act?.prompt?.trim() ?? "";
    const answer = obs ? textFromContent(obs.content).trim() : "";

    const rows = [
      ...(subagent
        ? [{ label: t(I18nKey.TASK$SUBAGENT), value: subagent }]
        : []),
      ...(taskId ? [{ label: t(I18nKey.TASK$TASK_ID), value: taskId }] : []),
    ];

    return (
      <div className="flex flex-col gap-2">
        {rows.length > 0 && <KeyValueGrid rows={rows} />}
        {query && (
          <MarkdownSection label={t(I18nKey.TASK$QUERY)} text={query} />
        )}
        {answer && (
          <MarkdownSection
            label={t(I18nKey.TASK$RESULT)}
            text={answer}
            isError={obs?.is_error}
          />
        )}
      </div>
    );
  },
});
