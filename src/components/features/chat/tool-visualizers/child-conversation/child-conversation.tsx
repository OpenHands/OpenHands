import React from "react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { MarkdownRenderer } from "#/components/features/markdown/markdown-renderer";
import { CopyableContentWrapper } from "#/components/shared/buttons/copyable-content-wrapper";
import { buildAgentCanvasPath } from "#/utils/base-path";
import { defineVisualizer } from "../define";
import { textFromContent } from "../text-content";
import { KeyValueGrid } from "../primitives/key-value-grid";

/**
 * Child conversation visualizer for the agent-server's
 * `start_child_conversation` tool, which launches the child on the backend that
 * runs the parent. The action card shows the isolation asked for and the brief
 * the child receives; the observation card adds the child's id, status,
 * workspace and a link into the conversation the server created.
 */
export const childConversationVisualizer = defineVisualizer({
  actionKinds: ["StartChildConversationAction"],
  observationKinds: ["StartChildConversationObservation"],
  Body: function ChildConversationBody({ action, observation }) {
    const { t } = useTranslation("openhands");
    const act = action?.action;
    const obs = observation?.observation;

    const task = act?.task?.trim() ?? "";
    const target = obs?.conversation_id
      ? buildAgentCanvasPath(`/conversations/${obs.conversation_id}`)
      : obs?.url;
    const result = obs ? textFromContent(obs.content).trim() : "";

    const rows = [
      ...(obs?.status
        ? [{ label: t(I18nKey.COMMON$STATUS), value: obs.status }]
        : []),
      ...(obs?.isolation
        ? [
            {
              label: t(I18nKey.CHILD_CONVERSATION$ISOLATION),
              value: obs.isolation,
            },
          ]
        : []),
      ...(obs?.workspace
        ? [
            {
              label: t(I18nKey.CONVERSATION$OVERVIEW_WORKSPACE),
              value: obs.workspace,
            },
          ]
        : []),
    ];

    return (
      <div className="flex flex-col gap-2">
        {rows.length > 0 && <KeyValueGrid rows={rows} />}
        {obs?.is_error && result && (
          <div className="overflow-auto rounded border border-surface-raised bg-surface-raised p-2 text-xs text-danger">
            <MarkdownRenderer includeStandard includeHeadings>
              {result}
            </MarkdownRenderer>
          </div>
        )}
        {target && !obs?.is_error && (
          <a
            href={target}
            className="w-fit text-xs text-primary underline"
            data-testid="child-conversation-link"
          >
            {t(I18nKey.CONVERSATION$OPEN_CREATED)}
          </a>
        )}
        {task && (
          <div className="flex flex-col gap-1">
            <span className="text-xs text-muted">
              {t(I18nKey.AUTOMATIONS$DETAIL$TASK_LABEL)}
            </span>
            <CopyableContentWrapper text={task}>
              <div className="overflow-auto rounded border border-surface-raised bg-surface-raised p-2 text-xs text-foreground">
                <MarkdownRenderer includeStandard includeHeadings>
                  {task}
                </MarkdownRenderer>
              </div>
            </CopyableContentWrapper>
          </div>
        )}
      </div>
    );
  },
});
