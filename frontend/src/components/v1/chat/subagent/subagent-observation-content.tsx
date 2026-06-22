import React from "react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { ActionEvent, ObservationEvent } from "#/types/v1/core";
import { TaskObservation } from "#/types/v1/core/base/observation";
import { MarkdownRenderer } from "#/components/features/markdown/markdown-renderer";

interface SubagentObservationContentProps {
  event: ObservationEvent<TaskObservation>;
  correspondingAction?: ActionEvent;
}

const getResultText = (observation: TaskObservation): string =>
  observation.content
    .filter((c) => c.type === "text")
    .map((c) => c.text)
    .join("\n");

const getResultImages = (observation: TaskObservation): string[] =>
  observation.content
    .filter((c) => c.type === "image")
    .flatMap((c) => c.image_urls);

const getQuery = (correspondingAction?: ActionEvent): string | undefined => {
  const action = correspondingAction?.action;
  return action?.kind === "TaskAction" ? action.prompt : undefined;
};

/**
 * Rich card body for a sub-agent task (TaskToolSet). Shows which sub-agent ran
 * the task, its task id, the query it was given (from the paired action) and
 * the result it returned. Mirrors the TaskTrackerObservation visualizer.
 */
export function SubagentObservationContent({
  event,
  correspondingAction,
}: SubagentObservationContentProps): React.ReactNode {
  const { t } = useTranslation();
  const { observation } = event;
  const query = getQuery(correspondingAction);
  const resultText = getResultText(observation);
  const resultImages = getResultImages(observation);

  return (
    <div className="flex flex-col gap-3 text-neutral-300">
      <div className="flex flex-col gap-1">
        <div className="flex gap-1">
          <span className="font-bold">
            {t(I18nKey.SUBAGENT_OBSERVATION$SUBAGENT)}:
          </span>
          <span>{observation.subagent}</span>
        </div>
        <div className="flex gap-1">
          <span className="font-bold">
            {t(I18nKey.SUBAGENT_OBSERVATION$TASK_ID)}:
          </span>
          <span>{observation.task_id}</span>
        </div>
      </div>

      {query && (
        <div className="flex flex-col gap-1">
          <span className="font-bold">
            {t(I18nKey.SUBAGENT_OBSERVATION$QUERY)}
          </span>
          <div className="rounded-md bg-neutral-700 p-2 whitespace-pre-wrap break-words">
            {query}
          </div>
        </div>
      )}

      <div className="flex flex-col gap-1">
        <span className="font-bold">
          {t(I18nKey.SUBAGENT_OBSERVATION$RESULT)}
        </span>
        {resultText && <MarkdownRenderer>{resultText}</MarkdownRenderer>}
        {resultImages.map((url) => (
          <img
            key={url}
            src={url}
            alt={t(I18nKey.SUBAGENT_OBSERVATION$RESULT)}
            className="max-w-full rounded-md"
          />
        ))}
      </div>
    </div>
  );
}
