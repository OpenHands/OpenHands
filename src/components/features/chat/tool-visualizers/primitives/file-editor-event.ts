/**
 * Shared file-editor event helpers used by the inline artifact previews
 * (Markdown and Mermaid). Both previews keep their own extension allow-list
 * but share the same path/command resolution from an action or observation.
 */
import type {
  ActionEvent,
  ObservationEvent,
  OpenHandsEvent,
} from "#/types/agent-server/core";
import {
  isActionEvent,
  isObservationEvent,
} from "#/types/agent-server/type-guards";
import type {
  FileEditorAction,
  StrReplaceEditorAction,
} from "#/types/agent-server/core/base/action";
import type {
  FileEditorObservation,
  StrReplaceEditorObservation,
} from "#/types/agent-server/core/base/observation";

export const FILE_EDITOR_ACTION_KINDS = new Set([
  "FileEditorAction",
  "StrReplaceEditorAction",
]);
export const FILE_EDITOR_OBSERVATION_KINDS = new Set([
  "FileEditorObservation",
  "StrReplaceEditorObservation",
]);

type FileEditorActionEvent = ActionEvent<
  FileEditorAction | StrReplaceEditorAction
>;
type FileEditorObservationEvent = ObservationEvent<
  FileEditorObservation | StrReplaceEditorObservation
>;

/**
 * Resolves the file path for a file-editor action/observation, including the
 * observation's originating action when the observation omits `path`.
 */
export function getFileEditorEventPath(
  event: OpenHandsEvent,
  correspondingAction?: ActionEvent,
): string | null {
  if (isActionEvent(event) && FILE_EDITOR_ACTION_KINDS.has(event.action.kind)) {
    return (event as FileEditorActionEvent).action.path || null;
  }

  if (
    isObservationEvent(event) &&
    FILE_EDITOR_OBSERVATION_KINDS.has(event.observation.kind)
  ) {
    const path = (event as FileEditorObservationEvent).observation.path;
    if (path) return path;
    if (
      correspondingAction &&
      FILE_EDITOR_ACTION_KINDS.has(correspondingAction.action.kind)
    ) {
      return (correspondingAction as FileEditorActionEvent).action.path || null;
    }
  }

  return null;
}

/**
 * Resolves the file-editor command (`create` / `view` / …), including the
 * observation's originating action when the observation omits `command`.
 */
export function getFileEditorEventCommand(
  event: OpenHandsEvent,
  correspondingAction?: ActionEvent,
): string | null {
  if (isActionEvent(event) && FILE_EDITOR_ACTION_KINDS.has(event.action.kind)) {
    return (event as FileEditorActionEvent).action.command || null;
  }

  if (
    isObservationEvent(event) &&
    FILE_EDITOR_OBSERVATION_KINDS.has(event.observation.kind)
  ) {
    const command = (event as FileEditorObservationEvent).observation.command;
    if (command) return command;
    if (
      correspondingAction &&
      FILE_EDITOR_ACTION_KINDS.has(correspondingAction.action.kind)
    ) {
      return (
        (correspondingAction as FileEditorActionEvent).action.command || null
      );
    }
  }

  return null;
}

/**
 * True for file-editor *create* events whose path passes `matchesPath`.
 *
 * Used to keep those cards expanded and outside collapsed action groups so
 * the inline preview is visible by default. Reads/edits of matching files
 * stay on the normal groupable path.
 */
export function isFileEditorCreateFor(matchesPath: (path: string) => boolean) {
  return (
    event: OpenHandsEvent,
    correspondingAction?: ActionEvent,
  ): boolean => {
    const path = getFileEditorEventPath(event, correspondingAction);
    const command = getFileEditorEventCommand(event, correspondingAction);
    return Boolean(path && command === "create" && matchesPath(path));
  };
}
