/**
 * Shared readers for `file_editor` / `str_replace_editor` action and
 * observation events. Both the Markdown and the sandboxed-frame artifact
 * previews need the touched path and the command, and observations sometimes
 * omit one or both, so the lookup has to consult the originating action.
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

/**
 * Resolves the file path for a file-editor action/observation, including the
 * observation's originating action when the observation omits `path`.
 */
export function getFileEditorEventPath(
  event: OpenHandsEvent,
  correspondingAction?: ActionEvent,
): string | null {
  if (isActionEvent(event) && FILE_EDITOR_ACTION_KINDS.has(event.action.kind)) {
    return (
      (event as ActionEvent<FileEditorAction | StrReplaceEditorAction>).action
        .path || null
    );
  }

  if (
    isObservationEvent(event) &&
    FILE_EDITOR_OBSERVATION_KINDS.has(event.observation.kind)
  ) {
    const path = (
      event as ObservationEvent<
        FileEditorObservation | StrReplaceEditorObservation
      >
    ).observation.path;
    if (path) return path;
    if (
      correspondingAction &&
      FILE_EDITOR_ACTION_KINDS.has(correspondingAction.action.kind)
    ) {
      return (
        (
          correspondingAction as ActionEvent<
            FileEditorAction | StrReplaceEditorAction
          >
        ).action.path || null
      );
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
    return (
      (event as ActionEvent<FileEditorAction | StrReplaceEditorAction>).action
        .command || null
    );
  }

  if (
    isObservationEvent(event) &&
    FILE_EDITOR_OBSERVATION_KINDS.has(event.observation.kind)
  ) {
    const command = (
      event as ObservationEvent<
        FileEditorObservation | StrReplaceEditorObservation
      >
    ).observation.command;
    if (command) return command;
    if (
      correspondingAction &&
      FILE_EDITOR_ACTION_KINDS.has(correspondingAction.action.kind)
    ) {
      return (
        (
          correspondingAction as ActionEvent<
            FileEditorAction | StrReplaceEditorAction
          >
        ).action.command || null
      );
    }
  }

  return null;
}
