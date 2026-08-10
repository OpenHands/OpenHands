import React from "react";
import { getLanguageFromPath } from "#/utils/get-language-from-path";
import { defineVisualizer } from "../define";
import { textFromContent } from "../text-content";
import { CodeBlock } from "../primitives/code-block";
import { DiffView } from "../primitives/diff-view";
import { FilePathChip } from "../primitives/file-path-chip";

/**
 * File-editor visualizer for `file_editor` / `str_replace_editor` tools.
 *
 * Observation card: error → message; `str_replace`/`insert` → diff of the file
 * before vs after; `create`/`view` → the file content. Action card (shown while
 * the edit is in flight): `create` → new content; `str_replace` → diff of the
 * replaced snippet; `view`/`undo_edit` → just the path + range.
 */
export const fileEditorVisualizer = defineVisualizer({
  actionKinds: ["FileEditorAction", "StrReplaceEditorAction"],
  observationKinds: ["FileEditorObservation", "StrReplaceEditorObservation"],
  Body: function FileEditorBody({ action, observation }) {
    const path = observation?.observation.path ?? action?.action.path ?? "";
    const command = observation?.observation.command ?? action?.action.command;
    const language = getLanguageFromPath(path);

    const viewRange = action?.action.view_range;
    const range =
      command === "view" && viewRange
        ? `${viewRange[0]}-${viewRange[1]}`
        : undefined;
    const chip = path ? <FilePathChip path={path} range={range} /> : null;

    if (observation) {
      const obs = observation.observation;
      let body: React.ReactNode = null;
      if (obs.error) {
        body = (
          <span className="whitespace-pre-wrap text-xs text-danger">
            {obs.error}
          </span>
        );
      } else if (obs.old_content != null && obs.new_content != null) {
        // Nullish, not truthy: an empty string is a valid side of the diff —
        // clearing a file or inserting into an empty file must still render it.
        body = <DiffView oldText={obs.old_content} newText={obs.new_content} />;
      } else {
        // `view` returns the snippet the agent saw in `content` (the `cat -n`
        // output) rather than `output`/`new_content`, so fall back to it.
        // Mirrors the markdown path's "prefer content for view" handling.
        const content =
          obs.new_content ||
          obs.output ||
          (obs.content ? textFromContent(obs.content) : "");
        body = content ? (
          <CodeBlock code={content} language={language} />
        ) : null;
      }
      return (
        <div className="flex flex-col gap-2">
          {chip}
          {body}
        </div>
      );
    }

    if (action) {
      const act = action.action;
      let body: React.ReactNode = null;
      if (act.command === "create" && act.file_text) {
        body = <CodeBlock code={act.file_text} language={language} />;
      } else if (
        (act.command === "str_replace" || act.command === "insert") &&
        act.new_str != null
      ) {
        // `insert` carries only `new_str`, no `old_str`. Key on `new_str` and
        // default `old_str` to "" so an in-flight insert shows an addition diff
        // instead of nothing.
        body = <DiffView oldText={act.old_str ?? ""} newText={act.new_str} />;
      }
      return (
        <div className="flex flex-col gap-2">
          {chip}
          {body}
        </div>
      );
    }

    return null;
  },
});
