/**
 * File-editor visualizer for `file_editor` / `str_replace_editor` tools.
 *
 * Observation card: error → message; `str_replace`/`insert` → diff of the file
 * before vs after; `create`/`view` → the file content. Action card (shown while
 * the edit is in flight): `create` → new content; `str_replace` → diff of the
 * replaced snippet; `view`/`undo_edit` → just the path + range.
 *
 * Created Markdown artifacts get a height-clipped rich preview with a View bar
 * that opens the Files drawer, instead of dumping the full source into a code
 * block. Reads/edits of `.md` files keep the normal CodeBlock / DiffView path.
 */
import React from "react";
import { getLanguageFromPath } from "#/utils/get-language-from-path";
import { toWorkspaceRelativePath } from "#/utils/workspace-relative-path";
import { useOptionalConversationId } from "#/hooks/use-conversation-id";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useSelectConversationTab } from "#/hooks/use-select-conversation-tab";
import { useFilesTabStore } from "#/stores/files-tab-store";
import type {
  FileEditorAction,
  StrReplaceEditorAction,
} from "#/types/agent-server/core/base/action";
import type {
  FileEditorObservation,
  StrReplaceEditorObservation,
} from "#/types/agent-server/core/base/observation";
import { defineVisualizer, VisualizerProps } from "../define";
import { textFromContent } from "../text-content";
import { CodeBlock } from "../primitives/code-block";
import { DiffView } from "../primitives/diff-view";
import { FilePathChip } from "../primitives/file-path-chip";
import {
  isMarkdownFilePath,
  MarkdownFilePreview,
} from "../primitives/markdown-file-preview";

type FileEditorCardProps = VisualizerProps<
  FileEditorAction | StrReplaceEditorAction,
  FileEditorObservation | StrReplaceEditorObservation
>;

/** The path the event touched; observations win because they are authoritative. */
const resolvePath = ({ action, observation }: FileEditorCardProps): string =>
  observation?.observation.path ?? action?.action.path ?? "";

interface FileEditorCardBodyProps extends FileEditorCardProps {
  /**
   * Deep-links the file into the Files drawer. Absent outside a conversation
   * route, where there is no drawer to open; the card then stays read-only.
   * Also omitted for in-flight creates so View does not open a missing file.
   */
  onOpenFile?: () => void;
}

function FileEditorCardBody({
  action,
  observation,
  onOpenFile,
}: FileEditorCardBodyProps) {
  const path = resolvePath({ action, observation });
  const command = observation?.observation.command ?? action?.action.command;
  const language = getLanguageFromPath(path);

  const viewRange = action?.action.view_range;
  const range =
    command === "view" && viewRange
      ? `${viewRange[0]}-${viewRange[1]}`
      : undefined;
  const chip = path ? (
    <FilePathChip path={path} range={range} onClick={onOpenFile} />
  ) : null;

  const renderFileContent = (content: string) => {
    // Only created Markdown artifacts get the rich preview — `view` returns
    // `cat -n` numbered snippets that must stay in a CodeBlock.
    if (path && command === "create" && isMarkdownFilePath(path)) {
      // Markdown artifacts own their card (clipped preview + optional View),
      // so skip the separate path chip to avoid a duplicate filename affordance.
      return {
        chip: null as React.ReactNode,
        body: (
          <MarkdownFilePreview
            content={content}
            path={path}
            onView={onOpenFile}
          />
        ),
      };
    }
    return {
      chip,
      body: <CodeBlock code={content} language={language} />,
    };
  };

  if (observation) {
    const obs = observation.observation;
    let body: React.ReactNode = null;
    let leadingChip: React.ReactNode = chip;
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
      if (content) {
        const rendered = renderFileContent(content);
        leadingChip = rendered.chip;
        body = rendered.body;
      }
    }
    return (
      <div className="flex flex-col gap-2">
        {leadingChip}
        {body}
      </div>
    );
  }

  if (action) {
    const act = action.action;
    let body: React.ReactNode = null;
    let leadingChip: React.ReactNode = chip;
    if (act.command === "create" && act.file_text) {
      const rendered = renderFileContent(act.file_text);
      leadingChip = rendered.chip;
      body = rendered.body;
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
        {leadingChip}
        {body}
      </div>
    );
  }

  return null;
}

/**
 * Conversation-scoped wrapper: owns the Files-drawer deep link. Split from the
 * card so the conversation-only hooks below never run outside a conversation
 * route (they throw on a missing conversation id).
 */
function FileEditorCardWithDrawerLink({
  conversationId,
  ...props
}: FileEditorCardProps & { conversationId: string }) {
  const { navigateToTab } = useSelectConversationTab();
  const setSelectedPath = useFilesTabStore((state) => state.setSelectedPath);
  const { data: conversation } = useActiveConversation();
  const path = resolvePath(props);
  const workingDir = conversation?.workspace?.working_dir;

  // Only deep-link once the observation exists — the file is not guaranteed
  // to be on disk while the create action is still in flight.
  const onOpenFile =
    path && props.observation
      ? () => {
          setSelectedPath(
            toWorkspaceRelativePath(path, workingDir),
            conversationId,
          );
          navigateToTab("files");
        }
      : undefined;

  return <FileEditorCardBody {...props} onOpenFile={onOpenFile} />;
}

export const fileEditorVisualizer = defineVisualizer({
  actionKinds: ["FileEditorAction", "StrReplaceEditorAction"],
  observationKinds: ["FileEditorObservation", "StrReplaceEditorObservation"],
  Body: function FileEditorBody(props) {
    const { conversationId } = useOptionalConversationId();

    return conversationId ? (
      <FileEditorCardWithDrawerLink
        {...props}
        conversationId={conversationId}
      />
    ) : (
      <FileEditorCardBody {...props} />
    );
  },
});
