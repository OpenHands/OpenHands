import React from "react";
import { useMatch } from "react-router";
import { useOptionalConversationId } from "#/hooks/use-conversation-id";
import { PROJECTS_PATH } from "#/utils/project-paths";
import { useActiveConversation } from "./use-active-conversation";
import { useProjectsEnabled } from "./use-projects";

export interface ProjectView {
  projectId: string;
  /** Whether the project's own page is open. */
  onProjectPage: boolean;
}

/**
 * The project the open conversation belongs to (its `project` tag), or
 * undefined when it belongs to none or Projects is turned off.
 */
export function useConversationProjectId(): string | undefined {
  const enabled = useProjectsEnabled();
  const { data: conversation } = useActiveConversation();
  return enabled ? (conversation?.tags?.project ?? undefined) : undefined;
}

/**
 * The project the user is in: the one whose page is open, or the one the
 * open conversation belongs to. Undefined outside a project view.
 *
 * While the next conversation loads, the last answer holds, so walking from
 * one project conversation to another does not flash the full list.
 */
export function useProjectView(): ProjectView | undefined {
  const enabled = useProjectsEnabled();
  const page = useMatch(`${PROJECTS_PATH}/:projectId`);
  const { conversationId } = useOptionalConversationId();
  const { data: conversation } = useActiveConversation();
  const last = React.useRef<ProjectView | undefined>(undefined);

  let view: ProjectView | undefined;
  if (!enabled) view = undefined;
  else if (page?.params.projectId) {
    view = { projectId: page.params.projectId, onProjectPage: true };
  } else if (conversationId && !conversation) {
    view = last.current;
  } else {
    const projectId = conversation?.tags?.project;
    view = projectId ? { projectId, onProjectPage: false } : undefined;
  }
  last.current = view;
  return view;
}
