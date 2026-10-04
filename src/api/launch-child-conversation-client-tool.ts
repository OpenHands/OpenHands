import {
  CHILD_CONVERSATION_ISOLATIONS,
  CHILD_CONVERSATION_RESULT_PREFIX,
  CHILD_CONVERSATION_TARGETS,
  LAUNCH_CHILD_CONVERSATION_TOOL_NAME,
} from "#/constants/child-conversation";
import type { ClientToolSpec } from "./canvas-ui-client-tool";

export {
  LAUNCH_CHILD_CONVERSATION_ACTION_KIND,
  LAUNCH_CHILD_CONVERSATION_TOOL_NAME,
} from "#/constants/child-conversation";

const LAUNCH_CHILD_CONVERSATION_DESCRIPTION = `Start a NEW, independent conversation to work on a well-scoped task in parallel with this one, without the user having to run any CLI command. The new conversation is recorded as a child of this one.

The child runs on its own. It does not block you, you do not see its output,
and it cannot see this conversation's history — everything it needs must be in
the task brief.

Choosing target:

* target="local" — runs on the same machine, in this conversation's workspace.
  Fast, no repository clone, no sandbox provisioning. Use this by default when
  the work is on code that is already checked out here.

* target="cloud" — runs on OpenHands Cloud in its own isolated sandbox, from a
  git repository. Use this when the work should not touch the user's machine or
  when it needs a repository that is not checked out locally. Requires the user
  to have an OpenHands Cloud backend connected in Agent Canvas; if none is
  connected you will be told so and should fall back to target="local".

Writing the task brief:

* Put everything the child needs in "task": the goal, the relevant file paths,
  the constraints, the expected deliverable, and how it should report back.
* Keep each child's scope independent of its siblings so parallel children do
  not fight over the same files.
* One call per delegated task. Do NOT call this tool twice for the same task.

Parameter rules (a call that breaks one of these launches nothing and comes
back with corrective guidance):

* "repository" and "branch" apply to target="cloud" only. A local child always
  inherits this conversation's workspace, so do not pass them with
  target="local".
* "isolation" applies to target="local" only. Cloud sandboxes are always
  isolated, so do not pass it with target="cloud".
* isolation="worktree" (the default) gives the child its own git worktree and
  branch, cut from the repository's default branch — it will NOT see this
  conversation's uncommitted or committed work. isolation="shared" puts the
  child in this conversation's exact directory; only choose it when the child
  genuinely must see work in progress, and expect the two agents to conflict.

What comes back: this tool is acknowledged immediately, before the conversation
exists. The real outcome arrives moments later as a follow-up message starting
with "${CHILD_CONVERSATION_RESULT_PREFIX.trim()}" that carries the child's
conversation id, URL, target and initial status — or an error with guidance.
Wait for it, then tell the user what you launched and give them the URL. If it
reports an error, fix the parameters and call this tool again.`;

export const LAUNCH_CHILD_CONVERSATION_CLIENT_TOOL: ClientToolSpec = {
  name: LAUNCH_CHILD_CONVERSATION_TOOL_NAME,
  description: LAUNCH_CHILD_CONVERSATION_DESCRIPTION,
  parameters: {
    type: "object",
    additionalProperties: false,
    properties: {
      target: {
        type: "string",
        enum: [...CHILD_CONVERSATION_TARGETS],
        description:
          "Where the child runs. 'local' reuses this machine and this conversation's workspace; 'cloud' runs in an isolated OpenHands Cloud sandbox.",
      },
      task: {
        type: "string",
        description:
          "Self-contained task brief, sent as the child's first message. The child cannot see this conversation, so state the goal, constraints, expected output and handoff criteria.",
      },
      title: {
        type: "string",
        description:
          "Optional short title for the child conversation. Omit to let it title itself from the task.",
      },
      repository: {
        type: "string",
        description:
          "target='cloud' only. Repository to clone into the sandbox, as 'owner/repo'. Defaults to this conversation's repository when it has one.",
      },
      branch: {
        type: "string",
        description:
          "target='cloud' only, and only together with 'repository'. Defaults to the repository's default branch.",
      },
      isolation: {
        type: "string",
        enum: [...CHILD_CONVERSATION_ISOLATIONS],
        description:
          "target='local' only. 'worktree' (default) gives the child its own git worktree and branch; 'shared' runs it in this conversation's exact directory.",
      },
    },
    required: ["target", "task"],
  },
  annotations: {
    // Launching a conversation writes real state and is not repeatable, so the
    // agent is asked to predict a security risk before every call.
    readOnlyHint: false,
    destructiveHint: false,
    idempotentHint: false,
    // The cloud target reaches OpenHands Cloud.
    openWorldHint: true,
  },
};
