import { CANVAS_UI_CLIENT_TOOL_NAME } from "#/constants/canvas-ui";

export {
  CANVAS_UI_CLIENT_ACTION_KIND,
  CANVAS_UI_CLIENT_TOOL_NAME,
  LEGACY_CANVAS_UI_TOOL_NAME,
} from "#/constants/canvas-ui";

export interface ClientToolSpec {
  name: string;
  description: string;
  parameters: Record<string, unknown>;
  annotations?: {
    title?: string | null;
    readOnlyHint: boolean;
    destructiveHint: boolean;
    idempotentHint: boolean;
    openWorldHint: boolean;
  };
}

const CANVAS_UI_DESCRIPTION = `The user is interacting with you inside Agent Canvas — a web UI with a chat panel on the left and a tabbed right-side panel (files, terminal, browser, vscode, planner, tasklist). This tool lets you drive that right-side panel so the user sees what you just produced.

They will NOT see the files you wrote, the terminal output, or the browser
unless you call this tool to switch the right-side panel to the relevant
tab. Call this every time you finish work that produces something the user
should look at — don't rely on them noticing on their own.

When to call (pick the most specific option that matches your last action):

* You wrote or modified a single file (ANY language, ANY size — including
  small scripts like a hello-world bash file) →
    command="navigate_to_file", path=<workspace-relative path of that file>

* You generated an HTML page, image, SVG, PDF, markdown report, or other
  previewable artifact →
    command="show_preview", path=<that file>

* You finished editing multiple files in one logical step →
    command="open_tab", tab="files"
    (The Files tab automatically renders a diff view when the workspace has
    uncommitted git changes, which covers the "highlight changes" case.)

* You ran a long-running terminal command, or one whose output the user
  should inspect →
    command="open_tab", tab="terminal"

* You browsed to a URL the user should see →
    First call browser_get_state(include_screenshot=true) after your final
    browser interaction so Agent Canvas has a screenshot to display, then call
    command="open_tab", tab="browser"
    (browser_navigate alone only updates the URL; without browser_get_state,
    the Browser tab will open without a screenshot.)

Call this BEFORE writing your chat-message summary of the change, so the
artifact is visible while the user reads what you did. One canvas_ui_control
call per logical step is enough — don't repeat it for the same file or tab in
the same turn.`;

export const CANVAS_UI_CLIENT_TOOL: ClientToolSpec = {
  name: CANVAS_UI_CLIENT_TOOL_NAME,
  description: CANVAS_UI_DESCRIPTION,
  parameters: {
    type: "object",
    additionalProperties: false,
    properties: {
      command: {
        type: "string",
        enum: ["navigate_to_file", "open_tab", "show_preview"],
        description: "UI command to dispatch.",
      },
      path: {
        type: "string",
        description:
          "Workspace-relative file path. Required for navigate_to_file and show_preview; ignored otherwise.",
      },
      tab: {
        type: "string",
        enum: ["files", "browser", "vscode", "terminal", "planner", "tasklist"],
        description: "Tab to open. Required for open_tab; ignored otherwise.",
      },
    },
    required: ["command"],
  },
  annotations: {
    readOnlyHint: true,
    destructiveHint: false,
    idempotentHint: true,
    openWorldHint: false,
  },
};
