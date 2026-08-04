import type { OHEvent } from "#/stores/use-event-store";
import {
  isActionEvent,
  isObservationEvent,
} from "#/types/agent-server/type-guards";
import type { AgentNotification } from "./agent-notifications.constants";

const SUBSTANTIVE_FILE_COMMANDS = new Set(["create", "str_replace", "insert"]);

const FILE_VIEW_COMMAND = "view";
const MIN_VIEW_PATHS_FOR_SKILL = 3;

const FILE_EDITOR_TOOL_NAMES = new Set(["file_editor", "str_replace_editor"]);
const TERMINAL_TOOL_NAMES = new Set(["terminal", "execute_bash"]);

const TEST_COMMAND_PATTERN =
  /\b(npm\s+(run\s+)?test|pnpm\s+test|yarn\s+test|pytest|vitest|jest)\b/i;

const CI_API_COMMAND_PATTERN =
  /\b(gh\s|github\s|curl\s+.*\b(api|webhook)\b|webhook)\b/i;

function slugify(value: string): string {
  return value
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

function humanizeBasename(path: string): string {
  const basename = path.split("/").pop() ?? path;
  const withoutExt = basename.replace(/\.[^.]+$/, "");
  return withoutExt
    .split(/[-_.]+/)
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function recordFileEdit(
  editedPaths: string[],
  command: string | undefined,
  path: string | null | undefined,
) {
  if (
    command &&
    path &&
    SUBSTANTIVE_FILE_COMMANDS.has(
      command as "create" | "str_replace" | "insert",
    )
  ) {
    editedPaths.push(path);
  }
}

function recordFileView(
  viewedPaths: string[],
  command: string | undefined,
  path: string | null | undefined,
) {
  if (command === FILE_VIEW_COMMAND && path) {
    viewedPaths.push(path);
  }
}

function recordTerminalCommand(
  bashCommands: string[],
  command: string | null | undefined,
  isInput = false,
) {
  const trimmed = command?.trim();
  if (trimmed && !isInput) {
    bashCommands.push(trimmed);
  }
}

function collectConversationSignals(events: OHEvent[]) {
  const editedPaths: string[] = [];
  const viewedPaths: string[] = [];
  const bashCommands: string[] = [];

  for (const event of events) {
    if (isActionEvent(event)) {
      const { action, tool_name: toolName } = event;

      if (
        action.kind === "FileEditorAction" ||
        action.kind === "StrReplaceEditorAction" ||
        FILE_EDITOR_TOOL_NAMES.has(toolName)
      ) {
        if ("command" in action && "path" in action) {
          recordFileEdit(editedPaths, action.command, action.path);
          recordFileView(viewedPaths, action.command, action.path);
        }
        continue;
      }

      if (
        action.kind === "ExecuteBashAction" ||
        action.kind === "TerminalAction" ||
        TERMINAL_TOOL_NAMES.has(toolName)
      ) {
        if ("command" in action) {
          recordTerminalCommand(
            bashCommands,
            typeof action.command === "string" ? action.command : null,
            "is_input" in action ? Boolean(action.is_input) : false,
          );
        }
      }
      continue;
    }

    if (!isObservationEvent(event)) {
      continue;
    }

    const { observation } = event;

    if (
      observation.kind === "FileEditorObservation" ||
      observation.kind === "StrReplaceEditorObservation"
    ) {
      recordFileEdit(editedPaths, observation.command, observation.path);
      recordFileView(viewedPaths, observation.command, observation.path);
      continue;
    }

    if (
      observation.kind === "TerminalObservation" ||
      observation.kind === "ExecuteBashObservation"
    ) {
      recordTerminalCommand(bashCommands, observation.command);
    }
  }

  return {
    editedPaths: [...new Set(editedPaths)],
    viewedPaths: [...new Set(viewedPaths)],
    bashCommands,
  };
}

function pickSkillPaths(
  editedPaths: string[],
  viewedPaths: string[],
): string[] {
  if (editedPaths.length > 0) {
    return editedPaths;
  }

  if (viewedPaths.length >= MIN_VIEW_PATHS_FOR_SKILL) {
    const filePaths = viewedPaths.filter((path) => /\.[^/]+$/.test(path));
    return filePaths.length > 0 ? filePaths : viewedPaths;
  }

  return [];
}

function buildSkillRecommendation(
  paths: string[],
  fromExploration = false,
): AgentNotification | null {
  if (paths.length === 0) {
    return null;
  }

  const primaryPath = paths[0];
  const label = humanizeBasename(primaryPath);
  const name = `${label} helper`;
  const filesSummary = paths.slice(0, 3).join(", ");

  const prompt = fromExploration
    ? `Save a reusable skill named "${name}" that captures the code exploration ` +
      `workflow from this conversation (including files reviewed such as ${filesSummary}) ` +
      "so I can rerun it in future sessions."
    : `Save a reusable skill named "${name}" that captures the file-editing ` +
      `workflow from this conversation (including work on ${filesSummary}) ` +
      "so I can rerun it in future sessions.";

  return {
    id: `detected-skill-${slugify(label)}`,
    kind: "skill",
    name,
    prompt,
    createdAt: new Date().toISOString(),
  };
}

function buildWorkflowRecommendation(
  commands: string[],
): AgentNotification | null {
  const testCommand = commands.find((command) =>
    TEST_COMMAND_PATTERN.test(command),
  );
  if (!testCommand) {
    return null;
  }

  const trimmedCommand = testCommand.trim();

  return {
    id: "detected-workflow-test-runner",
    kind: "workflow",
    name: "Test runner workflow",
    prompt:
      'Create a workflow named "Test runner workflow" that runs the same test ' +
      `command we used in this conversation (\`${trimmedCommand}\`) on every ` +
      "push and reports failures.",
    createdAt: new Date().toISOString(),
  };
}

function buildResponderRecommendation(
  commands: string[],
): AgentNotification | null {
  const ciCommand = commands.find((command) =>
    CI_API_COMMAND_PATTERN.test(command),
  );
  if (!ciCommand) {
    return null;
  }

  const trimmedCommand = ciCommand.trim().slice(0, 160);

  return {
    id: "detected-responder-ci-watchdog",
    kind: "responder",
    name: "CI/API watchdog",
    prompt:
      'Create a responder named "CI/API watchdog" that monitors failures like ' +
      `we handled in this conversation (e.g. \`${trimmedCommand}\`) and posts ` +
      "a short diagnosis when triggered.",
    createdAt: new Date().toISOString(),
  };
}

/**
 * MVP heuristic: infer skill/automation recommendations from tool-use events
 * after substantive agent work. Supplements (does not replace) fenced-block
 * output from a future detection skill.
 */
export function detectAgentNotificationsFromEvents(
  events: OHEvent[],
): AgentNotification[] {
  const { editedPaths, viewedPaths, bashCommands } =
    collectConversationSignals(events);
  const skillPaths = pickSkillPaths(editedPaths, viewedPaths);
  const recommendations: AgentNotification[] = [];

  const skill = buildSkillRecommendation(
    skillPaths,
    editedPaths.length === 0 && skillPaths.length > 0,
  );
  if (skill) {
    recommendations.push(skill);
  }

  const workflow = buildWorkflowRecommendation(bashCommands);
  if (workflow) {
    recommendations.push(workflow);
  } else {
    const responder = buildResponderRecommendation(bashCommands);
    if (responder) {
      recommendations.push(responder);
    }
  }

  return recommendations.slice(0, 2);
}
