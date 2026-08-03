import type { BuiltInSlashCommandItem } from "#/types/slash-command";
import { I18nKey } from "#/i18n/declaration";

export const JSON_VIEW_THEME = {
  base00: "transparent", // background
  base01: "var(--cool-grey-900)", // lighter background
  base02: "var(--cool-grey-700)", // selection background
  base03: "var(--cool-grey-600)", // comments, invisibles
  base04: "var(--cool-grey-500)", // dark foreground
  base05: "var(--cool-grey-200)", // default foreground
  base06: "var(--cool-grey-100)", // light foreground
  base07: "#ffffff", // light background
  base08: "#ff5370", // variables, red
  base09: "#f78c6c", // integers, orange
  base0A: "#ffcb6b", // booleans, yellow
  base0B: "#c3e88d", // strings, green
  base0C: "#89ddff", // support, cyan
  base0D: "#82aaff", // functions, blue
  base0E: "#c792ea", // keywords, purple
  base0F: "#ff5370", // deprecated, red
};

export const PRODUCT_URL = {
  PRODUCTION: "https://app.all-hands.dev",
};

export const SETTINGS_FORM = {
  LABEL_CLASSNAME: "text-[11px] font-medium leading-4 tracking-[0.11px]",
};

// Chat input constants
export const CHAT_INPUT = {
  HEIGHT_THRESHOLD: 100, // Height in pixels when suggestions should be hidden
};

// UI tolerance constants
export const EPS = 1.5; // px tolerance for "near min" height comparisons

/** The /btw slash command — asks a side question via the ask_agent endpoint. */
export const BTW_COMMAND = "/btw";

/** The /model slash command — lists or switches the conversation's LLM profile. */
export const MODEL_COMMAND = "/model";

/** The /goal slash command — drives the agent toward an objective, judging completion each round. */
export const GOAL_COMMAND = "/goal";

export const NEW_COMMAND = "/new";
export const HELP_COMMAND = "/help";
export const HISTORY_COMMAND = "/history";
export const SETTINGS_COMMAND = "/settings";
export const CONFIRM_COMMAND = "/confirm";
export const CONDENSE_COMMAND = "/condense";
export const SKILLS_COMMAND = "/skills";
export const FEEDBACK_COMMAND = "/feedback";
export const FORK_COMMAND = "/fork";

/**
 * Commands shared with OpenHands CLI help, in the order used by the CLI.
 * `/exit` is intentionally omitted because Canvas runs in the browser.
 */
export const CLI_HELP_COMMAND_ORDER = [
  HELP_COMMAND,
  NEW_COMMAND,
  HISTORY_COMMAND,
  SETTINGS_COMMAND,
  CONFIRM_COMMAND,
  CONDENSE_COMMAND,
  SKILLS_COMMAND,
  FEEDBACK_COMMAND,
] as const;

/** Built-in slash commands surfaced in the menu for V1 conversations. */
export const BUILT_IN_COMMANDS: BuiltInSlashCommandItem[] = [
  {
    skill: {
      name: "help",
      type: "agentskills",
      source: null,
      triggers: [HELP_COMMAND],
    },
    command: HELP_COMMAND,
    availability: "always",
    descriptionKey: I18nKey.SLASH_COMMAND$HELP_DESCRIPTION,
  },
  {
    skill: {
      name: "new",
      type: "agentskills",
      source: null,
      triggers: [NEW_COMMAND],
    },
    command: NEW_COMMAND,
    availability: "conversation",
    descriptionKey: I18nKey.SLASH_COMMAND$NEW_DESCRIPTION,
  },
  {
    skill: {
      name: "history",
      type: "agentskills",
      source: null,
      triggers: [HISTORY_COMMAND],
    },
    command: HISTORY_COMMAND,
    availability: "always",
    descriptionKey: I18nKey.SLASH_COMMAND$HISTORY_DESCRIPTION,
  },
  {
    skill: {
      name: "settings",
      type: "agentskills",
      source: null,
      triggers: [SETTINGS_COMMAND],
    },
    command: SETTINGS_COMMAND,
    availability: "always",
    descriptionKey: I18nKey.SLASH_COMMAND$SETTINGS_DESCRIPTION,
  },
  {
    skill: {
      name: "confirm",
      type: "agentskills",
      source: null,
      triggers: [CONFIRM_COMMAND],
    },
    command: CONFIRM_COMMAND,
    availability: "confirmation",
    descriptionKey: I18nKey.SLASH_COMMAND$CONFIRM_DESCRIPTION,
  },
  {
    skill: {
      name: "condense",
      type: "agentskills",
      source: null,
      triggers: [CONDENSE_COMMAND],
    },
    command: CONDENSE_COMMAND,
    availability: "manual-condensation-conversation",
    descriptionKey: I18nKey.SLASH_COMMAND$CONDENSE_DESCRIPTION,
  },
  {
    skill: {
      name: "skills",
      type: "agentskills",
      source: null,
      triggers: [SKILLS_COMMAND],
    },
    command: SKILLS_COMMAND,
    availability: "always",
    descriptionKey: I18nKey.SLASH_COMMAND$SKILLS_DESCRIPTION,
  },
  {
    skill: {
      name: "feedback",
      type: "agentskills",
      source: null,
      triggers: [FEEDBACK_COMMAND],
    },
    command: FEEDBACK_COMMAND,
    availability: "always",
    descriptionKey: I18nKey.SLASH_COMMAND$FEEDBACK_DESCRIPTION,
  },
  {
    skill: {
      name: "fork",
      type: "agentskills",
      source: null,
      triggers: [FORK_COMMAND],
    },
    command: FORK_COMMAND,
    availability: "local-openhands-conversation",
    descriptionKey: I18nKey.SLASH_COMMAND$FORK_DESCRIPTION,
  },
  {
    skill: {
      name: "btw",
      type: "agentskills",
      source: null,
      triggers: [BTW_COMMAND],
    },
    command: BTW_COMMAND,
    availability: "conversation",
    descriptionKey: I18nKey.SLASH_COMMAND$BTW_DESCRIPTION,
  },
  {
    skill: {
      name: "model",
      type: "agentskills",
      source: null,
      triggers: [MODEL_COMMAND],
    },
    command: MODEL_COMMAND,
    availability: "always",
    descriptionKey: I18nKey.SLASH_COMMAND$MODEL_DESCRIPTION,
  },
  {
    skill: {
      name: "goal",
      type: "agentskills",
      source: null,
      triggers: [GOAL_COMMAND],
    },
    command: GOAL_COMMAND,
    availability: "conversation",
    descriptionKey: I18nKey.SLASH_COMMAND$GOAL_DESCRIPTION,
  },
];

// Skill content metadata prefixes
export const METADATA_PREFIXES: readonly string[] = [
  "The following information has been included",
  "It may or may not be relevant",
  "Skill location:",
  "(Use this path to resolve",
];
