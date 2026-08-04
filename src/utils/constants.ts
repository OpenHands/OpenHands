import type { SlashCommandItem } from "#/types/slash-command";
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

export const HELP_COMMAND = "/help";
export const FEEDBACK_COMMAND = "/feedback";
export const SKILLS_COMMAND = "/skills";
export const CONDENSE_COMMAND = "/condense";

export const FEEDBACK_FORM_URL = "https://forms.gle/chHc5VdS3wty5DwW6";

/** Built-in slash commands surfaced in the menu for V1 conversations. */
export const BUILT_IN_COMMANDS: SlashCommandItem[] = [
  {
    skill: {
      name: "new",
      type: "agentskills",
      content: "Creates a new conversation using the same runtime",
      triggers: ["/new"],
    },
    command: "/new",
    descriptionKey: I18nKey.SLASH_COMMAND$NEW_DESCRIPTION,
  },
  {
    skill: {
      name: "btw",
      type: "agentskills",
      content: "Ask the agent a side question without derailing the main task",
      triggers: [BTW_COMMAND],
    },
    command: BTW_COMMAND,
    descriptionKey: I18nKey.SLASH_COMMAND$BTW_DESCRIPTION,
  },
  {
    skill: {
      name: "model",
      type: "agentskills",
      content:
        "List saved LLM profiles, or switch the conversation LLM profile with /model <name>",
      triggers: [MODEL_COMMAND],
    },
    command: MODEL_COMMAND,
    descriptionKey: I18nKey.SLASH_COMMAND$MODEL_DESCRIPTION,
  },
  {
    skill: {
      name: "goal",
      type: "agentskills",
      content:
        "Drive the agent toward an objective until a judge says it's done — /goal <objective> or /goal --max <n> <objective>",
      triggers: [GOAL_COMMAND],
    },
    command: GOAL_COMMAND,
    descriptionKey: I18nKey.SLASH_COMMAND$GOAL_DESCRIPTION,
  },
  {
    skill: {
      name: "help",
      type: "agentskills",
      content: "List all available slash commands",
      triggers: [HELP_COMMAND],
    },
    command: HELP_COMMAND,
    descriptionKey: I18nKey.SLASH_COMMAND$HELP_DESCRIPTION,
  },
  {
    skill: {
      name: "feedback",
      type: "agentskills",
      content: "Open the anonymous feedback form",
      triggers: [FEEDBACK_COMMAND],
    },
    command: FEEDBACK_COMMAND,
    descriptionKey: I18nKey.SLASH_COMMAND$FEEDBACK_DESCRIPTION,
  },
  {
    skill: {
      name: "skills",
      type: "agentskills",
      content: "List skills available in the current workspace",
      triggers: [SKILLS_COMMAND],
    },
    command: SKILLS_COMMAND,
    descriptionKey: I18nKey.SLASH_COMMAND$SKILLS_DESCRIPTION,
  },
  {
    skill: {
      name: "condense",
      type: "agentskills",
      content: "Condense the current conversation history",
      triggers: [CONDENSE_COMMAND],
    },
    command: CONDENSE_COMMAND,
    descriptionKey: I18nKey.SLASH_COMMAND$CONDENSE_DESCRIPTION,
  },
];

// Skill content metadata prefixes
export const METADATA_PREFIXES: readonly string[] = [
  "The following information has been included",
  "It may or may not be relevant",
  "Skill location:",
  "(Use this path to resolve",
];
